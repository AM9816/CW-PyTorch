import inspect
from time import time as now
import torch
from torch import nn, Tensor as T
from tqdm import tqdm

from torch_common import getDevices, holdoutTest
from attacks import _FGSM_linf

cpu, gpu = getDevices()


def computeLpDistances(x:T, adv:T, independentChannels=False):
    _delta = (adv - x).detach()
    if independentChannels or _delta.ndim < 3:
        _l0 = (_delta.abs() > 1e-6).flatten(1).sum(dim=1).float()
    else:
        _l0 = (_delta.abs().sum(dim=1) > 1e-6).flatten(1).sum(dim=1).float()

    _l2 = _delta.flatten(1).pow(2).sum(dim=1).sqrt()
    _linf = _delta.flatten(1).abs().max(dim=1).values
    return {"l0": _l0, "l2": _l2, "linf": _linf}


def _inferPrimaryMetric(attack):
    _name = str(attack).lower()
    if "l0" in _name:
        return "l0"
    if "linf" in _name:
        return "linf"
    return "l2"


def _callAttack(attack, x:T, y:T, targets:T = None, verbose=True):
    if hasattr(attack, "attack"):
        _sig = inspect.signature(attack.attack)
        if "verbose" in _sig.parameters:
            return attack.attack(x, y, targets, verbose=verbose)

    _sig = inspect.signature(attack.__call__)
    if len(_sig.parameters) >= 3 or any(
        p.kind == inspect.Parameter.VAR_POSITIONAL for p in _sig.parameters.values()
    ):
        return attack(x, y, targets)

    if targets is not None and isinstance(attack, _FGSM_linf):
        _device = x.device
        _model = attack.model.eval().to(_device)
        _xVar = x.clone().detach().to(_device).requires_grad_(True)
        _out = _model(_xVar)
        _loss = attack.lossf(_out, targets.to(_device))
        _grad = torch.autograd.grad([_loss], _xVar, retain_graph=False)[0]
        return (_xVar - attack.eps * _grad.sign()).clamp(0, 1).detach()

    return attack(x, y)


def randomIncorrectTargets(y:T, numClasses=10):
    _offsets = torch.randint(1, numClasses, y.shape, device=y.device)
    return (y + _offsets) % numClasses


def allIncorrectTargets(y:T, numClasses=10):
    _bsize = y.size(0)
    _classes = torch.arange(numClasses, device=y.device).unsqueeze(0).expand(_bsize, numClasses)
    _mask = _classes != y.unsqueeze(1)
    return _classes[_mask].view(_bsize, numClasses - 1)


def _summarizeDistortions(succMask:T, l0:T, l2:T, linf:T):
    if not succMask.any():
        return float("nan"), float("nan"), float("nan")
    return (
        l0[succMask].mean().item(),
        l2[succMask].mean().item(),
        linf[succMask].mean().item(),
    )


def _performAdversarialTest_BestWorst(
    model:nn.Module,
    attacks,
    dataset,
    mode="all",
    numClasses=10,
    primaryMetric=None,
    verbose=True,
    device=gpu,
):
    assert mode in ("untargeted", "average", "best", "worst", "all")

    model = model.eval().to(device)
    _start = now()

    _, _cleanAccuracy = holdoutTest(model, dataset, lambda a, b: 0, verbose, device)
    model.eval()
    _cleanAccVal = _cleanAccuracy.item() if isinstance(_cleanAccuracy, T) else float(_cleanAccuracy)

    _results = {
        f"clean {str(dataset)} with {str(model)}": {
            "accuracy": _cleanAccVal,
            "success": 0.0,
            "mean_l0": 0.0,
            "mean_l2": 0.0,
            "mean_linf": 0.0,
        }
    }

    if verbose:
        print(f"clean accuracy = {_cleanAccVal}")

    for _attack in attacks:
        _distMetric = primaryMetric or _inferPrimaryMetric(_attack)

        _iterator = enumerate(dataset)
        if verbose:
            _iterator = tqdm(_iterator, total=len(dataset))
            _iterator.set_description_str(f"{str(_attack)} ({mode})")

        if mode in ("untargeted", "average"):
            _totalAccuracy = 0.0
            _totalDatapoints = 0
            _allSucc, _allL0, _allL2, _allLinf = [], [], [], []

            for _, (_x, _y) in _iterator:
                _x, _y = _x.to(device), _y.to(device)
                _targets = randomIncorrectTargets(_y, numClasses) if mode == "average" else None

                _adv = _callAttack(_attack, _x, _y, _targets, verbose=verbose)
                with torch.no_grad():
                    _advPreds = model(_adv).argmax(dim=-1)
                    _advAcc = (_advPreds == _y).float().mean().item()
                    _succ = (_advPreds == _targets) if _targets is not None else (_advPreds != _y)
                    _dists = computeLpDistances(_x, _adv)

                _totalAccuracy += _advAcc * _x.size(0)
                _totalDatapoints += _x.size(0)
                _allSucc.append(_succ.cpu())
                _allL0.append(_dists["l0"].cpu())
                _allL2.append(_dists["l2"].cpu())
                _allLinf.append(_dists["linf"].cpu())

            _succCat = torch.cat(_allSucc)
            _l0Cat, _l2Cat, _linfCat = torch.cat(_allL0), torch.cat(_allL2), torch.cat(_allLinf)
            _meanL0, _meanL2, _meanLinf = _summarizeDistortions(_succCat, _l0Cat, _l2Cat, _linfCat)
            _avgAccuracy = _totalAccuracy / _totalDatapoints
            _avgSuccess = _succCat.float().mean().item()

            _caseLabel = "untargeted" if mode == "untargeted" else "average_case"
            _key = f"{str(_attack)} [{_caseLabel}] on {str(dataset)} with {str(model)}"
            _results[_key] = {
                "accuracy": _avgAccuracy,
                "success": _avgSuccess,
                "mean_l0": _meanL0,
                "mean_l2": _meanL2,
                "mean_linf": _meanLinf,
                f"mean_{_distMetric}": {"l0": _meanL0, "l2": _meanL2, "linf": _meanLinf}[_distMetric],
            }

            if verbose:
                print(
                    f"{_key}: acc = {_avgAccuracy:.2%}, success = {_avgSuccess:.2%}, "
                    f"mean L0 = {_meanL0:.2f}, mean L2 = {_meanL2:.4f}, mean Linf = {_meanLinf:.4f}"
                )

        else:
            _bestSuccList, _bestL0List, _bestL2List, _bestLinfList = [], [], [], []
            _avgSuccList, _avgL0List, _avgL2List, _avgLinfList = [], [], [], []
            _worstSuccList, _worstL0List, _worstL2List, _worstLinfList = [], [], [], []
            _bestAccCount = _avgAccCount = _worstAccCount = _totalDatapoints = 0

            for _, (_x, _y) in _iterator:
                _x, _y = _x.to(device), _y.to(device)
                _bsize = _x.size(0)
                _targetMatrix = allIncorrectTargets(_y, numClasses)
                _numTargets = _targetMatrix.size(1)

                with torch.no_grad():
                    _cleanPreds = model(_x).argmax(dim=-1)
                    _alreadyWrong = _cleanPreds != _y

                _succMat = torch.zeros((_bsize, _numTargets), dtype=torch.bool, device=device)
                _l0Mat = torch.zeros((_bsize, _numTargets), device=device)
                _l2Mat = torch.zeros((_bsize, _numTargets), device=device)
                _linfMat = torch.zeros((_bsize, _numTargets), device=device)
                _correctMat = torch.zeros((_bsize, _numTargets), dtype=torch.bool, device=device)

                _canVectorize = _bsize * _numTargets <= 2500

                if _canVectorize:
                    _xRep = _x.repeat_interleave(_numTargets, dim=0)
                    _yRep = _y.repeat_interleave(_numTargets, dim=0)
                    _tRep = _targetMatrix.reshape(-1)
                    _advRep = _callAttack(_attack, _xRep, _yRep, _tRep, verbose=verbose)
                    with torch.no_grad():
                        _predsRep = model(_advRep).argmax(dim=-1).view(_bsize, _numTargets)
                        _alreadyTargetMat = _alreadyWrong.unsqueeze(1) & (_cleanPreds.unsqueeze(1) == _targetMatrix)
                        _succMat = (_predsRep == _targetMatrix) | _alreadyTargetMat
                        _correctMat = _predsRep == _y.unsqueeze(1)

                        _distsRep = computeLpDistances(_xRep, _advRep)
                        _l0Mat = torch.where(_alreadyTargetMat, torch.zeros_like(_l0Mat), _distsRep["l0"].view(_bsize, _numTargets))
                        _l2Mat = torch.where(_alreadyTargetMat, torch.zeros_like(_l2Mat), _distsRep["l2"].view(_bsize, _numTargets))
                        _linfMat = torch.where(_alreadyTargetMat, torch.zeros_like(_linfMat), _distsRep["linf"].view(_bsize, _numTargets))
                else:
                    for _k in range(_numTargets):
                        _tK = _targetMatrix[:, _k]
                        _advK = _callAttack(_attack, _x, _y, _tK, verbose=verbose)
                        with torch.no_grad():
                            _predsK = model(_advK).argmax(dim=-1)
                            _alreadyTarget = _alreadyWrong & (_cleanPreds == _tK)
                            _succK = (_predsK == _tK) | _alreadyTarget

                            _distsK = computeLpDistances(_x, _advK)
                            _succMat[:, _k] = _succK
                            _l0Mat[:, _k] = torch.where(_alreadyTarget, torch.zeros_like(_distsK["l0"]), _distsK["l0"])
                            _l2Mat[:, _k] = torch.where(_alreadyTarget, torch.zeros_like(_distsK["l2"]), _distsK["l2"])
                            _linfMat[:, _k] = torch.where(_alreadyTarget, torch.zeros_like(_distsK["linf"]), _distsK["linf"])
                            _correctMat[:, _k] = _predsK == _y

                _primaryMat = {"l0": _l0Mat, "l2": _l2Mat, "linf": _linfMat}[_distMetric]
                _batchIdx = torch.arange(_bsize, device=device)

                _bestScores = _primaryMat.masked_fill(~_succMat, float("inf"))
                _bestIdx = _bestScores.argmin(dim=1)
                _bestSucc = _succMat.any(dim=1)
                _bestSuccList.append(_bestSucc.cpu())
                _bestL0List.append(_l0Mat[_batchIdx, _bestIdx].cpu())
                _bestL2List.append(_l2Mat[_batchIdx, _bestIdx].cpu())
                _bestLinfList.append(_linfMat[_batchIdx, _bestIdx].cpu())
                _bestAccCount += _correctMat[_batchIdx, _bestIdx].float().sum().item()

                _randIdx = torch.randint(0, _numTargets, (_bsize,), device=device)
                _avgSucc = _succMat[_batchIdx, _randIdx]
                _avgSuccList.append(_avgSucc.cpu())
                _avgL0List.append(_l0Mat[_batchIdx, _randIdx].cpu())
                _avgL2List.append(_l2Mat[_batchIdx, _randIdx].cpu())
                _avgLinfList.append(_linfMat[_batchIdx, _randIdx].cpu())
                _avgAccCount += _correctMat[_batchIdx, _randIdx].float().sum().item()

                _worstScores = _primaryMat.masked_fill(~_succMat, -float("inf"))
                _worstIdx = _worstScores.argmax(dim=1)
                _worstSucc = _succMat.all(dim=1)
                _worstSuccList.append(_worstSucc.cpu())
                _worstL0List.append(_l0Mat[_batchIdx, _worstIdx].cpu())
                _worstL2List.append(_l2Mat[_batchIdx, _worstIdx].cpu())
                _worstLinfList.append(_linfMat[_batchIdx, _worstIdx].cpu())
                _worstAccCount += _correctMat[_batchIdx, _worstIdx].float().sum().item()

                _totalDatapoints += _bsize

            _casesToReport = (
                [("best_case", _bestSuccList, _bestL0List, _bestL2List, _bestLinfList, _bestAccCount)]
                if mode == "best"
                else [("worst_case", _worstSuccList, _worstL0List, _worstL2List, _worstLinfList, _worstAccCount)]
                if mode == "worst"
                else [
                    ("best_case", _bestSuccList, _bestL0List, _bestL2List, _bestLinfList, _bestAccCount),
                    ("average_case", _avgSuccList, _avgL0List, _avgL2List, _avgLinfList, _avgAccCount),
                    ("worst_case", _worstSuccList, _worstL0List, _worstL2List, _worstLinfList, _worstAccCount),
                ]
            )

            for _caseName, _sList, _l0List, _l2List, _linfList, _accCnt in _casesToReport:
                _sCat = torch.cat(_sList)
                _l0Cat, _l2Cat, _linfCat = torch.cat(_l0List), torch.cat(_l2List), torch.cat(_linfList)
                _meanL0, _meanL2, _meanLinf = _summarizeDistortions(_sCat, _l0Cat, _l2Cat, _linfCat)
                _avgAccuracy = _accCnt / _totalDatapoints
                _avgSuccess = _sCat.float().mean().item()

                _key = f"{str(_attack)} [{_caseName}] on {str(dataset)} with {str(model)}"
                _results[_key] = {
                    "accuracy": _avgAccuracy,
                    "success": _avgSuccess,
                    "mean_l0": _meanL0,
                    "mean_l2": _meanL2,
                    "mean_linf": _meanLinf,
                    f"mean_{_distMetric}": {"l0": _meanL0, "l2": _meanL2, "linf": _meanLinf}[_distMetric],
                }

                if verbose:
                    print(
                        f"{_key}: acc = {_avgAccuracy:.2%}, success = {_avgSuccess:.2%}, "
                        f"mean L0 = {_meanL0:.2f}, mean L2 = {_meanL2:.4f}, mean Linf = {_meanLinf:.4f}"
                    )

    if verbose:
        print(f"tests done, time elapsed = {now() - _start:.2f}s")

    return _results
