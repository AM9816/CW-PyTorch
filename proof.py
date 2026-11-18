import os
import sys
import json
from time import time as now, strftime

if "__file__" in globals():
    _SRC_DIR = os.path.dirname(os.path.abspath(__file__))
else:
    _SRC_DIR = os.getcwd()

if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import torch
from torch import nn
import torch.nn.functional as F

from torch_common import (
    getDevices,
    loadModel,
    holdoutTest,
    SmallDataLoader,
)
from classifiers import CW_CNN, MNIST_CNN, getCIFAR, getMNIST, train
from distill import _trainStudent
from attacks import _FGSM_linf, _CW_L2, _CW_Linf
from cw_other import (
    _CW_L0_UNSTABLE,
    _CW_L2_MODULAR_UNSTABLE,
    _BIM,
    _greedySearchForL0,
)
from extra_tests import (
    _performAdversarialTest_BestWorst,
    computeLpDistances,
    randomIncorrectTargets,
)

cpu, gpu = getDevices()


# class for logging prints in case program crashes
class _TeeLogger:
    def __init__(self, logPath):
        self._logPath = logPath
        os.makedirs(os.path.dirname(
            os.path.abspath(logPath)), exist_ok=True)
        self._stdout = sys.stdout
        self._file = open(logPath, "w", encoding="utf-8")

    def write(self, data):
        self._stdout.write(data)
        self._stdout.flush()
        self._file.write(data)
        self._file.flush()

    def flush(self):
        self._stdout.flush(); self._file.flush()

    def close(self):
        sys.stdout = self._stdout; self._file.close()


def _saveFindingsJson(data, jsonPath):
    os.makedirs(os.path.dirname(os.path.abspath(jsonPath)), exist_ok=True)
    with open(jsonPath, "w", encoding="utf-8") as _f:
        json.dump(data, _f, indent=2, default=lambda o: float(o) if hasattr(o, "item") else str(o))


def _getCorrect(
    models:list[nn.Module],
    dataloader, n = 64, bsize = 64,
    name = "Dataset", device = gpu,
):
    _xs, _ys = [], []
    _count = 0
    for _m in models:
        _m.eval().to(device)

    with torch.no_grad():
        for _xBatch, _yBatch in dataloader:
            _xBatch, _yBatch = _xBatch.to(device), _yBatch.to(device)
            _mask = torch.ones(_xBatch.size(0), dtype=torch.bool, device=device)
            for _m in models:
                _preds = _m(_xBatch).argmax(dim=-1)
                _mask = _mask & (_preds == _yBatch)

            if _mask.any():
                _xs.append(_xBatch[_mask].cpu())
                _ys.append(_yBatch[_mask].cpu())
                _count += int(_mask.sum().item())
                if _count >= n:
                    break

    _x = torch.cat(_xs, dim=0)[:n]
    _y = torch.cat(_ys, dim=0)[:n]
    _batches = [
        (_x[i:i+bsize], _y[i:i+bsize])
        for i in range(0, _x.size(0), bsize)
    ]
    return SmallDataLoader(_batches, name=name)


def _metric(res, find, box):
    for _k, _v in res.items():
        if "clean" in _k:
            continue
        if find in _k and f"[{box}]" in _k:
            return _v
    raise KeyError(f"Could not find ({find!r}, {box!r}) in results keys: {list(res.keys())}")


def _compareFs_optims(
    mnistT1:nn.Module,
    cifarT1:nn.Module,
    _mnist, _cifar,
    optimSteps = 1000, searchN = 10,
    verbose = True,
):
    print("OBJECTIVE FUNCTIONS + BOX CONSTRAINTS ")

    _names = ["f1", "f2", "f3", "f4", "f5", "f6", "f7"]
    _methods = ["change_of_variables", "clipped", "projected"]

    torch.manual_seed(0)
    _attacks = [
        _CW_L2_MODULAR_UNSTABLE(
            mnistT1,
            boxMethod="change_of_variables",
            lossf=fn,
            optimSteps=optimSteps,
            searchN=searchN,
            cStart=1e-2,
        )
        for fn in _names
    ]

    _res = _performAdversarialTest_BestWorst(
        mnistT1, _attacks, _mnist, mode="all", verbose=verbose, device=gpu
    )

    print("\n f1..f7 with Change of Variables on MNIST")

    _objs = {}
    for fn in _names:
        _objs[fn] = {}
        _cells = []
        _tag = f"CW (L2, {fn}, change_of_variables)"
        for _case in ["best_case", "average_case", "worst_case"]:
            _m = _metric(_res, _tag, _case)
            _objs[fn][_case] = _m
            _cells.append(f"{_m['mean_l2']:>6.2f} ({_m['success']*100:>5.1f}%)")
        print(f"{fn:<6} || {_cells[0]:^28} || {_cells[1]:^28} || {_cells[2]:^28}")

    torch.manual_seed(0)
    _attacks = [
        _CW_L2_MODULAR_UNSTABLE(
            mnistT1,
            boxMethod=bm,
            lossf="f6",
            optimSteps=optimSteps,
            searchN=searchN,
            cStart=1e-2,
        )
        for bm in ["clipped", "projected"]
    ]

    _res2 = _performAdversarialTest_BestWorst(
        mnistT1, _attacks, _mnist, mode="all", verbose=verbose, device=gpu
    )
    _resMergedBox = {**_res, **_res2}

    print("\nBox Constraint Encodings with f6 on MNIST")

    _box2 = {}
    for bm in _methods:
        _box2[bm] = {}
        _cells = []
        _tag = f"CW (L2, f6, {bm})"
        for _case in ["best_case", "average_case", "worst_case"]:
            _m = _metric(_resMergedBox, _tag, _case)
            _box2[bm][_case] = _m
            _cells.append(f"{_m['mean_l2']:>6.2f} ({_m['success']*100:>5.1f}%)")
        print(f"{bm:<22} || {_cells[0]:^28} || {_cells[1]:^28} || {_cells[2]:^28}")

    torch.manual_seed(0)
    _attacks = [
        _CW_L2_MODULAR_UNSTABLE(
            cifarT1,
            boxMethod="change_of_variables",
            lossf=fn,
            optimSteps=optimSteps,
            searchN=searchN,
            cStart=1e-2,
        )
        for fn in _names
    ]
    _res3 = _performAdversarialTest_BestWorst(
        cifarT1, _attacks, _cifar, mode="average", verbose=verbose, device=gpu
    )

    print("\nCIFAR-10 Objective Functions")
    _metrics = {}
    for fn in _names:
        _m = _metric(_res3, f"CW (L2, {fn}, change_of_variables)", "average_case")
        _metrics[fn] = _m
        print(f"{fn:<12} | {_m['success']*100:>15.1f}%  | {_m['mean_l2']:>18.4f}")

    torch.manual_seed(0)
    _attacks2 = [
        _CW_L2_MODULAR_UNSTABLE(cifarT1, solver=s, lossf="f6", optimSteps=optimSteps, searchN=searchN, cStart=1e-2)
        for s in ["momentum", "sgd"]
    ]
    _res4 = _performAdversarialTest_BestWorst(
        cifarT1, _attacks2, _cifar, mode="average", verbose=verbose, device=gpu
    )
    _resMerged = {**_res3, **_res4}
    print("\nOptimizers on CIFAR-10")
    _metrics2 = {}
    for s in ["adam", "momentum", "sgd"]:
        _tag = "CW (L2, f6, change_of_variables)" if s == "adam" else f"CW (L2, f6, change_of_variables, {s})"
        _m = _metric(_resMerged, _tag, "average_case")
        _metrics2[s] = _m
        print(f"  {s:<10} | {_m['success']*100:>6.1f}% | {_m['mean_l2']:>8.4f}")

    _xs, _ys = next(iter(_cifar))
    _xs, _ys = _xs.to(gpu), _ys.to(gpu)
    torch.manual_seed(0)
    _ts = randomIncorrectTargets(_ys, numClasses=10)
    _attack = _CW_L2_MODULAR_UNSTABLE(cifarT1, lossf="f6", optimSteps=optimSteps, searchN=searchN, greedyDiscretize=False)
    _resOut = _attack.attack(_xs, _ys, _ts, verbose=verbose)
    _adv = (_resOut * 255.0).round().clamp(0.0, 255.0) / 255.0
    _advGreedy = _greedySearchForL0(cifarT1, _xs, _resOut, _ts, targetted=True)

    with torch.no_grad():
        _succ = (cifarT1(_resOut).argmax(-1) == _ts).float().mean().item()
        _total = (cifarT1(_adv).argmax(-1) == _ts).float().mean().item()
        _succGreedy = (cifarT1(_advGreedy).argmax(-1) == _ts).float().mean().item()

    print(
        f"\nDiscretization\n"
        f"  Continuous: {_succ*100:.1f}% -> Rounded: {_total*100:.1f}% -> Greedy: {_succGreedy*100:.1f}%"
    )

    _f6 = _objs["f6"]["average_case"]
    _f1 = _objs["f1"]["average_case"]
    _f6CIFAR = _metrics["f6"]

    _f6Out = (
        _f6["success"] >= 0.90
        and _f6CIFAR["success"] >= 0.90
        and _f6["mean_l2"] <= _f1["mean_l2"]
        and _succGreedy >= _total
    )

    return {
        "table_iii_objectives": _objs,
        "table_iii_box_methods": _box2,
        "cifar_objectives": _metrics,
        "discretization": {"continuous": _succ, "rounded": _total, "greedy": _succGreedy},
        "passed": _f6Out,
    }


def _compareNorms_manifold(
    mnistT1:nn.Module,
    cifarT1:nn.Module,
    _mnist, _cifar,
    l2Steps = 1000, l2Search = 20,
    linfSteps = 150, l0Steps = 100,
    verbose = True,
):
    print("\nNORMS + SYNTHETIC DIGITS")

    torch.manual_seed(0)
    _mnistAttacks = [
        _CW_L0_UNSTABLE(mnistT1, optimSteps=l0Steps, cStart=1e-2, cEnd=1e4),
        _CW_L2(mnistT1, optimSteps=l2Steps, searchN=l2Search, cStart=1e-2),
        _CW_Linf(mnistT1, optimSteps=linfSteps, cStart=1e-2, cEnd=1e4),
        _FGSM_linf(mnistT1, eps=0.2),
        _BIM(mnistT1, eps=0.2, alpha=1 / 255, steps=40),
    ]
    _mnistRes = _performAdversarialTest_BestWorst(
        mnistT1, _mnistAttacks, _mnist, mode="all", verbose=verbose, device=gpu
    )

    torch.manual_seed(0)
    _cifarAttacks = [
        _CW_L0_UNSTABLE(cifarT1, optimSteps=l0Steps, cStart=1e-2, cEnd=1e4),
        _CW_L2(cifarT1, optimSteps=l2Steps, searchN=l2Search, cStart=1e-2),
        _CW_Linf(cifarT1, optimSteps=linfSteps, cStart=1e-2, cEnd=1e4),
        _FGSM_linf(cifarT1, eps=8 / 255),
        _BIM(cifarT1, eps=8 / 255, alpha=1 / 255, steps=15),
    ]
    _cifarRes = _performAdversarialTest_BestWorst(
        cifarT1, _cifarAttacks, _cifar, mode="all", verbose=verbose, device=gpu
    )

    _rows = [
        ("L0", "CW (L0)", "mean_l0", "{:>6.1f}"),
        ("L2", "CW (L2)", "mean_l2", "{:>6.2f}"),
        ("Linf", "CW (linf)", "mean_linf", "{:>6.3f}"),
        ("FGSM", "FGSM", "mean_linf", "{:>6.3f}"),
        ("IGS", "IGS", "mean_linf", "{:>6.3f}"),
    ]

    print("\nUndistilled MNIST and CIFAR-10")
    for _label, _tag, _metricKey, _fmt in _rows:
        _cells = []
        for _case in ["best_case", "average_case", "worst_case"]:
            _mMn = _metric(_mnistRes, _tag, _case)
            _mCf = _metric(_cifarRes, _tag, _case)
            _mnStr = f"{_fmt.format(_mMn[_metricKey])}({_mMn['success']*100:>3.0f}%)"
            _cfStr = f"{_fmt.format(_mCf[_metricKey])}({_mCf['success']*100:>3.0f}%)"
            _cells.append(f"{_mnStr} / {_cfStr}")
        print(f"{_label:<24} || {_cells[0]:<30} || {_cells[1]:<30} || {_cells[2]:<30}")

    print("\nSynthetic Digits")
    _xBlack = torch.zeros((10, 1, 28, 28), device=gpu)
    _xWhite = torch.ones((10, 1, 28, 28), device=gpu)
    _allTargets = torch.arange(10, device=gpu)

    with torch.no_grad():
        _initBlack = int(mnistT1(_xBlack[:1]).argmax(-1).item())
        _initWhite = int(mnistT1(_xWhite[:1]).argmax(-1).item())

    _synthStats = {}
    for _imgName, _xStart, _initCls in [
        ("Black", _xBlack, _initBlack),
        ("White", _xWhite, _initWhite),
    ]:
        _yDummy = torch.full((10,), _initCls, dtype=torch.long, device=gpu)
        _advL2 = _CW_L2(mnistT1, optimSteps=l2Steps, searchN=l2Search, cStart=1e-2).attack(
            _xStart, _yDummy, targets=_allTargets, verbose=verbose
        )
        _advL0 = _CW_L0_UNSTABLE(mnistT1, optimSteps=l0Steps, cStart=1e-2, cEnd=1e4).attack(
            _xStart, _yDummy, targets=_allTargets, verbose=verbose
        )
        _advLinf = _CW_Linf(mnistT1, optimSteps=linfSteps, cStart=1e-2, cEnd=1e4).attack(
            _xStart, _yDummy, targets=_allTargets, verbose=verbose
        )
        _advL2[_initCls] = _xStart[_initCls]
        _advL0[_initCls] = _xStart[_initCls]
        _advLinf[_initCls] = _xStart[_initCls]

        with torch.no_grad():
            _sL2 = (mnistT1(_advL2).argmax(-1) == _allTargets).float().mean().item()
            _sL0 = (mnistT1(_advL0).argmax(-1) == _allTargets).float().mean().item()
            _sLinf = (mnistT1(_advLinf).argmax(-1) == _allTargets).float().mean().item()
            _dL2 = computeLpDistances(_xStart, _advL2)["l2"].mean().item()
            _dL0 = computeLpDistances(_xStart, _advL0)["l0"].mean().item()
            _dLinf = computeLpDistances(_xStart, _advLinf)["linf"].mean().item()

        _synthStats[_imgName] = {
            "initial_class": _initCls,
            "l0_succ": _sL0, "mean_l0": _dL0,
            "l2_succ": _sL2, "mean_l2": _dL2,
            "linf_succ": _sLinf, "mean_linf": _dLinf,
        }
        print(
            f"  {_imgName:<10} (init {_initCls}): "
            f"L0={_dL0:>5.1f} ({_sL0*100:.0f}%), L2={_dL2:>5.2f} ({_sL2*100:.0f}%), Linf={_dLinf:>5.3f} ({_sLinf*100:.0f}%)"
        )

    _mnBest = _metric(_mnistRes, "CW (L2)", "best_case")
    _mnAvg = _metric(_mnistRes, "CW (L2)", "average_case")
    _mnWorst = _metric(_mnistRes, "CW (L2)", "worst_case")
    _cfBest = _metric(_cifarRes, "CW (L2)", "best_case")
    _cfAvg = _metric(_cifarRes, "CW (L2)", "average_case")
    _cfWorst = _metric(_cifarRes, "CW (L2)", "worst_case")

    _orderOk = (
        _mnBest["mean_l2"] <= _mnAvg["mean_l2"] <= _mnWorst["mean_l2"] + 1e-4
        and _cfBest["mean_l2"] <= _cfAvg["mean_l2"] <= _cfWorst["mean_l2"] + 1e-4
    )
    _succOk = _mnAvg["success"] >= 0.90 and _cfAvg["success"] >= 0.90

    return {
        "mnist_table_iv": _mnistRes,
        "cifar_table_iv": _cifarRes,
        "synthetic_digits": _synthStats,
        "passed": _orderOk and _succOk,
    }


def _breakDistillation(
    mnistT1:nn.Module,
    mnistT100:nn.Module,
    cifarT1:nn.Module,
    cifarT40:nn.Module,
    _mnist, _cifar,
    l2Steps = 1000, l2Search = 10,
    linfSteps = 150, l0Steps = 100,
    verbose = True,
):
    print("\nDEFENSIVE DISTILLATION")

    _gradStats = {}
    for _label, _m, _ds in [
        ("MNIST T=1", mnistT1, _mnist),
        ("MNIST T=100", mnistT100, _mnist),
        ("CIFAR T=1", cifarT1, _cifar),
        ("CIFAR T=40", cifarT40, _cifar),
    ]:
        _x, _y = next(iter(_ds))
        _x, _y = _x.to(gpu), _y.to(gpu)
        _m.eval().to(gpu)
        _xVar = _x.clone().detach().requires_grad_(True)
        _logits = _m(_xVar)
        _l1Vals = _logits.abs().sum(dim=-1)
        _l1Mean = _l1Vals.mean().item()
        _l1Std = _l1Vals.std().item() if _l1Vals.numel() > 1 else 0.0
        _ceLoss = F.cross_entropy(_logits, _y, reduction="sum")
        _ceGrad = torch.autograd.grad(_ceLoss, _xVar)[0]
        _ceNorms = _ceGrad.abs().flatten(1).max(dim=1).values
        _gradStats[_label] = {
            "logit_l1_mean": _l1Mean,
            "logit_l1_std": _l1Std,
            "ce_grad_linf": _ceNorms.mean().item(),
            "vanished_frac": (_ceNorms < 1e-12).float().mean().item(),
        }

    print("\nLogit Norms and Gradients")
    for _label, _st in _gradStats.items():
        print(
            f"  {_label:<16} | L1 = {_st['logit_l1_mean']:>7.2f} +/- {_st['logit_l1_std']:<6.2f} | "
            f"grad = {_st['ce_grad_linf']:>10.3e} | "
            f"vanished = {_st['vanished_frac']*100:>5.1f}%"
        )

    torch.manual_seed(0)
    _mnistAttacks = [
        _CW_L0_UNSTABLE(mnistT100, optimSteps=l0Steps, cStart=1e-2, cEnd=1e4),
        _CW_L2(mnistT100, optimSteps=l2Steps, searchN=l2Search, cStart=1e-2),
        _CW_Linf(mnistT100, optimSteps=linfSteps, cStart=1e-2, cEnd=1e4),
        _FGSM_linf(mnistT100, eps=0.2),
    ]
    _mnistRes = _performAdversarialTest_BestWorst(
        mnistT100, _mnistAttacks, _mnist, mode="all", verbose=verbose, device=gpu
    )

    torch.manual_seed(0)
    _cifarAttacks = [
        _CW_L0_UNSTABLE(cifarT40, optimSteps=l0Steps, cStart=1e-2, cEnd=1e4),
        _CW_L2(cifarT40, optimSteps=l2Steps, searchN=l2Search, cStart=1e-2),
        _CW_Linf(cifarT40, optimSteps=linfSteps, cStart=1e-2, cEnd=1e4),
        _FGSM_linf(cifarT40, eps=8 / 255),
    ]
    _cifarRes = _performAdversarialTest_BestWorst(
        cifarT40, _cifarAttacks, _cifar, mode="all", verbose=verbose, device=gpu
    )

    _rows = [
        ("L0", "CW (L0)", "mean_l0", "{:>6.1f}"),
        ("L2", "CW (L2)", "mean_l2", "{:>6.2f}"),
        ("Linf", "CW (linf)", "mean_linf", "{:>6.3f}"),
        ("FGSM", "FGSM", "mean_linf", "{:>6.3f}"),
    ]

    print("\nDistilled MNIST (T=100) and CIFAR-10 (T=40)")
    for _label, _tag, _metricKey, _fmt in _rows:
        _cells = []
        for _case in ["best_case", "average_case", "worst_case"]:
            _mMn = _metric(_mnistRes, _tag, _case)
            _mCf = _metric(_cifarRes, _tag, _case)
            _mnStr = f"{_fmt.format(_mMn[_metricKey])}({_mMn['success']*100:>3.0f}%)"
            _cfStr = f"{_fmt.format(_mCf[_metricKey])}({_mCf['success']*100:>3.0f}%)"
            _cells.append(f"{_mnStr} / {_cfStr}")
        print(f"{_label:<24} || {_cells[0]:<30} || {_cells[1]:<30} || {_cells[2]:<30}")

    _mnOk = all(
        _metric(_mnistRes, _tag, "average_case")["success"] >= 0.90
        for _tag in ["CW (L0)", "CW (L2)", "CW (linf)"]
    )
    _cfOk = all(
        _metric(_cifarRes, _tag, "average_case")["success"] >= 0.85
        for _tag in ["CW (L0)", "CW (L2)", "CW (linf)"]
    )

    return {
        "grad_stats": _gradStats,
        "mnist_table_vi": _mnistRes,
        "cifar_table_vi": _cifarRes,
        "passed": _mnOk and _cfOk,
    }


def _testTransferability(
    mnistT1:nn.Module,
    mnistT100:nn.Module,
    cifarT1:nn.Module,
    cifarT40:nn.Module,
    _mnist, _cifar,
    kappas = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0],
    optimSteps = 1000, searchN = 10,
    verbose = True,
):
    print("\nTRANSFERABILITY")

    _allTransfer = {}
    for _domain, _mSrc, _mDst, _ds in [
        ("MNIST T=1 -> T=100", mnistT1, mnistT100, _mnist),
        ("CIFAR T=1 -> T=40", cifarT1, cifarT40, _cifar),
    ]:
        _x, _y = next(iter(_ds))
        _x, _y = _x.to(gpu), _y.to(gpu)
        torch.manual_seed(0)
        _targets = randomIncorrectTargets(_y, numClasses=10)

        _mSrc.eval().to(gpu)
        _mDst.eval().to(gpu)

        print(f"\n{_domain}")
        _statsK = {}
        for _k in kappas:
            _atk = _CW_L2(_mSrc, optimSteps=optimSteps, searchN=searchN, cStart=1e-2, kappa=_k)
            _adv = _atk.attack(_x, _y, targets=_targets, verbose=verbose)

            with torch.no_grad():
                _predsSrc = _mSrc(_adv).argmax(dim=-1)
                _predsDst = _mDst(_adv).argmax(dim=-1)

                _wbSucc = (_predsSrc == _targets).float().mean().item()
                _untargetedTr = (_predsDst != _y).float().mean().item()
                _targetedTr = (_predsDst == _targets).float().mean().item()
                _meanL2 = computeLpDistances(_x, _adv)["l2"].mean().item()

            _statsK[_k] = {
                "whitebox_success": _wbSucc,
                "untargeted_transfer": _untargetedTr,
                "targeted_transfer": _targetedTr,
                "mean_l2": _meanL2,
            }
            print(
                f"{_k:<12.1f} | {_wbSucc*100:>16.1f}%  | "
                f"{_untargetedTr*100:>19.1f}%  | "
                f"{_targetedTr*100:>17.1f}%  | {_meanL2:>8.4f}"
            )

        _allTransfer[_domain] = _statsK

    _improves = True
    for _domain, _statsK in _allTransfer.items():
        _lowK = _statsK[kappas[0]]
        _highK = max(_statsK.values(), key=lambda d: d["untargeted_transfer"])
        _ok = _highK["untargeted_transfer"] >= _lowK["untargeted_transfer"] and _highK["untargeted_transfer"] > 0.0
        _improves = _improves and _ok

    return {"transfer_stats": _allTransfer, "passed": _improves}


def _ensureCheckpoints(ckptDir = "checkpoints"):
    if not os.path.exists(ckptDir) and os.path.exists(os.path.join(_SRC_DIR, ckptDir)):
        ckptDir = os.path.join(_SRC_DIR, ckptDir)
    os.makedirs(ckptDir, exist_ok=True)

    _mnistPath = os.path.join(ckptDir, "mnist.pt")
    _mnistDistPath = os.path.join(ckptDir, "mnist_distil_dict.pt")
    _cifarPath = os.path.join(ckptDir, "latest_T1.pt")
    _cifarDistPath = os.path.join(ckptDir, "distil_dict.pt")

    if not os.path.exists(_mnistPath) or not os.path.exists(_mnistDistPath):
        _mnistData = getMNIST(bsize=128, shuffle=True)
        if not os.path.exists(_mnistPath):
            train(MNIST_CNN(1, True), _mnistData, epochs=10, lr=1e-3, temperature=1, saveName="mnist.pt", device=gpu)
        if not os.path.exists(_mnistDistPath):
            _mnistTeacher = MNIST_CNN(1, True)
            train(_mnistTeacher, _mnistData, epochs=10, lr=1e-3, temperature=100, saveName="mnist_teacher_T100.pt", device=gpu)
            _trainStudent(
                MNIST_CNN(1, True), _mnistTeacher, _mnistData,
                temp=100, epochs=10, lr=1e-3,
                dictSaveName="mnist_distil_dict.pt", modelSaveName=None, device=gpu,
            )

    if not os.path.exists(_cifarPath) or not os.path.exists(_cifarDistPath):
        _cifarData = getCIFAR(bsize=128, shuffle=True)
        if not os.path.exists(_cifarPath):
            train(CW_CNN(3, True), _cifarData, epochs=50, lr=1e-3, temperature=1, saveName="latest_T1.pt", device=gpu)
        if not os.path.exists(_cifarDistPath):
            _cifarTeacher = CW_CNN(3, True)
            train(_cifarTeacher, _cifarData, epochs=50, lr=1e-3, temperature=40, saveName="cifar_teacher_T40.pt", device=gpu)
            _trainStudent(
                CW_CNN(3, True), _cifarTeacher, _cifarData,
                temp=40, epochs=50, lr=1e-3,
                dictSaveName="distil_dict.pt", modelSaveName=None, device=gpu,
            )


def runProofs(
    n = 64, bsize = 64,
    verbose = True,
    logPath = "results/proof_findings.log",
    jsonPath = "results/proof_findings.json",
):
    _origCwd = os.getcwd()
    if not os.path.exists("checkpoints") and os.path.exists(os.path.join(_SRC_DIR, "checkpoints")):
        os.chdir(_SRC_DIR)

    _ensureCheckpoints()

    _tee = _TeeLogger(logPath)
    sys.stdout = _tee
    _t0 = now()
    _findings = {
        "timestamp": strftime("%Y-%m-%d %H:%M:%S"),
        "n_samples": n,
        "bsize": bsize,
    }
    try:
        _mnistT1 = loadModel("mnist.pt", MNIST_CNN(1)).to(gpu).eval()
        _mnistT100 = loadModel("mnist_distil_dict.pt", MNIST_CNN(1)).to(gpu).eval()
        _cifarT1 = loadModel("latest_T1.pt", CW_CNN(3)).to(gpu).eval()
        _cifarT40 = loadModel("distil_dict.pt", CW_CNN(3)).to(gpu).eval()
        _mnistT1.name = "MNIST_T1"
        _mnistT100.name = "MNIST_T100"
        _cifarT1.name = "CIFAR_T1"
        _cifarT40.name = "CIFAR_T40"

        _mnistLoaders = getMNIST(bsize=bsize, shuffle=False)
        _cifarLoaders = getCIFAR(bsize=bsize, shuffle=False)

        _, _mnT1Acc = holdoutTest(_mnistT1, _mnistLoaders[1], nn.CrossEntropyLoss(), verbose=verbose, device=gpu)
        _, _mnT100Acc = holdoutTest(_mnistT100, _mnistLoaders[1], nn.CrossEntropyLoss(), verbose=verbose, device=gpu)
        _, _cfT1Acc = holdoutTest(_cifarT1, _cifarLoaders[1], nn.CrossEntropyLoss(), verbose=verbose, device=gpu)
        _, _cfT40Acc = holdoutTest(_cifarT40, _cifarLoaders[1], nn.CrossEntropyLoss(), verbose=verbose, device=gpu)

        _findings["clean_accuracies"] = {
            "MNIST_T1": float(_mnT1Acc),
            "MNIST_T100": float(_mnT100Acc),
            "CIFAR_T1": float(_cfT1Acc),
            "CIFAR_T40": float(_cfT40Acc),
        }
        _saveFindingsJson(_findings, jsonPath)

        print("\nClean Accuracy")
        print(f"  MNIST T=1:   {float(_mnT1Acc)*100:.2f}%")
        print(f"  MNIST T=100: {float(_mnT100Acc)*100:.2f}%")
        print(f"  CIFAR T=1:   {float(_cfT1Acc)*100:.2f}%")
        print(f"  CIFAR T=40:  {float(_cfT40Acc)*100:.2f}%")

        _mnistDs = _getCorrect(
            [_mnistT1, _mnistT100], _mnistLoaders[1], n=n, bsize=bsize, name="MNIST", device=gpu
        )
        _cifarDs = _getCorrect(
            [_cifarT1, _cifarT40], _cifarLoaders[1], n=n, bsize=bsize, name="CIFAR10", device=gpu
        )

        _p1 = _compareFs_optims(_mnistT1, _cifarT1, _mnistDs, _cifarDs, verbose=verbose)
        _findings["proof1"] = _p1
        _saveFindingsJson(_findings, jsonPath)

        _p2 = _compareNorms_manifold(_mnistT1, _cifarT1, _mnistDs, _cifarDs, verbose=verbose)
        _findings["proof2"] = _p2
        _saveFindingsJson(_findings, jsonPath)

        _p3 = _breakDistillation(_mnistT1, _mnistT100, _cifarT1, _cifarT40, _mnistDs, _cifarDs, verbose=verbose)
        _findings["proof3"] = _p3
        _saveFindingsJson(_findings, jsonPath)

        _p4 = _testTransferability(_mnistT1, _mnistT100, _cifarT1, _cifarT40, _mnistDs, _cifarDs, verbose=verbose)
        _findings["proof4"] = _p4

        _elapsedMin = (now() - _t0) / 60.0
        _findings["elapsed_minutes"] = _elapsedMin
        _saveFindingsJson(_findings, jsonPath)

        print(f"\nDone in {now() - _t0:.2f}s")
        return _findings
    finally:
        _tee.close()
        os.chdir(_origCwd)


if __name__ == "__main__":
    n, bsize = 512, 64
    _ensureCheckpoints()
    runProofs(n=n, bsize=bsize)
