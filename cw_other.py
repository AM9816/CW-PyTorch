import torch
from torch import nn, Tensor as T
from tqdm import tqdm

from torch_common import getDevices, getAcc
from attacks import _toCWspace, _fromCWspace
from cw_losses import (
    _CW_L1,
    _CW_L2,
    _CW_L3,
    _CW_L4,
    _CW_L5,
    _CW_L6,
    _CW_L7,
)

cpu, gpu = getDevices()

CW_OBJECTIVES = {
    "f1": _CW_L1,
    "f2": _CW_L2,
    "f3": _CW_L3,
    "f4": _CW_L4,
    "f5": _CW_L5,
    "f6": _CW_L6,
    "f7": _CW_L7,
}


def _resolveLossf(lossf):
    if isinstance(lossf, str):
        key = lossf.lower()
        assert key in CW_OBJECTIVES, f"Unknown CW objective '{lossf}'. Choose from {list(CW_OBJECTIVES.keys())}."
        return CW_OBJECTIVES[key], key
    name = getattr(lossf, "__name__", "custom")
    return lossf, name


def _greedySearchForL0(
    model:nn.Module, x:T, advContinuous:T, targets:T, targetted=False,
    maxLatticeSteps=256,
):

    advDisc = (advContinuous * 255.0).round().clamp(0.0, 255.0) / 255.0

    with torch.no_grad():
        preds = model(advDisc).argmax(dim=-1)
        fooled = (preds == targets) if targetted else (preds != targets)

    if fooled.all() or maxLatticeSteps <= 0:
        return advDisc

    unfooledIds = (~fooled).nonzero(as_tuple=True)[0]

    for idx in unfooledIds.tolist():

        _current = advDisc[idx : idx + 1].clone().detach()
        _currentTarget = targets[idx : idx + 1]
        _changedMask = torch.zeros_like(_current, dtype=torch.bool).flatten()

        for _ in range(maxLatticeSteps):
            with torch.no_grad():
                p = model(_current).argmax(dim=-1)
                if (p == _currentTarget).item() if targetted else (p != _currentTarget).item():
                    break

            curVar = _current.clone().detach().requires_grad_(True)
            logits = model(curVar)
            loss = _CW_L6(logits, _currentTarget, doTarget=targetted, kappa=1e6).sum()
            grad = torch.autograd.grad(loss, curVar)[0].detach().flatten()

            with torch.no_grad():

                stepDir = -grad.sign() / 255.0
                curFlat = _current.flatten()

                candidateVals = (curFlat + stepDir).clamp(0.0, 1.0)
                validMove = (candidateVals != curFlat) & (~_changedMask)

                if not validMove.any():
                    break

                scores = grad.abs().masked_fill(~validMove, -float("inf"))
                bestCoord = scores.argmax()
                curFlat[bestCoord] = candidateVals[bestCoord]
                _changedMask[bestCoord] = True
                _current = curFlat.view_as(_current)


        advDisc[idx:idx+1] = _current

    return advDisc


class _CW_L2_MODULAR_UNSTABLE:
    def __init__(self, model: nn.Module, boxMethod = "change_of_variables",
        lossf="f6", solver = "adam", numRestarts = 1,
        greedyDiscretize = False,
        lr = 1e-2, optimSteps = 1000,
        searchN = 20, cStart = 1e-3, cEnd = 1e10, cFactor = 10.0,
        kappa = 0.0,
    ):
        assert boxMethod in ("change_of_variables", "clipped", "projected")
        assert solver in ("adam", "sgd", "momentum")

        self.model = model
        self.boxMethod = boxMethod
        self.lossf, self.lossName = _resolveLossf(lossf)
        self.solver = solver
        self.numRestarts = max(1, int(numRestarts))
        self.greedyDiscretize = greedyDiscretize
        self.lr = lr
        self.optimN = optimSteps
        self.searchN = searchN
        self.c0 = cStart
        self.cMax = cEnd
        self.cMultip = cFactor
        self.kappa = kappa

    def __call__(self, x:T, y:T, targets=None):
        return self.attack(x, y, targets)

    def __str__(self):
        extras = [self.lossName, self.boxMethod]
        if self.solver != "adam":
            extras.append(self.solver)
        if self.numRestarts > 1:
            extras.append(f"restarts={self.numRestarts}")

        return f"CW (L2, {', '.join(extras)})"

    def _getOptimSelf(self, param: T):
        if self.solver == "adam":
            return torch.optim.Adam([param], lr=self.lr)
        elif self.solver == "momentum":
            return torch.optim.SGD([param], lr=self.lr, momentum=0.9)
        return torch.optim.SGD([param], lr=self.lr)

    def _sampleNormBallStart(self, likeTensor:T, radii:T):
        bsize = likeTensor.size(0)
        flatDim = likeTensor[0].numel()
        direction = torch.randn_like(likeTensor)
        dirNorm = direction.flatten(1).norm(p=2, dim=1).clamp(min=1e-12)
        u = torch.rand(bsize, device=likeTensor.device).pow(1.0 / flatDim)
        safeRadii = torch.where(torch.isfinite(radii), radii, torch.full_like(radii, 0.1))
        scale = (u * safeRadii / dirNorm).view([-1] + [1] * (likeTensor.ndim - 1))

        return direction * scale

    def attack(self, X:T, Y:T, targets=None, verbose=True, *args, **kwargs):
        device = next(self.model.parameters()).device
        modelState = self.model.training

        X = X.to(device)
        self.model.eval()

        bsize = X.size(0)
        targetted = targets is not None
        globalBestAdv = X.detach().to(device).clone()
        x = X.detach().to(device)
        t = (targets if targetted else Y).detach().to(device)

        C = torch.full((bsize,), self.c0, device=device)
        cLeast = torch.zeros(bsize, device=device)
        cMost = torch.full((bsize,), float("inf"), device=device)
        bestScore = torch.full((bsize,), float("inf"), device=device)

        pbar = tqdm(total=self.searchN) if verbose else None

        for i in range(self.searchN):
            active = C <= self.cMax
            if not active.any():
                break

            activeIds = active.nonzero(as_tuple=True)[0]
            _x = x[activeIds]
            _t = t[activeIds]
            _c = C[activeIds]

            found = torch.zeros_like(activeIds, dtype=torch.bool)

            for r in range(self.numRestarts):
                if r == 0:
                    startX = _x
                else:
                    noise = self._sampleNormBallStart(_x, bestScore[activeIds])
                    startX = (_x + noise).clamp(0.0, 1.0)

                if self.boxMethod == "change_of_variables":
                    optParam = _toCWspace(startX).detach().requires_grad_(True)
                else:
                    optParam = (startX - _x).detach().requires_grad_(True)

                optim = self._getOptimSelf(optParam)
                prevLoss = float("inf")
                checkEvery = max(15, self.optimN // 10)

                for stepIdx in range(self.optimN):
                    optim.zero_grad()

                    if self.boxMethod == "change_of_variables":
                        _X = _fromCWspace(optParam)
                    elif self.boxMethod == "clipped":
                        _X = (_x + optParam).clamp(0.0, 1.0)
                    else:
                        _X = _x + optParam

                    logits = self.model(_X)
                    delta = _X - _x
                    l2DistSq = delta.pow(2).flatten(1).sum(dim=1)
                    l2Dist = l2DistSq.sqrt()

                    cwLoss = self.lossf(logits, _t, targetted, self.kappa)

                    with torch.no_grad():
                        acc = getAcc(logits, Y[activeIds])
                        preds = logits.argmax(dim=-1)
                        fooled = (preds == _t) if targetted else (preds != _t)
                        fooledConfidently = fooled & (cwLoss <= 0)

                        found = found | fooledConfidently
                        improved = fooledConfidently & (l2Dist < bestScore[activeIds])
                        bestScore[activeIds[improved]] = l2Dist[improved]
                        globalBestAdv[activeIds[improved]] = _X[improved].detach()

                    loss = (l2DistSq + _c * cwLoss).sum()
                    if stepIdx > 0 and stepIdx % checkEvery == 0:
                        curLoss = loss.item()
                        if curLoss > prevLoss * 0.9999:
                            break
                        prevLoss = curLoss

                    loss.backward()
                    optim.step()

                    if self.boxMethod == "projected":
                        with torch.no_grad():
                            optParam.copy_((_x + optParam).clamp(0.0, 1.0) - _x)

            fooled = activeIds[found]
            failed = activeIds[~found]

            cMost[fooled] = torch.min(cMost[fooled], C[fooled])
            C[fooled] = 0.5 * (cLeast[fooled] + cMost[fooled])

            cLeast[failed] = torch.max(cLeast[failed], C[failed])
            bounds = torch.isfinite(cMost[failed])
            C[failed] = torch.where(
                bounds,
                0.5 * (cLeast[failed] + cMost[failed]),
                C[failed] * self.cMultip,
            )

            if pbar is not None:
                pbar.set_description_str(
                    f"{str(self)}: acc = {acc:.1%}, step = {i + 1}/{self.searchN}, "
                    f"C avg = {C.mean().item():.3g}"
                )
                pbar.update()

        if pbar is not None:
            pbar.close()

        if self.greedyDiscretize:
            outAdv = _greedySearchForL0(self.model, x, globalBestAdv, t, targetted)
        else:
            outAdv = globalBestAdv

        self.model.train(modelState)
        return outAdv


class _CW_L2_BOX_PROJECTION(_CW_L2_MODULAR_UNSTABLE):
    def __init__(self, model: nn.Module, lossf="f6", **kwargs):
        super().__init__(model, boxMethod="projected", lossf=lossf, **kwargs)


class _CW_L2_BOX_CLIPPED(_CW_L2_MODULAR_UNSTABLE):
    def __init__(self, model: nn.Module, lossf="f6", **kwargs):
        super().__init__(model, boxMethod="clipped", lossf=lossf, **kwargs)


class _CW_L2_WARMSTART(_CW_L2_MODULAR_UNSTABLE):
    def __init__(self, model:nn.Module, numRestarts=5, lossf="f6", **kwargs):
        super().__init__(model, boxMethod="change_of_variables", lossf=lossf, numRestarts=numRestarts,
            **kwargs,)



class _CW_L0_UNSTABLE:
    def __init__(
        self, model: nn.Module, lr = 1e-2,
        optimSteps = 100, cStart = 1e-4, cEnd = 1e10,
        cFactor = 2.0, kappa = 0.0, lossf=_CW_L6,
        independentChannels=False,
        forcedBsize=None,
    ):
        self.model = model
        self.lr = lr
        self.c0 = cStart
        self.cMax = cEnd
        self.cMultip = cFactor
        self.N = optimSteps
        self.kappa = kappa
        self.lossf, _ = _resolveLossf(lossf)
        self.independentChannels = independentChannels
        self.forcedBsize = forcedBsize

    def __call__(self, x: T, y: T, targets=None):
        return self.attack(x, y, targets)

    def __str__(self):
        return "CW (L0)"

    def _l0Dist(self, delta:T):
        if self.independentChannels or delta.ndim < 3:
            return (delta.abs() > 1e-6).flatten(1).sum(dim=1)
        return (delta.abs().sum(dim=1) > 1e-6).flatten(1).sum(dim=1)

    def attack(self, X:T, Y:T, targets=None, verbose=True, *args, **kwargs):
        device = next(self.model.parameters()).device
        modelState = self.model.training
        forcedBsize = self.forcedBsize

        X = X.to(device)
        self.model.eval()
        bsizeInput = X.size(0)
        targetted = targets is not None
        globalBestAdv = X.detach().to(device).clone()

        step = (
            bsizeInput
            if (forcedBsize is None or forcedBsize >= bsizeInput)
            else forcedBsize
        )
        lossTargets = targets if targetted else Y

        for index in range(0, bsizeInput, step):
            indexEnd = min(index + step, bsizeInput)
            bsize = indexEnd - index

            x = X[index:indexEnd].detach().to(device)
            t = lossTargets[index:indexEnd].detach().to(device)

            W = _toCWspace(x)
            C = torch.full((bsize,), self.c0, device=device)

            if self.independentChannels:
                allowed = torch.ones_like(x)
            else:
                allowed = torch.ones((bsize, 1, *x.shape[2:]), device=device, dtype=x.dtype)

            globalBestL0 = torch.full((bsize,), float("inf"), device=device)
            alive = torch.ones(bsize, dtype=torch.bool, device=device)

            shrinkIterations = cIterations = 0
            pbar = tqdm() if verbose else None

            while (alive & (allowed.flatten(1).sum(dim=1) > 0)).any():
                acc = (~torch.isfinite(globalBestL0)).float().mean().item()
                meanL0 = (
                    globalBestL0[torch.isfinite(globalBestL0)].mean().item()
                    if torch.isfinite(globalBestL0).any()
                    else float("inf")
                )

                outerIds = (alive & (allowed.flatten(1).sum(dim=1) > 0)).nonzero(as_tuple=True)[0]
                foundGivenAllowed = torch.zeros_like(outerIds, dtype=torch.bool)
                bestAdvForAllowed = x[outerIds].clone()

                shrinkIterations += 1

                while (cMask := (~foundGivenAllowed & (C[outerIds] <= self.cMax))).any():
                    if pbar is not None:
                        pbar.set_description_str(
                            f"{str(self)}: acc = {acc:.1%}, mean L0 = {meanL0:.1f}, "
                            f"shrink iters = {shrinkIterations}, c iters = {cIterations}"
                        )
                        pbar.update(1)

                    cIterations += 1

                    cMaskIds = cMask.nonzero(as_tuple=True)[0]
                    activeIds = outerIds[cMaskIds]

                    _x = x[activeIds]
                    _t = t[activeIds]
                    _c = C[activeIds]
                    _allowed = allowed[activeIds]

                    _W = W[activeIds].clone().detach().requires_grad_(True)
                    optim = torch.optim.Adam([_W], lr=self.lr)

                    foundGivenC = torch.zeros_like(activeIds, dtype=torch.bool)
                    bestL2GivenC = torch.full((activeIds.size(0),), float("inf"), device=device)
                    prevLoss = float("inf")

                    for stepIdx in range(self.N):
                        optim.zero_grad()
                        _X_unmasked = _fromCWspace(_W)
                        _X = _x + _allowed * (_X_unmasked - _x)

                        logits = self.model(_X)
                        delta = _X - _x
                        l2DistSq = delta.pow(2).flatten(1).sum(dim=1)
                        l2Dist = l2DistSq.sqrt()

                        cwLoss = self.lossf(logits, _t, targetted, self.kappa)

                        with torch.no_grad():
                            preds = logits.argmax(dim=-1)
                            fooled = (preds == _t) if targetted else (preds != _t)
                            fooledConfidently = fooled & (cwLoss == 0)

                            l0Dist = self._l0Dist(delta)

                            improvedGlobal = fooledConfidently & (l0Dist < globalBestL0[activeIds])
                            globalBestL0[activeIds[improvedGlobal]] = l0Dist[improvedGlobal].float()
                            globalBestAdv[index + activeIds[improvedGlobal]] = _X[improvedGlobal].detach()

                            roundImproved = fooledConfidently & (l2Dist < bestL2GivenC)
                            foundGivenC = foundGivenC | roundImproved
                            bestL2GivenC[roundImproved] = l2Dist[roundImproved]

                            W[activeIds[roundImproved]] = _W[roundImproved].detach()
                            bestAdvForAllowed[cMaskIds[roundImproved]] = _X[roundImproved].detach()

                            if stepIdx >= 5 and foundGivenC.all():
                                break

                        loss = (l2DistSq + _c * cwLoss).sum()
                        if stepIdx > 0 and stepIdx % 20 == 0:
                            curLoss = loss.item()
                            if curLoss > prevLoss * 0.9999:
                                break
                            prevLoss = curLoss

                        loss.backward()
                        optim.step()

                    W[activeIds[~foundGivenC]] = _W[~foundGivenC].detach()
                    foundGivenAllowed[cMaskIds[foundGivenC]] = True
                    C[activeIds[~foundGivenC]] *= self.cMultip

                fooledSubIdx = foundGivenAllowed.nonzero(as_tuple=True)[0]
                fooledIds = outerIds[fooledSubIdx]
                failedIds = outerIds[~foundGivenAllowed]

                alive[failedIds] = False

                if fooledIds.numel() > 0:
                    advX = bestAdvForAllowed[fooledSubIdx].clone().detach().requires_grad_(True)
                    logitsAdv = self.model(advX)
                    fVal = self.lossf(logitsAdv, t[fooledIds], targetted, 1e6).sum()
                    gradF = torch.autograd.grad(fVal, advX)[0]

                    with torch.no_grad():
                        deltaAdv = advX - x[fooledIds]
                        importance = (gradF * deltaAdv).abs()

                        if not self.independentChannels:
                            importance = importance.sum(dim=1, keepdim=True)
                            unchanged = deltaAdv.abs().max(dim=1, keepdim=True).values < (0.5 / 255.0)
                        else:
                            unchanged = deltaAdv.abs() < (0.5 / 255.0)

                        curAllowed = allowed[fooledIds]
                        curAllowed = curAllowed.masked_fill(unchanged, 0.0)

                        importanceMasked = importance.masked_fill(curAllowed == 0, float("inf"))
                        flatImp = importanceMasked.flatten(1)
                        curAllowedFlat = curAllowed.flatten(1)

                        for b in range(fooledIds.size(0)):
                            nActive = int(curAllowedFlat[b].sum().item())
                            if nActive > 0:
                                kFix = max(1, nActive // 15)
                                _, idxFix = torch.topk(flatImp[b], k=kFix, largest=False)
                                curAllowedFlat[b, idxFix] = 0.0

                        allowed[fooledIds] = curAllowedFlat.view_as(curAllowed)

            if pbar is not None:
                pbar.close()

        self.model.train(modelState)
        return globalBestAdv



class _BIM:
    def __init__(
        self,
        model: nn.Module, eps = 8 / 255,
        alpha = 1 / 255, steps = 10, lossf=nn.CrossEntropyLoss()):

        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.lossf = lossf

    def __call__(self, x:T, labels:T, targets=None):
        device = x.device
        self.model = self.model.eval().to(device)
        x0 = x.clone().detach().to(device)
        adv = x0.clone()
        targetted = targets is not None
        t = (targets if targetted else labels).clone().to(device)

        for _ in range(self.steps):
            adv.requires_grad_(True)
            out = self.model(adv)
            loss = self.lossf(out, t)
            grad = torch.autograd.grad([loss], adv, retain_graph=False)[0]

            with torch.no_grad():
                step = -self.alpha * grad.sign() if targetted else self.alpha * grad.sign()
                delta = (adv + step - x0).clamp(-self.eps, self.eps)
                adv = (x0 + delta).clamp(0.0, 1.0).detach()

        return (adv * 255.0).round() / 255.0

    def __str__(self):
        return f"IGS (Linf, eps = {self.eps * 255:.0f}/255)"
