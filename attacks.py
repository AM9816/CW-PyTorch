import torch
from torch import nn, Tensor as T
from tqdm import tqdm

from torch_common import getAcc


class _FGSM_linf:
    def __init__(self, model, eps=16 / 255, lossf=nn.CrossEntropyLoss()):
        self.model = model
        self.eps = eps
        self.lossf = lossf

    def __call__(self, x: T, labels: T) -> T:
        device = x.device
        self.model = self.model.eval().to(device)

        x = x.clone().detach().to(device).requires_grad_(True)
        y = self.model(x).clone().to(device)
        labels = labels.clone().to(device)

        loss = self.lossf(y, labels)
        Dloss_Dimg = torch.autograd.grad([loss], x, retain_graph=False)[0]

        advImage = x + self.eps * Dloss_Dimg.sign()
        return advImage.clamp(0, 1).detach()

    def __str__(self):
        return f"FGSM (Linf, eps = {self.eps * 255}/255)"


def maskImportant(target, numClasses=10):
    return nn.functional.one_hot(target, numClasses).bool()


def _CW_loss(logits: T, targets: T, doTarget=False, kappa=0.0):
    bsize, classes = logits.shape
    mask = maskImportant(targets, classes)

    allButTarget = logits.masked_fill(mask, -float("inf")).max(dim=-1).values
    justTarget = logits[torch.arange(bsize, device=logits.device), targets]

    diff = (allButTarget - justTarget) if doTarget else (justTarget - allButTarget)
    return (diff + kappa).clamp(min=0.0)


_clipEpsilon = 1e-4
_toCWspace = lambda x: torch.atanh(
    (x * 2.0 - 1.0).clamp(-1 + _clipEpsilon, 1 - _clipEpsilon)
)
_fromCWspace = lambda w: 0.5 * (torch.tanh(w) + 1.0)


class _CW_Linf:
    def __init__(
        self,
        model: nn.Module,
        lr=5e-3,
        optimSteps=1000,
        cStart=1e-4,
        cEnd=1e10,
        cFactor=2.0,
        kappa0=0.0,
        tauDecay=0.9,
        forcedBsize=None,
    ):
        self.model = model
        self.lr = lr
        self.c0 = cStart
        self.cMax = cEnd
        self.cMultip = cFactor
        self.N = optimSteps
        self.kappa = kappa0
        self.tauDecay = tauDecay
        self.forcedBsize = forcedBsize

    def __call__(self, x: T, y: T, targets=None):
        return self.attack(x, y, targets)

    def __str__(self):
        return "CW (linf)"

    def attack(self, X: T, Y: T, targets=None, verbose=True, *args, **kwargs):
        device = next(self.model.parameters()).device
        modelState = self.model.training
        forcedBsize = self.forcedBsize

        X = X.to(device)
        self.model.eval()
        bsizeInput = X.size(0)
        targetted = targets is not None
        globalBestAdv = X.detach().to(device).clone()
        shape = [-1] + [1 for _ in range(X.ndim - 1)]
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
            tau = torch.ones(bsize, device=device)
            globalBestDmax = torch.full((bsize,), float("inf"), device=device)
            C = torch.full((bsize,), self.c0, device=device)
            alive = torch.ones(bsize, dtype=torch.bool, device=device)

            tauIterations = cIterations = 0
            pbar = tqdm() if verbose else None

            while (tauIds := alive & (tau >= 1 / 255)).any():
                acc = (~torch.isfinite(globalBestDmax)).float().mean().item()

                tauIds = tauIds.nonzero(as_tuple=True)[0]
                foundGivenTau = torch.zeros_like(tauIds, dtype=torch.bool)
                tauBestDmax = tau[tauIds].clone()

                tauIterations += 1

                while (cMask := (~foundGivenTau & (C[tauIds] <= self.cMax))).any():
                    if pbar is not None:
                        pbar.set_description_str(
                            f"{str(self)}: acc = {acc}, tau iters = {tauIterations}, "
                            f"c iters = {cIterations}"
                        )
                        pbar.update(cIterations)

                    cIterations += 1

                    cMaskIds = cMask.nonzero(as_tuple=True)[0]
                    activeIds = tauIds[cMaskIds]

                    _x = x[activeIds]
                    _t = t[activeIds]
                    _c = C[activeIds]
                    _tau = tau[activeIds]

                    _W = W[activeIds].clone().detach().requires_grad_(True)
                    optim = torch.optim.Adam([_W], lr=self.lr)

                    foundGivenC = torch.zeros_like(activeIds, dtype=torch.bool)
                    bestDmaxGivenC = _tau.clone()
                    prevLoss = float("inf")

                    for stepIdx in range(self.N):
                        optim.zero_grad()
                        _X = _fromCWspace(_W)
                        logits = self.model(_X)
                        delta = (_X - _x).abs()
                        cwLoss = _CW_loss(logits, _t, targetted, self.kappa)

                        with torch.no_grad():
                            preds = logits.argmax(dim=-1)
                            fooled = (
                                (preds == _t) if targetted else (preds != _t)
                            ) & (cwLoss == 0)
                            dmax = delta.flatten(1).max(dim=1).values

                            improved = fooled & (dmax < globalBestDmax[activeIds])
                            globalBestDmax[activeIds[improved]] = dmax[improved]
                            globalBestAdv[index + activeIds[improved]] = _X[improved].detach()

                            roundImproved = fooled & (dmax < bestDmaxGivenC)
                            foundGivenC = foundGivenC | roundImproved
                            bestDmaxGivenC[roundImproved] = dmax[roundImproved]

                            W[activeIds[roundImproved]] = _W[roundImproved].detach()

                            if foundGivenC.all() or (fooled & (dmax < _tau)).all():
                                break

                        penalty = (
                            (delta - _tau.view(shape))
                            .clamp(min=0)
                            .flatten(1)
                            .sum(dim=1)
                        )

                        loss = (penalty + _c * cwLoss).sum()
                        if stepIdx > 0 and stepIdx % 25 == 0:
                            curLoss = loss.item()
                            if curLoss > prevLoss * 0.9999:
                                break
                            prevLoss = curLoss

                        loss.backward()
                        optim.step()

                    W[activeIds[~foundGivenC]] = _W[~foundGivenC].detach()
                    foundGivenTau[cMaskIds[foundGivenC]] = True
                    tauBestDmax[cMaskIds[foundGivenC]] = bestDmaxGivenC[foundGivenC]
                    C[activeIds[~foundGivenC]] *= self.cMultip

                fooledIds = tauIds[foundGivenTau]
                failedIds = tauIds[~foundGivenTau]

                tau[fooledIds] = tauBestDmax[foundGivenTau] * self.tauDecay
                alive[failedIds] = False

            if pbar is not None:
                pbar.close()

        self.model.train(modelState)
        return globalBestAdv


class _CW_L2:
    def __init__(
        self,
        model: nn.Module,
        lr=1e-2,
        optimSteps=1000,
        searchN=20,
        cStart=1e-3,
        cEnd=1e10,
        cFactor=10.0,
        kappa=0.0,
    ):
        self.model = model
        self.lr = lr
        self.c0 = cStart
        self.cMax = cEnd
        self.cMultip = cFactor
        self.optimN = optimSteps
        self.searchN = searchN
        self.kappa = kappa

    def __call__(self, x: T, y: T, targets=None):
        return self.attack(x, y, targets)

    def __str__(self):
        return "CW (L2)"

    def attack(self, X: T, Y: T, targets=None, verbose=True, *args, **kwargs):
        device = next(self.model.parameters()).device
        modelState = self.model.training

        X = X.to(device)
        self.model.eval()

        bsize = X.size(0)
        targetted = targets is not None
        globalBestAdv = X.detach().to(device).clone()
        x = X.detach().to(device)
        t = (targets if targetted else Y).detach().to(device)

        W0 = _toCWspace(x)

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

            _W = W0[activeIds].clone().detach().requires_grad_(True)
            optim = torch.optim.Adam([_W], lr=self.lr)

            found = torch.zeros_like(activeIds, dtype=torch.bool)
            prevLoss = float("inf")
            checkEvery = max(15, self.optimN // 10)

            for stepIdx in range(self.optimN):
                optim.zero_grad()
                _X = _fromCWspace(_W)
                logits = self.model(_X)
                delta = _X - _x
                l2DistSq = delta.pow(2).flatten(1).sum(dim=1)
                l2Dist = l2DistSq.sqrt()

                cwLoss = _CW_loss(logits, _t, targetted, self.kappa)

                with torch.no_grad():
                    acc = getAcc(logits, Y[activeIds])
                    preds = logits.argmax(dim=-1)
                    fooled = (preds == _t) if targetted else (preds != _t)
                    fooledConfidently = fooled & (cwLoss == 0)

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
                    f"C avg = {C.mean().item()}"
                )
                pbar.update()

        if pbar is not None:
            pbar.close()

        self.model.train(modelState)
        return globalBestAdv
