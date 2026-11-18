import math
import torch
from torch import Tensor as T
from torch.nn import functional

_neginf, _posinf = -float("inf"), float("inf")
_log2 = math.log(2)
_eps = 1e-12


def _significantLogits(inputs: T, targets: T):
    device = inputs.device
    bsize, classes = inputs.shape
    mask = functional.one_hot(targets, classes).bool()

    allButTarget = inputs.masked_fill(mask, _neginf).max(dim=-1).values
    justTarget = inputs[torch.arange(0, bsize, device=device), targets]

    return justTarget, allButTarget


def _CW_L1(logits: T, targets: T, doTarget=False, kappa=0):
    ceLoss = functional.cross_entropy(logits, targets, reduction="none")
    if doTarget:
        return ceLoss - _log2 + kappa
    return -ceLoss + 1.0 + kappa


def _CW_L2(logits: T, targets: T, doTarget=False, kappa=0):
    probDist = functional.softmax(logits, dim=-1)
    justTarget, allButTarget = _significantLogits(probDist, targets)

    diff = (allButTarget - justTarget) if doTarget else (justTarget - allButTarget)
    return (diff + kappa).clamp(min=0)


def _CW_L3(logits: T, targets: T, doTarget=False, kappa=0):
    probDist = functional.softmax(logits, dim=-1)
    justTarget, allButTarget = _significantLogits(probDist, targets)
    diff = (allButTarget - justTarget) if doTarget else (justTarget - allButTarget)
    return functional.softplus(diff + kappa) - _log2


def _CW_L4(logits: T, targets: T, doTarget=False, kappa=0):
    probDist = functional.softmax(logits, dim=-1)
    justTarget, _ = _significantLogits(probDist, targets)
    diff = (0.5 - justTarget) if doTarget else (justTarget - 0.5)
    return (diff + kappa).clamp(min=0)


def _CW_L5(logits: T, targets: T, doTarget=False, kappa=0):
    probDist = functional.softmax(logits, dim=-1)
    justTarget, _ = _significantLogits(probDist, targets)
    p = justTarget if doTarget else (1.0 - justTarget)
    return -torch.log((2.0 * p).clamp(min=_eps)) + kappa


def _CW_L6(logits: T, targets: T, doTarget=False, kappa=0):
    justTarget, allButTarget = _significantLogits(logits, targets)
    diff = (allButTarget - justTarget) if doTarget else (justTarget - allButTarget)
    return (diff + kappa).clamp(min=0)


def _CW_L7(logits: T, targets: T, doTarget=False, kappa=0):
    justTarget, allButTarget = _significantLogits(logits, targets)
    diff = (allButTarget - justTarget) if doTarget else (justTarget - allButTarget)
    return functional.softplus(diff + kappa) - _log2
