import os
import torch
from torch import nn, Tensor as T
from tqdm import tqdm


def getDevices():
    cpu = torch.device("cpu")
    gpu = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return cpu, gpu


cpu, gpu = getDevices()
_checkpoint_dir = "checkpoints"


class SmallDataLoader:
    def __init__(self, datapoints, name="test DataLoader"):
        self.datapoints = datapoints
        self.name = name

    def __str__(self):
        return self.name

    def __iter__(self):
        for datapoint in self.datapoints:
            yield datapoint

    def __len__(self):
        return len(self.datapoints)


def loadModel(fileName, model=None):
    os.makedirs(_checkpoint_dir, exist_ok=True)
    path = os.path.join(_checkpoint_dir, fileName)
    if model is None:
        with open(path, "rb") as f:
            return torch.load(f, weights_only=False)
    model.load_state_dict(torch.load(path, map_location=gpu))
    return model


def saveModel(fileName, model: nn.Module, saveOnlyStateDict=True, appendClassName=False):
    extraName = f"{model.__class__}_" if appendClassName else ""
    finalName = f"{extraName}{fileName}"
    os.makedirs(_checkpoint_dir, exist_ok=True)
    path = os.path.join(_checkpoint_dir, finalName)
    payload = model.state_dict() if saveOnlyStateDict else model
    with open(path, "wb") as f:
        torch.save(payload, f)


def getAcc(out: T, y: T):
    preds = out.argmax(dim=1)
    return (y == preds).float().mean()


def holdoutTest(model, dataset, lossf, verbose=False, device=gpu):
    with torch.no_grad():
        model.eval()
        sumTestLoss = 0.0
        testDatapoints = 0
        sumAcc = 0.0

        iterator = enumerate(dataset)
        if verbose:
            iterator = tqdm(iterator, total=len(dataset))

        for _, (x, y) in iterator:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = lossf(out, y)
            sumTestLoss += loss * x.size(0)
            testDatapoints += x.size(0)
            sumAcc += getAcc(out, y) * x.size(0)

        model.train()

    return sumTestLoss / testDatapoints, sumAcc / testDatapoints
