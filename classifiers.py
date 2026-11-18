from tqdm import tqdm
import os
import torch
from torch import nn
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

from torch_common import getDevices, saveModel, loadModel, getAcc, holdoutTest

toTensor = transforms.ToTensor()
cpu, gpu = getDevices()


def getMNIST(
    dir="mnist",
    bsize=128,
    returnTest=True,
    transforms=toTensor,
    shuffle=False,
    downloadIfNotExists=True,
):
    if not os.path.exists(os.path.join(dir, "MNIST")) and os.path.exists(
        os.path.join("..", dir, "MNIST")
    ):
        dir = os.path.join("..", dir)
    mnistTrain = MNIST(dir, True, transforms, download=downloadIfNotExists)
    mnistTest = MNIST(dir, False, transforms, download=downloadIfNotExists) if returnTest else None

    if returnTest:
        return (
            DataLoader(mnistTrain, bsize, shuffle),
            DataLoader(mnistTest, bsize, shuffle),
        )
    return DataLoader(mnistTrain, bsize, shuffle)


class Dataloader_named(DataLoader):
    def __init__(self, *args, **kwargs):
        assert "name" in kwargs, "using a named dataloader requires passing name argument by keyword"
        self.name = kwargs.pop("name")
        super().__init__(*args, **kwargs)

    def __str__(self):
        return self.name


def getCIFAR(
    dir="cifar",
    bsize=64,
    returnTest=True,
    transforms=toTensor,
    shuffle=False,
    downloadIfNotExists=True,
):
    if not os.path.exists(os.path.join(dir, "cifar-10-batches-py")) and os.path.exists(
        os.path.join("..", dir, "cifar-10-batches-py")
    ):
        dir = os.path.join("..", dir)
    cifarTrain = CIFAR10(dir, True, transforms, download=downloadIfNotExists)
    cifarTest = CIFAR10(dir, False, transforms, download=downloadIfNotExists) if returnTest else None

    if returnTest:
        return (
            Dataloader_named(cifarTrain, bsize, True, name="CIFAR10"),
            Dataloader_named(cifarTest, bsize, shuffle, name="CIFAR10"),
        )
    return Dataloader_named(cifarTrain, bsize, shuffle, name="CIFAR10")


class CW_CNN(nn.Module):
    def __init__(self, inChannels=3, doKaimingInit=False, name=None):
        super().__init__()
        self.name = "CW_CNN" if name is None else name

        self.conv = nn.Sequential(
            nn.Conv2d(inChannels, 64, 3), nn.ReLU(),
            nn.Conv2d(64, 64, 3), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3), nn.ReLU(),
            nn.Conv2d(128, 128, 3), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
        )

        _n = 128 * 5 * 5
        self.fcn = nn.Sequential(
            nn.Linear(_n, 256), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 10),
        )

        if doKaimingInit:
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.fcn(self.conv(x))

    def __str__(self):
        return self.name


class MNIST_CNN(nn.Module):
    def __init__(self, inChannels=1, doKaimingInit=False, name=None):
        super().__init__()
        self.name = "MNIST_CNN" if name is None else name

        self.conv = nn.Sequential(
            nn.Conv2d(inChannels, 32, 3), nn.ReLU(),
            nn.Conv2d(32, 32, 3), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3), nn.ReLU(),
            nn.Conv2d(64, 64, 3), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
        )

        _n = 64 * 4 * 4
        self.fcn = nn.Sequential(
            nn.Linear(_n, 200), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(200, 200), nn.ReLU(),
            nn.Linear(200, 10),
        )

        if doKaimingInit:
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.fcn(self.conv(x))

    def __str__(self):
        return self.name


def train(
    model: nn.Module,
    datasets,
    epochs=10,
    lr=1e-2,
    lrDecay=None,
    temperature=None,
    verbose=True,
    lossf=nn.CrossEntropyLoss(),
    forceSGDoptim=False,
    tickSchedEvery=1,
    device=gpu,
    saveName=None,
):
    model.to(device).train()

    optim = (
        torch.optim.Adam(model.parameters(), lr=lr)
        if not forceSGDoptim
        else torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    )
    sched = (
        None
        if lrDecay is None
        else torch.optim.lr_scheduler.ExponentialLR(optim, lrDecay)
    )
    train_loader, test_loader = datasets
    temp = 1 if temperature is None else temperature

    for epoch in range(1, epochs + 1):
        sumTrainLoss = 0.0
        sumAcc = 0.0
        datapoints = 0
        iterator = enumerate(train_loader)
        if verbose:
            iterator = tqdm(iterator, total=len(train_loader))

        for _, (x, y) in iterator:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()

            out = model(x)
            loss = lossf(out / temp, y)
            loss.backward()
            optim.step()

            sumTrainLoss += loss.item() * x.size(0)
            datapoints += x.size(0)
            sumAcc += getAcc(out, y) * x.size(0)

            if verbose:
                iterator.set_description_str(
                    f"loss = {sumTrainLoss / datapoints}, acc = {sumAcc / datapoints}"
                )

        saveModel(f"latest_T{temp}.pt" if saveName is None else saveName, model)

        if test_loader is not None:
            testLoss, testAcc = holdoutTest(
                model,
                test_loader,
                lambda a, b: lossf(a / temp, b),
                verbose,
                device=device,
            )
            print(
                f"epoch {epoch} done "
                f"loss = {sumTrainLoss / datapoints} "
                f"test loss = {testLoss} "
                f"train acc = {sumAcc / datapoints} "
                f"test acc = {testAcc}"
            )

        if sched is not None:
            print(f"lr = {sched.get_last_lr()}")
            if epoch % tickSchedEvery == 0:
                sched.step()
