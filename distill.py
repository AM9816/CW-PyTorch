import torch
from torch import nn, Tensor as T
from torch.nn import functional
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from torch_common import getDevices, saveModel, getAcc, holdoutTest

cpu, gpu = getDevices()


def _distillLoss(logits: T, softLabels: T, temp=100.0):
    logProb = functional.log_softmax(logits / temp, dim=-1)
    return -(softLabels * logProb).sum(dim=-1).mean() * (temp ** 2)


def _genStudentDataset(model: nn.Module, dataloader: DataLoader, temp=100, device=gpu):
    with torch.no_grad():
        state = model.training
        model.eval().to(device)

        xs, ys, ps = [], [], []
        for x, y in dataloader:
            x = x.to(device)
            logits = model(x)
            probs = functional.softmax(logits / temp, dim=-1)
            xs.append(x.cpu())
            ys.append(y.cpu())
            ps.append(probs.cpu())

        model.train(state)
        return DataLoader(
            TensorDataset(torch.cat(xs), torch.cat(ys), torch.cat(ps)),
            dataloader.batch_size,
            True,
        )


def _trainStudent(
    student: nn.Module,
    teacher: nn.Module,
    datasets,
    temp=100,
    epochs=50,
    lr=1e-3,
    lrDecay=None,
    verbose=True,
    forceSGDoptim=False,
    schedTickEvery=1,
    device=gpu,
    dictSaveName="distil_dict.pt",
    modelSaveName="distill.pt",
):
    student, teacher = student.to(device).train(), teacher.to(device).eval()
    train_loader, test_loader = datasets

    softTrain = _genStudentDataset(teacher, train_loader, temp, device)

    optim = (
        torch.optim.Adam(student.parameters(), lr=lr)
        if not forceSGDoptim
        else torch.optim.SGD(student.parameters(), lr=lr, momentum=0.9)
    )
    sched = (
        None
        if lrDecay is None
        else torch.optim.lr_scheduler.ExponentialLR(optim, lrDecay)
    )

    testLossf = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        student.train()
        sumTrainLoss = sumAcc = datapoints = 0

        iterator = enumerate(softTrain)
        if verbose:
            iterator = tqdm(iterator, total=len(softTrain))

        for _, (x, y, soft) in iterator:
            x, y, soft = x.to(device), y.to(device), soft.to(device)

            optim.zero_grad()
            out = student(x)
            loss = _distillLoss(out, soft, temp)
            loss.backward()
            optim.step()

            with torch.no_grad():
                bsize = x.size(0)
                sumTrainLoss += loss.item() * bsize
                sumAcc += getAcc(out, y).item() * bsize
                datapoints += bsize

                status = (
                    f"[Student T={temp:g}] ep {epoch}/{epochs}: "
                    f"loss = {sumTrainLoss / datapoints:.4f}, "
                    f"acc = {sumAcc / datapoints:.2%}"
                )
                if verbose:
                    iterator.set_description_str(status)

        if sched is not None and epoch % schedTickEvery == 0:
            sched.step()
        if verbose:
            print(status)

        if dictSaveName is not None:
            saveModel(dictSaveName, student)
        if modelSaveName is not None:
            saveModel(modelSaveName, student, False)

        if test_loader is not None:
            with torch.no_grad():
                testLoss, testAcc = holdoutTest(student, test_loader, testLossf, verbose, device)
                if verbose:
                    print(f"test loss = {testLoss.item()}, test acc = {testAcc.item()}")

    return student
