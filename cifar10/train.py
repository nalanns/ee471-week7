import argparse
import json
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

CLASSES = ["plane", "car", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

BASELINE = {"batch_size": 4, "lr": 1e-3, "optimizer": "sgd",
            "c1": 6, "c2": 16, "fc": 120, "dropout": 0.0, "epochs": 2}

TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


class Net(nn.Module):
    def __init__(self, c1=6, c2=16, fc=120, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(3, c1, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(c1, c2, 5)
        self.fc1 = nn.Linear(c2 * 5 * 5, fc)
        self.fc2 = nn.Linear(fc, 84)
        self.fc3 = nn.Linear(84, 10)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        return self.fc3(x)


def build_model(config):
    return Net(config["c1"], config["c2"], config["fc"], config["dropout"])


def get_datasets():
    full = datasets.CIFAR10(root="../data", train=True, download=True, transform=TRANSFORM)
    test = datasets.CIFAR10(root="../data", train=False, download=True, transform=TRANSFORM)
    train, val = random_split(full, [45000, 5000], generator=torch.Generator().manual_seed(42))
    return train, val, test


def make_optimizer(name, params, lr):
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    return torch.optim.SGD(params, lr=lr, momentum=0.9)


def train_epoch(model, loader, loss_fn, optimizer):
    model.train()
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        loss = loss_fn(model(X), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def evaluate(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            correct += (model(X).argmax(1) == y).sum().item()
    return correct / len(loader.dataset)


def run(config):
    train_set, val_set, test_set = get_datasets()
    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1000)
    test_loader = DataLoader(test_set, batch_size=1000)

    model = build_model(config).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = make_optimizer(config["optimizer"], model.parameters(), config["lr"])

    for epoch in range(config["epochs"]):
        train_epoch(model, train_loader, loss_fn, optimizer)
        print(f"epoch {epoch + 1}: val acc {evaluate(model, val_loader):.4f}")

    test_acc = evaluate(model, test_loader)
    print(f"test acc {test_acc:.4f}")
    return model, test_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", default="baseline.pth")
    args = parser.parse_args()

    config = BASELINE
    if args.config:
        with open(args.config) as f:
            config = json.load(f)

    model, test_acc = run(config)
    torch.save({"config": config, "state_dict": model.to("cpu").state_dict(),
                "test_acc": test_acc}, args.out)
    print(f"saved {args.out}")