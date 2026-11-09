import argparse
import json
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor

CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

BASELINE = {"batch_size": 64, "lr": 1e-3, "optimizer": "sgd",
            "hidden": 512, "dropout": 0.0, "epochs": 5}


class NeuralNetwork(nn.Module):
    def __init__(self, hidden=512, dropout=0.0):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 10),
        )

    def forward(self, x):
        return self.linear_relu_stack(self.flatten(x))


def get_datasets():
    full = datasets.FashionMNIST(root="../data", train=True, download=True, transform=ToTensor())
    test = datasets.FashionMNIST(root="../data", train=False, download=True, transform=ToTensor())
    train, val = random_split(full, [50000, 10000], generator=torch.Generator().manual_seed(42))
    return train, val, test


def make_optimizer(name, params, lr):
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    return torch.optim.SGD(params, lr=lr)


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

    model = NeuralNetwork(config["hidden"], config["dropout"]).to(DEVICE)
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