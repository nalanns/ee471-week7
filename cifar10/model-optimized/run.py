import logging
import torch
from torch import nn
import torch.nn.functional as F
from cog import BaseRunner, Input, Path
from PIL import Image, ImageOps
from torchvision import transforms

CLASSES = ["plane", "car", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

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


class Runner(BaseRunner):
    def setup(self):
        self.log = logging.getLogger("cifar10-model")
        self.log.setLevel(logging.INFO)
        handler = logging.FileHandler("/tmp/predictions.log")
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        self.log.addHandler(handler)

        ckpt = torch.load("model.pth", map_location="cpu")
        cfg = ckpt["config"]
        self.model = Net(cfg["c1"], cfg["c2"], cfg["fc"], cfg["dropout"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.log.info("model loaded config=%s test_acc=%.4f", cfg, ckpt["test_acc"])

    def run(self, image: Path = Input(description="Image to classify")) -> dict:
        img = ImageOps.fit(Image.open(image).convert("RGB"), (32, 32))
        with torch.no_grad():
            probs = self.model(TRANSFORM(img).unsqueeze(0))[0].softmax(0)
        top3 = probs.topk(3)
        result = {CLASSES[i]: p.item() for p, i in zip(*top3)}
        self.log.info("image=%s prediction=%s", image.name, result)
        return result