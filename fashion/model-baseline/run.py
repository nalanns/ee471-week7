import logging
import torch
from torch import nn
from cog import BaseRunner, Input, Path
from PIL import Image, ImageOps, ImageStat
from torchvision.transforms import ToTensor

CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


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


class Runner(BaseRunner):
    def setup(self):
        self.log = logging.getLogger("fashion-model")
        self.log.setLevel(logging.INFO)
        handler = logging.FileHandler("/tmp/predictions.log")
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        self.log.addHandler(handler)

        ckpt = torch.load("model.pth", map_location="cpu")
        cfg = ckpt["config"]
        self.model = NeuralNetwork(cfg["hidden"], cfg["dropout"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.log.info("model loaded config=%s test_acc=%.4f", cfg, ckpt["test_acc"])

    def run(self, image: Path = Input(description="Garment image")) -> dict:
        img = Image.open(image).convert("L")
        if ImageStat.Stat(img).mean[0] > 127:
            img = ImageOps.invert(img)
        img = ImageOps.pad(img, (28, 28), color=0)
        with torch.no_grad():
            probs = self.model(ToTensor()(img).unsqueeze(0))[0].softmax(0)
        top3 = probs.topk(3)
        result = {CLASSES[i]: p.item() for p, i in zip(*top3)}
        self.log.info("image=%s prediction=%s", image.name, result)
        return result