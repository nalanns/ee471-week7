
# EE 471 – Week 7: PyTorch Classifiers with Cog

Two image classifiers (FashionMNIST and CIFAR10). Each has a baseline model and a model tuned with Optuna. Both models are served with Cog, and a Flask website shows their predictions side by side.

## Results (test accuracy)

| Dataset      | Baseline | Optimized |
|--------------|----------|-----------|
| FashionMNIST | 64.4%    | 86.3%     |
| CIFAR10      | 52.7%    | 72.2%     |

## Requirements
- Docker Desktop
- Cog (`brew install replicate/tap/cog`)
- Python 3.11 with `torch torchvision optuna pandas`

```bash
conda create -n ee471 python=3.11 -y
conda activate ee471
pip install torch torchvision optuna pandas
```

## Structure
Each of `fashion/` and `cifar10/` contains:
- `train.py` – trains the baseline (`python train.py`) or a model from a config file
- `hpo.py` – Optuna search on a validation split, writes `best_params.json` and `hpo_results.csv`
- `model-baseline/`, `model-optimized/` – Cog packages (`cog.yaml`, `run.py`, `model.pth`)
- `web/` – Flask website that sends the image to both models
- `compose.yaml` – starts both models and the website

## Train (optional, trained weights are included)
```bash
cd fashion        # or cifar10
python train.py
python hpo.py
python train.py --config best_params.json --out optimized.pth
```

## Run
FashionMNIST:
```bash
cd fashion
(cd model-baseline && cog build -t fashion-baseline)
(cd model-optimized && cog build -t fashion-optimized)
docker compose up --build
```
Open http://localhost:8000

CIFAR10:
```bash
cd cifar10
(cd model-baseline && cog build -t cifar10-baseline)
(cd model-optimized && cog build -t cifar10-optimized)
docker compose up --build
```
Open http://localhost:8001

## Ports
| Service   | FashionMNIST | CIFAR10 |
|-----------|--------------|---------|
| Baseline  | 5051         | 5061    |
| Optimized | 5052         | 5062    |
| Website   | 8000         | 8001    |

## Live prediction logs
```bash
docker exec -it fashion-optimized tail -f /tmp/predictions.log
docker exec -it cifar10-optimized tail -f /tmp/predictions.log
```

## API (Postman / curl)
Cog accepts JSON only, so the image is sent as a base64 data URI:
```bash
curl http://localhost:5052/predictions -X POST \
  -H 'Content-Type: application/json' \
  -d "{\"input\": {\"image\": \"data:image/jpeg;base64,$(base64 -i image.jpg)\"}}"
```
