import argparse
import json
import optuna
from torch import nn
from torch.utils.data import DataLoader
from train import build_model, get_datasets, make_optimizer, train_epoch, evaluate, DEVICE

EPOCHS = 10

train_set, val_set, _ = get_datasets()
val_loader = DataLoader(val_set, batch_size=1000)


def objective(trial):
    config = {
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        "lr": trial.suggest_float("lr", 1e-4, 1e-1, log=True),
        "optimizer": trial.suggest_categorical("optimizer", ["sgd", "adam"]),
        "c1": trial.suggest_categorical("c1", [6, 16, 32]),
        "c2": trial.suggest_categorical("c2", [16, 32, 64]),
        "fc": trial.suggest_categorical("fc", [120, 256, 512]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
    }
    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)
    model = build_model(config).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = make_optimizer(config["optimizer"], model.parameters(), config["lr"])

    for epoch in range(EPOCHS):
        train_epoch(model, train_loader, loss_fn, optimizer)
        acc = evaluate(model, val_loader)
        trial.report(acc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=20)
    args = parser.parse_args()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
    )
    study.optimize(objective, n_trials=args.trials)

    best = dict(study.best_params, epochs=EPOCHS)
    with open("best_params.json", "w") as f:
        json.dump(best, f, indent=2)
    study.trials_dataframe().to_csv("hpo_results.csv", index=False)

    print("best val acc:", study.best_value)
    print("best params:", best)