from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modelos.common.data import build_imagefolder_loaders
from modelos.common.paths import resolve_from_root
from modelos.common.utils import (
    append_row_to_csv,
    copy_if_best,
    ensure_dir,
    load_json,
    pick_device,
    save_json,
    set_seed,
    timestamp_now,
)
from modelos.cnn_baseline.evaluate import evaluate_checkpoint
from modelos.cnn_baseline.model import CNNBaseline


def build_optimizer(
    name: str,
    model: torch.nn.Module,
    learning_rate: float,
    weight_decay: float,
):
    name_lower = name.lower()

    if name_lower == "adam":
        return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if name_lower == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay,
        )

    if name_lower == "adamw":
        return optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if name_lower == "rmsprop":
        return optim.RMSprop(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay,
        )

    raise ValueError(f"Optimizer não suportado: {name}")


class EarlyStopping:
    def __init__(self, patience: int = 10):
        self.patience = patience
        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def step(self, current_score: float) -> bool:
        if self.best_score is None or current_score > self.best_score:
            self.best_score = current_score
            self.counter = 0
            return True

        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True

        return False


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / max(total, 1)
    avg_acc = correct / max(total, 1)
    return avg_loss, avg_acc


def save_history_csv(history: list[dict], history_path: Path) -> None:
    ensure_dir(history_path.parent)
    with open(history_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def plot_history(history: list[dict], output_dir: Path) -> None:
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]
    train_acc = [row["train_acc"] for row in history]
    val_acc = [row["val_acc"] for row in history]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_loss, label="Train loss")
    ax.plot(epochs, val_loss, label="Val loss")
    ax.set_xlabel("Época")
    ax.set_ylabel("Loss")
    ax.set_title("Curva de loss")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "grafico_loss.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_acc, label="Train acc")
    ax.plot(epochs, val_acc, label="Val acc")
    ax.set_xlabel("Época")
    ax.set_ylabel("Acurácia")
    ax.set_title("Curva de acurácia")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "grafico_acuracia.png", dpi=200)
    plt.close(fig)


def build_run_name(config: dict) -> str:
    prefix = config.get("run_prefix", "cnn_baseline")
    return (
        f"{prefix}_{timestamp_now()}"
        f"_bs{config['batch_size']}"
        f"_ep{config['epochs']}"
        f"_lr{config['learning_rate']}"
        f"_opt{config['optimizer']}"
        f"_df{config['dropout_fc']}"
        f"_wd{config['weight_decay']}"
    )


def train_single_run(config: dict) -> dict:
    seed = int(config.get("seed", 42))
    set_seed(seed)

    device = pick_device(config.get("device", "auto"))
    split_dir = resolve_from_root(config["data_split_dir"])
    experiments_dir = resolve_from_root(config["experiments_dir"])
    reports_csv = resolve_from_root(config["reports_csv"])
    best_overall_checkpoint = resolve_from_root(config["best_overall_checkpoint"])

    ensure_dir(experiments_dir)
    ensure_dir(reports_csv.parent)
    ensure_dir(best_overall_checkpoint.parent)

    run_name = config.get("run_name") or build_run_name(config)
    run_dir = ensure_dir(experiments_dir / run_name)

    train_loader, val_loader, _, classes = build_imagefolder_loaders(
        split_dir=split_dir,
        img_size=int(config["img_size"]),
        batch_size=int(config["batch_size"]),
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=bool(config.get("pin_memory", False)),
    )

    model = CNNBaseline(
        num_classes=len(classes),
        dropout_fc=float(config["dropout_fc"]),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(
        config["optimizer"],
        model,
        float(config["learning_rate"]),
        float(config.get("weight_decay", 0.0)),
    )

    best_model_path = run_dir / "best_model.pt"
    last_model_path = run_dir / "last_model.pt"
    config_path = run_dir / "config.json"
    history_path = run_dir / "history.csv"
    metrics_path = run_dir / "metrics.json"

    save_json(config_path, config)

    use_early_stopping = bool(config.get("use_early_stopping", True))
    early_stopping = EarlyStopping(
        patience=int(config.get("early_stopping_patience", 10))
    )

    history: list[dict] = []
    best_val_acc = 0.0
    best_epoch = 0
    start_time = time.time()

    for epoch in range(1, int(config["epochs"]) + 1):
        epoch_start = time.time()

        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False
        )

        epoch_time = time.time() - epoch_start
        total_elapsed = time.time() - start_time

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 6),
            "epoch_time_sec": round(epoch_time, 4),
            "total_time_sec": round(total_elapsed, 4),
        }
        history.append(row)

        print(
            f"[Epoch {epoch:03d}/{config['epochs']}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        improved = early_stopping.step(val_acc)
        if improved:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)

        if use_early_stopping and early_stopping.should_stop:
            print(f"Early stopping acionado na época {epoch}.")
            break

    if not best_model_path.exists():
        torch.save(model.state_dict(), best_model_path)
        if history:
            best_epoch = int(history[-1]["epoch"])
            best_val_acc = float(history[-1]["val_acc"])

    if bool(config.get("save_last_model", True)):
        torch.save(model.state_dict(), last_model_path)

    if history:
        save_history_csv(history, history_path)
        plot_history(history, run_dir)

    test_metrics = evaluate_checkpoint(
        model_path=best_model_path,
        split_dir=split_dir,
        img_size=int(config["img_size"]),
        batch_size=int(config["batch_size"]),
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=bool(config.get("pin_memory", False)),
        dropout_fc=float(config["dropout_fc"]),
        device_name=config.get("device", "auto"),
        output_dir=run_dir,
    )

    final_metrics = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "seed": seed,
        "device": str(device),
        "img_size": int(config["img_size"]),
        "batch_size": int(config["batch_size"]),
        "epochs_requested": int(config["epochs"]),
        "epochs_executed": len(history),
        "learning_rate": float(config["learning_rate"]),
        "optimizer": config["optimizer"],
        "dropout_fc": float(config["dropout_fc"]),
        "weight_decay": float(config.get("weight_decay", 0.0)),
        "use_early_stopping": use_early_stopping,
        "early_stopping_patience": int(config.get("early_stopping_patience", 10)),
        "best_val_acc": round(float(best_val_acc), 6),
        "best_epoch": int(best_epoch),
        "final_val_acc": round(float(history[-1]["val_acc"]), 6),
        "final_val_loss": round(float(history[-1]["val_loss"]), 6),
        "total_training_time_sec": round(float(history[-1]["total_time_sec"]), 4),
        "test_accuracy": round(float(test_metrics["test_accuracy"]), 6),
        "test_precision_macro": round(float(test_metrics["test_precision_macro"]), 6),
        "test_recall_macro": round(float(test_metrics["test_recall_macro"]), 6),
        "test_f1_macro": round(float(test_metrics["test_f1_macro"]), 6),
        "best_model_path": str(best_model_path),
    }

    save_json(metrics_path, final_metrics)
    append_row_to_csv(reports_csv, final_metrics)

    current_best_metric = final_metrics["test_f1_macro"]
    overall_best_path = best_overall_checkpoint.with_suffix(".json")
    should_replace_overall = True

    if overall_best_path.exists():
        previous = load_json(overall_best_path)
        should_replace_overall = current_best_metric > float(
            previous.get("test_f1_macro", -1)
        )

    if should_replace_overall:
        copy_if_best(best_model_path, best_overall_checkpoint)
        save_json(overall_best_path, final_metrics)

    return final_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina uma configuração da CNN baseline.")
    parser.add_argument("--config", required=True, help="Caminho do JSON de configuração.")
    args = parser.parse_args()

    config = load_json(resolve_from_root(args.config))
    metrics = train_single_run(config)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()