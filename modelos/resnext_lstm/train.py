import sys
import json
import time
import argparse
import random
from pathlib import Path
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import matplotlib.pyplot as plt

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modelos.resnext_lstm.data import VideoFramesDataset
from modelos.resnext_lstm.model import ResNeXtLSTM


def resolve_from_root(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(device_name):
    if device_name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_name


def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def create_optimizer(model, optimizer_name, learning_rate, weight_decay):
    params = filter(lambda p: p.requires_grad, model.parameters())

    if optimizer_name == "Adam":
        return torch.optim.Adam(
            params,
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    if optimizer_name == "SGD":
        return torch.optim.SGD(
            params,
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay,
        )
    
    if optimizer_name == "AdamW":
        return torch.optim.AdamW(
            params,
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    if optimizer_name == "RMSprop":
        return torch.optim.RMSprop(
            params,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=0.9,
        )

    raise ValueError(f"Otimizador inválido: {optimizer_name}")


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0
    all_labels = []
    all_preds = []

    for frames, labels, _ in loader:
        frames = frames.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(frames)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    avg_loss = total_loss / len(loader)
    metrics = calculate_metrics(all_labels, all_preds)

    return avg_loss, metrics


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for frames, labels, _ in loader:
            frames = frames.to(device)
            labels = labels.to(device)

            outputs = model(frames)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    avg_loss = total_loss / len(loader)
    metrics = calculate_metrics(all_labels, all_preds)

    return avg_loss, metrics


def save_training_plots(history_df, run_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history_df["epoch"], history_df["train_loss"], label="Treino")
    ax.plot(history_df["epoch"], history_df["val_loss"], label="Validação")
    ax.set_title("Loss por época")
    ax.set_xlabel("Época")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(run_dir / "grafico_loss.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history_df["epoch"], history_df["train_accuracy"], label="Treino")
    ax.plot(history_df["epoch"], history_df["val_accuracy"], label="Validação")
    ax.set_title("Acurácia por época")
    ax.set_xlabel("Época")
    ax.set_ylabel("Acurácia")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(run_dir / "grafico_accuracy.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history_df["epoch"], history_df["train_f1_macro"], label="Treino")
    ax.plot(history_df["epoch"], history_df["val_f1_macro"], label="Validação")
    ax.set_title("F1-macro por época")
    ax.set_xlabel("Época")
    ax.set_ylabel("F1-macro")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(run_dir / "grafico_f1_macro.png", dpi=200)
    plt.close(fig)


def run_experiment(base_config, exp_config, exp_index, total_exps):
    seed = base_config["seed"]
    set_seed(seed)

    device = pick_device(base_config.get("device", "auto"))

    data_dir = base_config["data_dir"]
    img_size = base_config["img_size"]
    batch_size = base_config["batch_size"]
    num_workers = base_config.get("num_workers", 0)
    pin_memory = base_config.get("pin_memory", True)

    hidden_dim = base_config["hidden_dim"]
    num_layers = base_config["num_layers"]
    freeze_backbone = base_config["freeze_backbone"]

    optimizer_name = exp_config["optimizer"]
    seq_len = exp_config["seq_len"]
    epochs = exp_config["epochs"]
    learning_rate = exp_config["learning_rate"]
    dropout = exp_config["dropout"]
    weight_decay = exp_config["weight_decay"]

    run_prefix = base_config.get("run_prefix", "resnext_lstm")
    run_name = datetime.now().strftime(
        f"{run_prefix}_%Y-%m-%d_%H-%M-%S"
        f"_bs{batch_size}"
        f"_ep{epochs}"
        f"_lr{learning_rate}"
        f"_opt{optimizer_name}"
        f"_seq{seq_len}"
        f"_drop{dropout}"
        f"_wd{weight_decay}"
    )

    experiments_dir = resolve_from_root(base_config["experiments_dir"])
    experiments_dir.mkdir(parents=True, exist_ok=True)

    run_dir = experiments_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = run_dir / "best_model.pt"
    last_model_path = run_dir / "last_model.pt"
    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.json"
    history_json_path = run_dir / "history.json"
    history_csv_path = run_dir / "history.csv"

    full_config = {
        **base_config,
        **exp_config,
        "run_name": run_name,
        "run_dir": str(run_dir.resolve()),
        "device_used": device,
        "model": "ResNeXt50_32x4d + LSTM",
    }

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(full_config, f, indent=4, ensure_ascii=False)

    print("=" * 70)
    print(f"EXPERIMENTO {exp_index}/{total_exps}")
    print("TREINAMENTO RESNEXT + LSTM")
    print("=" * 70)
    print(f"Run: {run_name}")
    print(f"Device: {device}")
    print(f"Seq len: {seq_len}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Optimizer: {optimizer_name}")
    print(f"Learning rate: {learning_rate}")
    print(f"Weight decay: {weight_decay}")
    print(f"Dropout: {dropout}")
    print(f"Freeze backbone: {freeze_backbone}")
    print("=" * 70)

    train_ds = VideoFramesDataset(
        data_dir,
        seq_len=seq_len,
        img_size=img_size,
        split="train",
    )

    val_ds = VideoFramesDataset(
        data_dir,
        seq_len=seq_len,
        img_size=img_size,
        split="val",
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device == "cuda" and pin_memory else False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device == "cuda" and pin_memory else False,
    )

    model = ResNeXtLSTM(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=2,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = create_optimizer(
        model=model,
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    history = []
    best_val_f1 = -1
    best_val_acc = -1
    best_epoch = 0

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        val_loss, val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        epoch_data = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_metrics["accuracy"],
            "train_precision_macro": train_metrics["precision_macro"],
            "train_recall_macro": train_metrics["recall_macro"],
            "train_f1_macro": train_metrics["f1_macro"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision_macro": val_metrics["precision_macro"],
            "val_recall_macro": val_metrics["recall_macro"],
            "val_f1_macro": val_metrics["f1_macro"],
        }

        history.append(epoch_data)

        print(
            f"Epoch [{epoch:03d}/{epochs}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Train F1: {train_metrics['f1_macro']:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val F1: {val_metrics['f1_macro']:.4f}"
        )

        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            best_val_acc = val_metrics["accuracy"]
            best_epoch = epoch

            torch.save(model.state_dict(), best_model_path)

            print(f"  Novo melhor modelo salvo: {best_model_path}")

    torch.save(model.state_dict(), last_model_path)

    total_training_time_sec = time.time() - start_time
    history_df = pd.DataFrame(history)
    final_epoch = history[-1]

    history_df.to_csv(history_csv_path, index=False, encoding="utf-8")

    with open(history_json_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4, ensure_ascii=False)

    save_training_plots(history_df, run_dir)

    metrics = {
        "run_name": run_name,
        "run_dir": str(run_dir.resolve()),
        "seed": seed,
        "device": device,
        "model": "ResNeXt50_32x4d + LSTM",

        "data_dir": data_dir,
        "seq_len": seq_len,
        "img_size": img_size,
        "batch_size": batch_size,
        "epochs_requested": epochs,
        "epochs_executed": len(history),

        "learning_rate": learning_rate,
        "optimizer": optimizer_name,
        "weight_decay": weight_decay,

        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "dropout": dropout,
        "freeze_backbone": freeze_backbone,

        "best_val_f1": best_val_f1,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,

        "final_val_acc": final_epoch["val_accuracy"],
        "final_val_loss": final_epoch["val_loss"],
        "final_val_f1_macro": final_epoch["val_f1_macro"],

        "total_training_time_sec": round(total_training_time_sec, 4),
        "best_model_path": str(best_model_path.resolve()),
        "last_model_path": str(last_model_path.resolve()),
        "config_file": str(config_path.resolve()),
        "history_file": str(history_csv_path.resolve()),
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    print("=" * 70)
    print("Treinamento finalizado.")
    print(f"Melhor F1 validação: {best_val_f1:.4f}")
    print(f"Melhor época: {best_epoch}")
    print(f"Tempo total: {total_training_time_sec:.2f} segundos")
    print(f"Run dir: {run_dir}")
    print("=" * 70)

    return metrics


def build_grid(search_space):
    keys = list(search_space.keys())
    values = [search_space[k] for k in keys]

    configs = []

    for combination in product(*values):
        configs.append(dict(zip(keys, combination)))

    return configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/resnext_lstm_grid.json",
        help="Caminho para o arquivo JSON de configuração.",
    )
    args = parser.parse_args()

    config_path = resolve_from_root(args.config)

    with open(config_path, "r", encoding="utf-8") as f:
        base_config = json.load(f)

    search_space = base_config.pop("search_space")
    grid = build_grid(search_space)

    print("=" * 70)
    print("GRID SEARCH RESNEXT + LSTM")
    print("=" * 70)
    print(f"Config: {config_path}")
    print(f"Total de experimentos: {len(grid)}")
    print("=" * 70)

    all_results = []

    for idx, exp_config in enumerate(grid, start=1):
        metrics = run_experiment(base_config, exp_config, idx, len(grid))
        all_results.append(metrics)

    results_df = pd.DataFrame(all_results)
    out_dir = resolve_from_root(base_config["experiments_dir"])
    results_df.to_csv(out_dir / "grid_results.csv", index=False, encoding="utf-8")

    print("Grid search finalizado.")
    print(f"Resultados gerais salvos em: {out_dir / 'grid_results.csv'}")


if __name__ == "__main__":
    main()