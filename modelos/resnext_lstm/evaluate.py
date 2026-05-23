from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)

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


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def pick_device(device_name="auto"):
    if device_name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_name


def plot_confusion_matrix(cm, classes, output_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm)

    ax.set_title("Matriz de confusão - teste")
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_roc_curve(y_true, y_prob_fake, output_path):
    if len(np.unique(y_true)) < 2:
        return

    fpr, tpr, _ = roc_curve(y_true, y_prob_fake)
    auc = roc_auc_score(y_true, y_prob_fake)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--")

    ax.set_title("Curva ROC - teste")
    ax.set_xlabel("Taxa de Falsos Positivos")
    ax.set_ylabel("Taxa de Verdadeiros Positivos")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_precision_recall_curve(y_true, y_prob_fake, output_path):
    if len(np.unique(y_true)) < 2:
        return

    precision, recall, _ = precision_recall_curve(y_true, y_prob_fake)
    ap = average_precision_score(y_true, y_prob_fake)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, label=f"AP = {ap:.4f}")

    ax.set_title("Curva Precision-Recall - teste")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def evaluate_run(run_dir: Path, ignore_lrs=None):
    ignore_lrs = ignore_lrs or []

    config_path = run_dir / "config.json"
    best_model_path = run_dir / "best_model.pt"

    if not config_path.exists() or not best_model_path.exists():
        return None

    config = load_json(config_path)

    learning_rate = float(config.get("learning_rate", -1))
    if learning_rate in ignore_lrs:
        print(f"[IGNORADO] {run_dir.name} | learning_rate={learning_rate}")
        return None

    device = pick_device(config.get("device", "auto"))

    data_dir = resolve_from_root(config["data_dir"])
    seq_len = int(config.get("seq_len", 10))
    img_size = int(config.get("img_size", 256))
    batch_size = int(config.get("batch_size", 2))
    num_workers = int(config.get("num_workers", 0))
    pin_memory = bool(config.get("pin_memory", True))
    seed = int(config.get("seed", 42))

    test_ds = VideoFramesDataset(
        data_dir,
        seq_len=seq_len,
        img_size=img_size,
        split="test",
        seed=seed,
    )

    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device == "cuda" and pin_memory else False,
    )

    model = ResNeXtLSTM(
        hidden_dim=int(config.get("hidden_dim", 128)),
        num_layers=int(config.get("num_layers", 2)),
        num_classes=2,
        dropout=float(config.get("dropout", 0.5)),
        freeze_backbone=bool(config.get("freeze_backbone", True)),
    ).to(device)

    checkpoint = torch.load(best_model_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    y_true = []
    y_pred = []
    y_prob_fake = []
    video_ids = []

    with torch.no_grad():
        for frames, labels, vids in test_loader:
            frames = frames.to(device)
            labels = labels.to(device)

            outputs = model(frames)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob_fake.extend(probs[:, 1].cpu().numpy())
            video_ids.extend(list(vids))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob_fake = np.array(y_prob_fake)

    classes = ["real", "fake"]
    cm = confusion_matrix(y_true, y_pred)

    train_metrics_path = run_dir / "metrics.json"
    train_metrics = load_json(train_metrics_path) if train_metrics_path.exists() else {}

    test_metrics = {
        "run_name": config.get("run_name", run_dir.name),
        "run_dir": str(run_dir),
        "model_path": str(best_model_path),
        "config_file": str(config_path),
        "device": device,

        "model": "ResNeXt50_32x4d + LSTM",
        "seed": seed,
        "data_dir": str(data_dir),
        "seq_len": seq_len,
        "img_size": img_size,
        "batch_size": batch_size,

        "epochs_requested": int(config.get("epochs", config.get("epochs_requested", 0))),
        "epochs_executed": int(train_metrics.get("epochs_executed", config.get("epochs", 0))),

        "learning_rate": learning_rate,
        "optimizer": config.get("optimizer"),
        "dropout": float(config.get("dropout")),
        "weight_decay": float(config.get("weight_decay")),
        "hidden_dim": int(config.get("hidden_dim", 128)),
        "num_layers": int(config.get("num_layers", 2)),
        "freeze_backbone": bool(config.get("freeze_backbone", True)),

        "best_val_f1": train_metrics.get("best_val_f1"),
        "best_val_acc": train_metrics.get("best_val_acc"),
        "best_epoch": train_metrics.get("best_epoch"),
        "final_val_acc": train_metrics.get("final_val_acc"),
        "final_val_loss": train_metrics.get("final_val_loss"),
        "final_val_f1_macro": train_metrics.get("final_val_f1_macro"),
        "total_training_time_sec": train_metrics.get("total_training_time_sec"),

        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "test_precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "test_recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "test_f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),

        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=classes,
            output_dict=True,
            zero_division=0,
        ),

        "classes": classes,
        "confusion_matrix": cm.tolist(),
    }

    if len(np.unique(y_true)) >= 2:
        test_metrics["test_auc_roc"] = float(roc_auc_score(y_true, y_prob_fake))
        test_metrics["test_average_precision"] = float(average_precision_score(y_true, y_prob_fake))
    else:
        test_metrics["test_auc_roc"] = None
        test_metrics["test_average_precision"] = None

    df_pred = pd.DataFrame({
        "video_id": video_ids,
        "label": y_true,
        "label_name": [classes[i] for i in y_true],
        "pred": y_pred,
        "pred_name": [classes[i] for i in y_pred],
        "prob_fake": y_prob_fake,
        "correct": y_true == y_pred,
    })

    df_pred.to_csv(run_dir / "predictions_test.csv", index=False, encoding="utf-8")

    save_json(run_dir / "metrics_test.json", test_metrics)

    plot_confusion_matrix(cm, classes, run_dir / "confusion_matrix_test.png")
    plot_roc_curve(y_true, y_prob_fake, run_dir / "roc_curve_test.png")
    plot_precision_recall_curve(y_true, y_prob_fake, run_dir / "precision_recall_curve_test.png")

    print(
        f"[OK] {run_dir.name} | "
        f"F1={test_metrics['test_f1_macro']:.4f} | "
        f"Acc={test_metrics['test_accuracy']:.4f}"
    )

    return test_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Avalia todos os experimentos ResNeXt+LSTM no conjunto de teste."
    )

    parser.add_argument(
        "--experiments-root",
        default="experiments/resnext_lstm",
    )

    parser.add_argument(
        "--ignore-lr",
        nargs="*",
        type=float,
        default=[0.01],
        help="Learning rates que devem ser ignorados.",
    )

    args = parser.parse_args()

    experiments_root = resolve_from_root(args.experiments_root)

    run_dirs = sorted([
        p for p in experiments_root.rglob("*")
        if p.is_dir() and (p / "best_model.pt").exists() and (p / "config.json").exists()
    ])

    print("=" * 70)
    print("AVALIAÇÃO RESNEXT + LSTM")
    print("=" * 70)
    print(f"Experimentos encontrados: {len(run_dirs)}")
    print(f"Ignorando learning rates: {args.ignore_lr}")
    print("=" * 70)

    all_metrics = []

    for run_dir in run_dirs:
        result = evaluate_run(run_dir, ignore_lrs=args.ignore_lr)
        if result is not None:
            all_metrics.append(result)

    if not all_metrics:
        print("Nenhum experimento avaliado.")
        return

    reports_dir = resolve_from_root("reports/resnext_lstm/evaluation")
    reports_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(all_metrics)
    df = df.sort_values("test_f1_macro", ascending=False).reset_index(drop=True)

    df.to_csv(reports_dir / "resnext_lstm_avaliacoes_consolidadas.csv", index=False, encoding="utf-8")

    save_json(reports_dir / "melhor_experimento_teste.json", df.iloc[0].to_dict())

    print("=" * 70)
    print("Avaliação finalizada.")
    print(f"Total avaliados: {len(df)}")
    print(f"CSV: {reports_dir / 'resnext_lstm_avaliacoes_consolidadas.csv'}")
    print("=" * 70)


if __name__ == "__main__":
    main()