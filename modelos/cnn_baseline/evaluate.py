from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modelos.common.data import build_imagefolder_loaders
from modelos.common.paths import resolve_from_root
from modelos.common.utils import ensure_dir, pick_device
from modelos.cnn_baseline.model import CNNBaseline


def evaluate_checkpoint(
    model_path: str | Path,
    split_dir: str | Path,
    img_size: int = 256,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = False,
    dropout_fc: float = 0.5,
    device_name: str = "auto",
    output_dir: str | Path | None = None,
) -> dict:
    model_path = resolve_from_root(model_path)
    split_dir = resolve_from_root(split_dir)
    output_dir = resolve_from_root(output_dir) if output_dir else model_path.parent
    ensure_dir(output_dir)

    device = pick_device(device_name)
    _, _, test_loader, classes = build_imagefolder_loaders(
        split_dir=split_dir,
        img_size=img_size,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model = CNNBaseline(
        num_classes=len(classes),
        dropout_fc=dropout_fc,
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(labels.numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "test_accuracy": float((y_true == y_pred).mean()),
        "test_precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "test_recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "test_f1_macro": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "classification_report": report,
        "classes": classes,
        "confusion_matrix": cm.tolist(),
    }

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
    fig.savefig(output_dir / "confusion_matrix.png", dpi=200)
    plt.close(fig)

    with open(output_dir / "metrics_test.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Avaliar checkpoint da CNN no conjunto de teste.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--split-dir", default="data/splits/split_01_interval_10")
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--dropout-fc", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    metrics = evaluate_checkpoint(
        model_path=args.model_path,
        split_dir=args.split_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        dropout_fc=args.dropout_fc,
        device_name=args.device,
        output_dir=args.output_dir,
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()