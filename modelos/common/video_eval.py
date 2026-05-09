from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torchvision import transforms

from modelos.common.paths import resolve_from_root
from modelos.common.utils import ensure_dir, pick_device


CLASS_NAMES = ["fake", "real"]


def get_video_id(image_path: Path) -> str:
    return image_path.stem.split("_frame")[0]


def list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []

    return sorted([
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ])


def group_frames_by_video(faces_dir: Path) -> dict[str, list[Path]]:
    grouped = defaultdict(list)

    for class_name in CLASS_NAMES:
        class_dir = faces_dir / class_name

        if not class_dir.exists():
            raise FileNotFoundError(f"Pasta não encontrada: {class_dir}")

        for img_path in list_images(class_dir):
            video_id = get_video_id(img_path)
            full_video_id = f"{class_name}_{video_id}"
            grouped[full_video_id].append(img_path)

    return dict(grouped)


def predict_image(model, image_path: Path, transform, device: torch.device) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    return probs


def save_confusion_matrix(
    cm: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm)

    ax.set_title(title)
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_yticklabels(CLASS_NAMES)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def evaluate_videos_from_faces(
    model_name: str,
    model_builder: Callable[[], torch.nn.Module],
    model_path: str | Path,
    faces_dir: str | Path = "data/processed/faces_256_interval_10",
    output_dir: str | Path = "reports/videos",
    img_size: int = 256,
    device_name: str = "auto",
) -> dict:
    model_path = resolve_from_root(model_path)
    faces_dir = resolve_from_root(faces_dir)
    output_dir = ensure_dir(resolve_from_root(output_dir))

    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint não encontrado: {model_path}")

    device = pick_device(device_name)

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    model = model_builder().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    grouped = group_frames_by_video(faces_dir)

    rows = []
    y_true = []
    y_pred = []

    for video_key, frame_paths in grouped.items():
        true_class = video_key.split("_")[0]
        true_label = CLASS_NAMES.index(true_class)

        probs_list = []

        for frame_path in frame_paths:
            probs = predict_image(model, frame_path, transform, device)
            probs_list.append(probs)

        mean_probs = np.mean(probs_list, axis=0)
        pred_label = int(np.argmax(mean_probs))
        pred_class = CLASS_NAMES[pred_label]

        y_true.append(true_label)
        y_pred.append(pred_label)

        rows.append({
            "video_id": video_key,
            "true_class": true_class,
            "pred_class": pred_class,
            "prob_fake": float(mean_probs[0]),
            "prob_real": float(mean_probs[1]),
            "num_frames": len(frame_paths),
            "correct": pred_label == true_label,
        })

    if not rows:
        raise RuntimeError(f"Nenhum vídeo/frame encontrado em: {faces_dir}")

    csv_path = output_dir / f"resultados_videos_{model_name}.csv"

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "model_name": model_name,
        "model_path": str(model_path),
        "faces_dir": str(faces_dir),
        "total_videos": len(rows),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": cm.tolist(),
    }

    metrics_path = output_dir / f"metrics_videos_{model_name}.json"

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    confusion_path = output_dir / f"confusion_matrix_videos_{model_name}.png"

    save_confusion_matrix(
        cm=cm,
        output_path=confusion_path,
        title=f"Matriz de confusão em vídeos - {model_name}",
    )

    print("\nAvaliação concluída!")
    print(f"Modelo: {model_name}")
    print(f"Total de vídeos: {metrics['total_videos']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision macro: {metrics['precision_macro']:.4f}")
    print(f"Recall macro: {metrics['recall_macro']:.4f}")
    print(f"F1 macro: {metrics['f1_macro']:.4f}")
    print(f"CSV salvo em: {csv_path}")
    print(f"Métricas salvas em: {metrics_path}")
    print(f"Matriz de confusão salva em: {confusion_path}")

    return metrics