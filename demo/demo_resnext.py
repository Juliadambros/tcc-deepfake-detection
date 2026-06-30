from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

CURRENT_FILE = Path(__file__).resolve()


def find_project_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "modelos").exists() and (parent / "data").exists():
            return parent
    raise RuntimeError("Não foi possível encontrar a raiz do projeto.")


PROJECT_ROOT = find_project_root(CURRENT_FILE)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modelos.resnext_lstm.data import VideoFramesDataset
from modelos.resnext_lstm.model import ResNeXtLSTM


CLASSES = ["real", "fake"]


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_from_root(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def denormalize(img_tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    img = img_tensor.cpu() * std + mean
    img = torch.clamp(img, 0, 1)
    img = img.permute(1, 2, 0).numpy()

    return img


def load_model(config: dict, model_path: Path, device: str) -> ResNeXtLSTM:
    model = ResNeXtLSTM(
        hidden_dim=int(config.get("hidden_dim", 128)),
        num_layers=int(config.get("num_layers", 2)),
        num_classes=2,
        dropout=float(config.get("dropout", 0.5)),
        freeze_backbone=bool(config.get("freeze_backbone", False)),
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def predict_dataset(model, dataset, device: str):
    y_true = []
    y_pred = []
    y_prob_fake = []
    video_ids = []

    with torch.no_grad():
        for frames, label, video_id in dataset:
            frames_batch = frames.unsqueeze(0).to(device)

            outputs = model(frames_batch)
            probs = torch.softmax(outputs, dim=1)[0]

            pred = int(torch.argmax(probs).item())
            prob_fake = float(probs[1].item())

            y_true.append(int(label.item()))
            y_pred.append(pred)
            y_prob_fake.append(prob_fake)
            video_ids.append(video_id)

    return np.array(y_true), np.array(y_pred), np.array(y_prob_fake), video_ids


def choose_sample(dataset, y_true, y_pred, y_prob_fake, video_ids, mode: str, chosen_video_id: str | None):
    if chosen_video_id:
        for idx, vid in enumerate(video_ids):
            if vid == chosen_video_id:
                return idx

        raise ValueError(f"Vídeo '{chosen_video_id}' não encontrado no conjunto de teste.")

    confidences = np.abs(y_prob_fake - 0.5)

    if mode == "hardest":
        return int(np.argmin(confidences))

    if mode == "wrong":
        wrong = np.where(y_true != y_pred)[0]
        if len(wrong) > 0:
            return int(wrong[0])
        return int(np.argmin(confidences))

    if mode == "fake":
        fake = np.where(y_true == 1)[0]
        return int(fake[0])

    if mode == "real":
        real = np.where(y_true == 0)[0]
        return int(real[0])

    return 0



def make_demo(
    run_dir: Path,
    out_path: Path,
    mode: str = "hardest",
    chosen_video_id: str | None = None,
):
    config_path = run_dir / "config.json"
    model_path = run_dir / "best_model.pt"
    metrics_path = run_dir / "metrics_test.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Não encontrei config.json em: {config_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"Não encontrei best_model.pt em: {model_path}")

    config = load_json(config_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_dir = resolve_from_root(config["data_dir"])
    seq_len = int(config.get("seq_len", 10))
    img_size = int(config.get("img_size", 256))
    seed = int(config.get("seed", 42))

    dataset = VideoFramesDataset(
        data_dir,
        seq_len=seq_len,
        img_size=img_size,
        split="test",
        seed=seed,
    )

    model = load_model(config, model_path, device)

    y_true, y_pred, y_prob_fake, video_ids = predict_dataset(model, dataset, device)

    print("\nVídeos disponíveis:\n")

    for vid in sorted(video_ids):
        print(vid)

    sample_idx = choose_sample(
        dataset=dataset,
        y_true=y_true,
        y_pred=y_pred,
        y_prob_fake=y_prob_fake,
        video_ids=video_ids,
        mode=mode,
        chosen_video_id=chosen_video_id,
    )

    frames, label, video_id = dataset[sample_idx]

    prob_fake = y_prob_fake[sample_idx]
    prob_real = 1.0 - prob_fake

    pred_idx = y_pred[sample_idx]
    true_idx = y_true[sample_idx]

    pred_name = CLASSES[pred_idx]
    true_name = CLASSES[true_idx]


    if metrics_path.exists():
        metrics = load_json(metrics_path)
        acc = float(metrics.get("test_accuracy", 0))
        precision = float(metrics.get("test_precision_macro", 0))
        recall = float(metrics.get("test_recall_macro", 0))
        f1 = float(metrics.get("test_f1_macro", 0))
        auc = metrics.get("test_auc_roc", None)
    else:
        acc = float(np.mean(y_true == y_pred))
        precision = recall = f1 = auc = None

    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_show = min(seq_len, 6)
    frame_indices = np.linspace(0, seq_len - 1, n_show).astype(int)

    fig = plt.figure(figsize=(22, 11))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.2, 1])

    for col, frame_idx in enumerate(frame_indices):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(denormalize(frames[frame_idx]))
        ax.set_title(f"Frame {frame_idx + 1}", fontsize=16, fontweight="bold")
        ax.axis("off")

    ax_info = fig.add_subplot(gs[1, 0:3])
    ax_info.axis("off")

    status = "correta" if pred_idx == true_idx else "incorreta"

    info_text = (
        "ResNeXt+LSTM\n\n"
        f"Vídeo: {video_id}\n"
        f"Classe real: {true_name}\n"
        f"Predição: {pred_name} ({status})\n\n"
        f"Prob. real: {prob_real:.2%}\n"
        f"Prob. fake: {prob_fake:.2%}\n\n"
        f"Seq_len: {seq_len}\n"
        f"Backbone congelado: {config.get('freeze_backbone', '-')}"
    )

    ax_info.text(
        0.02,
        0.98,
        info_text,
        va="top",
        ha="left",
        fontsize=21,
        fontweight="bold",
        bbox=dict(boxstyle="round", alpha=0.12),
    )

    ax_bar = fig.add_subplot(gs[1, 3:6])

    bars = ax_bar.bar(
        ["Real", "Fake"],
        [prob_real, prob_fake],
        width=0.65
    )

    ax_bar.set_ylim(0, 1)

    ax_bar.set_title(
        "Probabilidades da amostra",
        fontsize=22,
        fontweight="bold",
        pad=15
    )

    ax_bar.set_ylabel(
        "Probabilidade",
        fontsize=18,
        fontweight="bold"
    )

    ax_bar.tick_params(
        axis="both",
        labelsize=16
    )

    for lbl in ax_bar.get_xticklabels():
        lbl.set_fontweight("bold")

    for lbl in ax_bar.get_yticklabels():
        lbl.set_fontweight("bold")

    ax_bar.grid(
        axis="y",
        linestyle="--",
        linewidth=1.4,
        alpha=0.8
    )

    for i, value in enumerate([prob_real, prob_fake]):
        ax_bar.text(
            i,
            value + 0.03,
            f"{value:.1%}",
            ha="center",
            fontsize=18,
            fontweight="bold",
            color="black"
        )

    title = (
        "Análise visual da predição com ResNeXt+LSTM\n"
        f"Acc={acc:.2%} | Precision={precision:.2%} | Recall={recall:.2%} | F1={f1:.2%}"
    )

    if auc is not None:
        title += f" | AUC={float(auc):.2%}"

    fig.suptitle(title, fontsize=28, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f"Figura salva em: {out_path}")
    print(f"Vídeo usado: {video_id}")
    print(f"Classe real: {true_name}")
    print(f"Predição: {pred_name}")
    print(f"Prob real: {prob_real:.4f}")
    print(f"Prob fake: {prob_fake:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Gera uma figura demo da predição ResNeXt+LSTM."
    )

    parser.add_argument(
        "--run-dir",
        default="experiments/resnext_lstm/videosResNeXtLSTMfase8/videosResNeXtLSTMfase8_2026-05-21_21-02-09_bs2_ep150_lr0.001_optSGD_seq10_drop0.7_wd1e-05",
        help="Pasta do experimento contendo config.json e best_model.pt.",
    )

    parser.add_argument(
        "--out",
        default="reports/resnext_lstm/figures/demo_resnext_lstm.png",
        help="Caminho de saída da imagem.",
    )

    parser.add_argument(
        "--mode",
        default="hardest",
        choices=["hardest", "wrong", "fake", "real", "first"],
        help="Tipo de exemplo usado na figura.",
    )

    parser.add_argument(
        "--video-id",
        default=None,
        help="Opcional: força o uso de um video_id específico.",
    )

    args = parser.parse_args()

    make_demo(
        run_dir=resolve_from_root(args.run_dir),
        out_path=resolve_from_root(args.out),
        mode=args.mode,
        chosen_video_id=args.video_id,
    )


if __name__ == "__main__":
    main()