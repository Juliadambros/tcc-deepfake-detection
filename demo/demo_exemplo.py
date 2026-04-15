from __future__ import annotations

import random
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modelos.mesonet.model import Meso4

# ===== CAMINHOS =====
XML_PATH = PROJECT_ROOT / "preprocessamento" / "haarcascade_frontalface_default.xml"
MODEL_PATH = PROJECT_ROOT / "checkpoints" / "best_treinamento_imagens_4_fase1_1.pt"

FRAMES_REAL_DIR = PROJECT_ROOT / "data" / "interim" / "frames_interval_10" / "real"
FRAMES_FAKE_DIR = PROJECT_ROOT / "data" / "interim" / "frames_interval_10" / "fake"

OUT_PATH = PROJECT_ROOT / "reports" / "mesonet_image" / "figures" / "demo" / "demo_real_fake.png"

IMG_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PURPLE = "#6A0DAD"
PURPLE_DARK = "#4B0082"
PURPLE_BRIGHT_BGR = (180, 0, 180) 


def list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]


def softmax(x: np.ndarray) -> np.ndarray:
    ex = np.exp(x - np.max(x))
    return ex / np.sum(ex)


def resolve_frames_dirs() -> tuple[Path, Path]:
    if FRAMES_REAL_DIR.exists() and FRAMES_FAKE_DIR.exists():
        return FRAMES_REAL_DIR, FRAMES_FAKE_DIR

    fallback_real = PROJECT_ROOT / "data" / "interim" / "frames" / "real"
    fallback_fake = PROJECT_ROOT / "data" / "interim" / "frames" / "fake"
    return fallback_real, fallback_fake


def load_model(model_path: Path) -> Meso4:
    if not model_path.exists():
        raise FileNotFoundError(f"Não achei o modelo em: {model_path}")

    model = Meso4(dropout_conv=0.25, dropout_fc=0.3).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model


def pick_image_with_face(images: list[Path], face_cascade: cv2.CascadeClassifier, max_attempts: int = 30):
    attempts = 0
    while attempts < max_attempts:
        img_path = random.choice(images)
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            attempts += 1
            continue

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
        )

        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
            return img_path, img_bgr, (x, y, w, h)

        attempts += 1

    raise RuntimeError("Não foi possível encontrar uma imagem com rosto detectado.")


def predict_face(model: Meso4, face_bgr: np.ndarray) -> tuple[str, np.ndarray, np.ndarray]:
    face_resized = cv2.resize(face_bgr, (IMG_SIZE, IMG_SIZE))
    face_input = torch.from_numpy(face_resized[:, :, ::-1].copy()).float() / 255.0
    face_input = face_input.permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(face_input).cpu().numpy()[0]
        probs = softmax(logits)

    pred_idx = int(np.argmax(probs))
    pred_class = ["fake", "real"][pred_idx]
    return pred_class, probs, face_resized


def prepare_visual(img_bgr: np.ndarray, face_box: tuple[int, int, int, int]):
    x, y, w, h = face_box
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    img_draw_bgr = img_bgr.copy()
    cv2.rectangle(img_draw_bgr, (x, y), (x + w, y + h), PURPLE_BRIGHT_BGR, 3)
    img_draw_rgb = cv2.cvtColor(img_draw_bgr, cv2.COLOR_BGR2RGB)

    face_crop = img_bgr[y:y + h, x:x + w]
    return img_rgb, img_draw_rgb, face_crop


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    face_cascade = cv2.CascadeClassifier(str(XML_PATH))
    if face_cascade.empty():
        raise RuntimeError(f"Não consegui carregar o XML: {XML_PATH}")

    model = load_model(MODEL_PATH)

    real_dir, fake_dir = resolve_frames_dirs()
    real_imgs = list_images(real_dir)
    fake_imgs = list_images(fake_dir)

    if not real_imgs or not fake_imgs:
        raise RuntimeError(
            f"Não encontrei imagens em:\n- {real_dir}\n- {fake_dir}"
        )

    # escolhe uma imagem real com rosto
    real_path, real_bgr, real_face_box = pick_image_with_face(real_imgs, face_cascade)
    real_rgb, real_draw_rgb, real_face_crop = prepare_visual(real_bgr, real_face_box)
    real_pred, real_probs, real_face_resized = predict_face(model, real_face_crop)

    # escolhe uma imagem fake com rosto
    fake_path, fake_bgr, fake_face_box = pick_image_with_face(fake_imgs, face_cascade)
    fake_rgb, fake_draw_rgb, fake_face_crop = prepare_visual(fake_bgr, fake_face_box)
    fake_pred, fake_probs, fake_face_resized = predict_face(model, fake_face_crop)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))


    axes[0, 0].imshow(real_rgb)
    axes[0, 0].set_title("Imagem real - original", color=PURPLE, fontsize=12, fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(real_draw_rgb)
    axes[0, 1].set_title("Imagem real - rosto detectado", color=PURPLE, fontsize=12, fontweight="bold")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(cv2.cvtColor(real_face_resized, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title(
        f"Predição: {real_pred}\nFake={real_probs[0]:.3f} | Real={real_probs[1]:.3f}",
        color=PURPLE_DARK,
        fontsize=11,
        fontweight="bold",
    )
    axes[0, 2].axis("off")


    axes[1, 0].imshow(fake_rgb)
    axes[1, 0].set_title("Imagem fake - original", color=PURPLE, fontsize=12, fontweight="bold")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(fake_draw_rgb)
    axes[1, 1].set_title("Imagem fake - rosto detectado", color=PURPLE, fontsize=12, fontweight="bold")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(cv2.cvtColor(fake_face_resized, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title(
        f"Predição: {fake_pred}\nFake={fake_probs[0]:.3f} | Real={fake_probs[1]:.3f}",
        color=PURPLE_DARK,
        fontsize=11,
        fontweight="bold",
    )
    axes[1, 2].axis("off")

    fig.suptitle(
        "Exemplos visuais de predição com o melhor modelo MesoNet",
        color=PURPLE,
        fontsize=16,
        fontweight="bold",
    )

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=220, bbox_inches="tight")
    plt.show()

    print(f"Figura salva em: {OUT_PATH}")
    print(f"Imagem real usada: {real_path}")
    print(f"Imagem fake usada: {fake_path}")
    print(f"Modelo usado: {MODEL_PATH}")


if __name__ == "__main__":
    main()