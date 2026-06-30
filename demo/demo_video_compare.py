import sys
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

from modelos.mesonet.model import Meso4
from modelos.cnn_baseline.model import CNNBaseline

DEVICE = "cuda"

VIDEO_PATH = "data/raw/FaceForensics++_C23/deepfakes/000_003.mp4"

MESO_PATH = "checkpoints/mesonet_images/best_treinamento_imagens_4_fase1_1.pt"
CNN_PATH = "checkpoints/cnn/best_imagensCNNfase4.pt"

CASCADE_PATH = "preprocessamento/haarcascade_frontalface_default.xml"

OUTPUT_DIR = Path("reports/videos/demos")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VIDEO_NAME = Path(VIDEO_PATH).stem

def load_models():
    meso = Meso4(num_classes=2, dropout_conv=0.25, dropout_fc=0.3).to(DEVICE)
    meso.load_state_dict(torch.load(MESO_PATH, map_location=DEVICE))
    meso.eval()

    cnn = CNNBaseline(num_classes=2, dropout_fc=0.7).to(DEVICE)
    cnn.load_state_dict(torch.load(CNN_PATH, map_location=DEVICE))
    cnn.eval()

    return meso, cnn

def get_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])


def process_video(meso, cnn, transform):

    face_detector = cv2.CascadeClassifier(CASCADE_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)

    frames = []
    meso_scores = []
    cnn_scores = []

    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % 10 == 0:

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

            if len(faces) > 0:
                x, y, w, h = faces[0]

                if w < 80 or h < 80:
                    frame_id += 1
                    continue

                face = frame[y:y+h, x:x+w]
                rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

                tensor = transform(rgb).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    out_meso = meso(tensor)
                    out_cnn = cnn(tensor)

                    probs_meso = torch.softmax(out_meso, dim=1)[0]
                    probs_cnn = torch.softmax(out_cnn, dim=1)[0]

                    prob_fake_meso = probs_meso[0].item()
                    prob_fake_cnn = probs_cnn[0].item()

                frames.append(rgb)
                meso_scores.append(prob_fake_meso)
                cnn_scores.append(prob_fake_cnn)

        frame_id += 1

    cap.release()

    return frames, meso_scores, cnn_scores

def plot_frames(frames, meso_scores, cnn_scores):

    meso_mean = np.mean(meso_scores)
    cnn_mean = np.mean(cnn_scores)

    meso_label = "FAKE" if meso_mean > 0.5 else "REAL"
    cnn_label = "FAKE" if cnn_mean > 0.5 else "REAL"

    N = min(6, len(frames))

    plt.figure(figsize=(15, 6))

    for i in range(N):
        plt.subplot(2, N, i + 1)
        plt.imshow(frames[i])
        plt.title(f"Meso: {meso_scores[i]:.2f}")
        plt.axis("off")

        plt.subplot(2, N, i + 1 + N)
        plt.imshow(frames[i])
        plt.title(f"CNN: {cnn_scores[i]:.2f}")
        plt.axis("off")

    plt.suptitle(
        f"MesoNet: {meso_label} ({meso_mean:.2f}) | "
        f"CNN: {cnn_label} ({cnn_mean:.2f})",
        fontsize=14
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{VIDEO_NAME}_frames.png", dpi=200)
    plt.close()

def plot_temporal(meso_scores, cnn_scores):

    plt.figure(figsize=(8, 3.5))

    plt.plot(
        meso_scores,
        marker='o',
        linewidth=2.5,
        markersize=6,
        label="Meso-4"
    )

    plt.plot(
        cnn_scores,
        marker='s',
        linewidth=2.5,
        markersize=6,
        label="CNN Baseline"
    )

    plt.axhline(
        0.5,
        linestyle="--",
        linewidth=2,
        color="black",
        label="Limiar"
    )

    plt.ylim(0.45, 1.02)

    plt.xlabel(
        "Frame (amostrado)",
        fontsize=14,
        fontweight="bold"
    )

    plt.ylabel(
        "Probabilidade de FAKE",
        fontsize=14,
        fontweight="bold"
    )

    plt.title(
        "Evolução da probabilidade ao longo do vídeo",
        fontsize=16,
        fontweight="bold"
    )

    plt.xticks(
        fontsize=12,
        fontweight="bold"
    )

    plt.yticks(
        fontsize=12,
        fontweight="bold"
    )

    plt.legend(
        fontsize=12,
        frameon=True
    )

    plt.grid(
        True,
        linestyle="--",
        linewidth=1.2,
        alpha=0.8,
        color="gray"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"{VIDEO_NAME}_temporal.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()
def main():

    print("Carregando modelos...")
    meso, cnn = load_models()

    transform = get_transform()

    print("Processando vídeo...")
    frames, meso_scores, cnn_scores = process_video(meso, cnn, transform)

    print("Gerando gráficos...")
    plot_frames(frames, meso_scores, cnn_scores)
    plot_temporal(meso_scores, cnn_scores)

    print("\n Demo salvo em:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()