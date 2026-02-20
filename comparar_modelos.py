import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

IMAGE_PATH = "data/dataset/fake/000_003_20.jpg"
TRUE_CLASS = "fake" 

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

XML_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")

CNN_MODEL_PATH = os.path.join(BASE_DIR, "resultados", "cnn_baseline", "best_model.pt")
MESO_MODEL_PATH = os.path.join(BASE_DIR, "resultados", "mesonet", "best_model.pt")

OUT_PATH = os.path.join(BASE_DIR, "resultados", "comparacao_cnn_vs_mesonet.png")

IMG_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def softmax(x):
    ex = np.exp(x - np.max(x))
    return ex / np.sum(ex)


#cnn_baseline
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 128),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        return self.fc_layers(x)


#MesoNet 
class Meso4(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            nn.Conv2d(8, 8, kernel_size=5, padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            nn.Conv2d(8, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            nn.Conv2d(16, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 16 * 16, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(0.0),
            nn.Linear(16, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def detect_and_crop_face(img_bgr, cascade):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )
    if len(faces) == 0:
        return None, None, None

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    boxed = img_rgb.copy()
    cv2.rectangle(boxed, (x, y), (x + w, y + h), (0, 255, 0), 3)

    face_rgb = img_rgb[y:y + h, x:x + w]
    face_resized = cv2.resize(face_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return img_rgb, boxed, face_resized


def predict(model, face_resized_rgb):
    tensor = torch.from_numpy(face_resized_rgb).float() / 255.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor).cpu().numpy()[0]
    probs = softmax(logits)
    pred_idx = int(np.argmax(probs))
    pred_label = "fake" if pred_idx == 0 else "real"
    conf = float(probs[pred_idx]) * 100
    return pred_label, conf


def main():
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"Imagem não encontrada: {IMAGE_PATH}")

    face_cascade = cv2.CascadeClassifier(XML_PATH)
    if face_cascade.empty():
        raise RuntimeError(f"Não consegui carregar o XML: {XML_PATH}")

    img_bgr = cv2.imread(IMAGE_PATH)
    if img_bgr is None:
        raise RuntimeError(f"Não consegui ler a imagem: {IMAGE_PATH}")

    original_rgb, boxed_rgb, face_resized = detect_and_crop_face(img_bgr, face_cascade)
    if face_resized is None:
        raise RuntimeError("Não detectei rosto nessa imagem. Escolha outra ou ajuste parâmetros.")

    # carregar modelos
    cnn = SimpleCNN().to(DEVICE)
    cnn.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=DEVICE))
    cnn.eval()

    meso = Meso4().to(DEVICE)
    meso.load_state_dict(torch.load(MESO_MODEL_PATH, map_location=DEVICE))
    meso.eval()

    cnn_pred, cnn_conf = predict(cnn, face_resized)
    meso_pred, meso_conf = predict(meso, face_resized)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    plt.figure(figsize=(16, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(original_rgb)
    plt.title("Frame original")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(boxed_rgb)
    plt.title("Detecção (Haar)")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(face_resized)
    plt.title(f"Entrada 256x256\nTrue: {TRUE_CLASS}")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(face_resized)
    plt.title(
        f"Predições\nCNN: {cnn_pred} ({cnn_conf:.1f}%)\n"
        f"MesoNet: {meso_pred} ({meso_conf:.1f}%)"
    )
    plt.axis("off")

    plt.suptitle(os.path.basename(IMAGE_PATH), y=1.05)
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=250, bbox_inches="tight")
    plt.close()

    print("Comparação salva em:", OUT_PATH)
    print(f"CNN -> {cnn_pred} ({cnn_conf:.2f}%) | MesoNet -> {meso_pred} ({meso_conf:.2f}%)")


if __name__ == "__main__":
    main()