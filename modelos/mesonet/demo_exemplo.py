import os
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

XML_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
MODEL_PATH = os.path.join(BASE_DIR, "resultados", "mesonet", "best_model.pt")

FRAMES_REAL_DIR = os.path.join(BASE_DIR, "data", "dataset", "real")
FRAMES_FAKE_DIR = os.path.join(BASE_DIR, "data", "dataset", "fake")

OUT_PATH = os.path.join(BASE_DIR, "resultados", "mesonet", "exemplo_pipeline.png")

IMG_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def list_images(folder):
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

def softmax(x):
    ex = np.exp(x - np.max(x))
    return ex / np.sum(ex)

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
            nn.Dropout(0.0),  # avaliação
            nn.Linear(16, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

def main():
    # 1)Detector de rosto 
    face_cascade = cv2.CascadeClassifier(XML_PATH)
    if face_cascade.empty():
        raise RuntimeError(f"Não consegui carregar o XML: {XML_PATH}")

    # 2)Carregar MesoNet treinada
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Não achei o modelo em: {MODEL_PATH}")

    model = Meso4().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 3) Escolher 1 frame aleatório (real/fake)
    real_imgs = list_images(FRAMES_REAL_DIR)
    fake_imgs = list_images(FRAMES_FAKE_DIR)
    if not real_imgs or not fake_imgs:
        raise RuntimeError("Não encontrei imagens em data/dataset/real ou data/dataset/fake")

    true_class = random.choice(["real", "fake"])
    img_path = random.choice(real_imgs if true_class == "real" else fake_imgs)

    # 4)Ler e detectar rosto
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise RuntimeError(f"Não consegui ler a imagem: {img_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    # tenta outras imagens se essa não tiver rosto detectável
    attempts = 0
    while len(faces) == 0 and attempts < 20:
        true_class = random.choice(["real", "fake"])
        img_path = random.choice(real_imgs if true_class == "real" else fake_imgs)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            attempts += 1
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        attempts += 1

    if len(faces) == 0:
        raise RuntimeError("Não consegui detectar rosto em 20 tentativas. Ajuste minSize/params ou escolha outra imagem.")

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    boxed = img_rgb.copy()
    cv2.rectangle(boxed, (x, y), (x + w, y + h), (0, 255, 0), 3)

    face_rgb = img_rgb[y:y + h, x:x + w]
    face_resized = cv2.resize(face_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    # 5) Inferência (probabilidade)
    tensor = torch.from_numpy(face_resized).float() / 255.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor).cpu().numpy()[0]
    probs = softmax(logits)

    #índice 0=fake, 1=real 
    pred_idx = int(np.argmax(probs))
    pred_label = "fake" if pred_idx == 0 else "real"
    conf = float(probs[pred_idx]) * 100

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    plt.figure(figsize=(14, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    plt.title("Frame original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(boxed)
    plt.title("Detecção (Haar Cascade)")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(face_resized)
    plt.title(f"Rosto 256x256\nPred: {pred_label} ({conf:.1f}%)\nTrue (pasta): {true_class}")
    plt.axis("off")

    plt.suptitle(f"Arquivo: {os.path.basename(img_path)}", y=1.05)
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=250, bbox_inches="tight")
    plt.close()

    print("Exemplo MesoNet salvo em:", OUT_PATH)
    print("Imagem escolhida:", img_path)
    print(f"True (pasta): {true_class} | Pred: {pred_label} | Confiança: {conf:.2f}%")

if __name__ == "__main__":
    main()