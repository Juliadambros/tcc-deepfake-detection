import cv2
import os

xml_path = "haarcascade_frontalface_default.xml"

entrada_real = "data/dataset/real"
entrada_fake = "data/dataset/fake"

saida_real = "data/dataset_faces/real"
saida_fake = "data/dataset_faces/fake"

tamanho = 256
face_cascade = cv2.CascadeClassifier(xml_path)

def processar_imagens(pasta_entrada, pasta_saida):
    os.makedirs(pasta_saida, exist_ok=True)

    imagens = [img for img in os.listdir(pasta_entrada) if img.endswith(".jpg")]
    total_salvas = 0

    for i, nome_img in enumerate(imagens):
        caminho_img = os.path.join(pasta_entrada, nome_img)
        img = cv2.imread(caminho_img)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        if len(faces) == 0:
            continue

        x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
        rosto = img[y:y+h, x:x+w]
        rosto = cv2.resize(rosto, (tamanho, tamanho))

        caminho_saida = os.path.join(pasta_saida, nome_img)
        cv2.imwrite(caminho_saida, rosto)

        total_salvas += 1
        if (i+1) % 500 == 0:
            print(f"{i+1}/{len(imagens)} imagens processadas")

    print(f"Total de rostos salvos: {total_salvas}")

print("=== PROCESSANDO REAL ===")
processar_imagens(entrada_real, saida_real)

print("\n=== PROCESSANDO FAKE ===")
processar_imagens(entrada_fake, saida_fake)

print("\nFINALIZADO")

