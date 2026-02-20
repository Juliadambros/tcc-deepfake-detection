import cv2
import os

pasta_base = "data/FaceForensics++_C23"
pasta_original = os.path.join(pasta_base, "original")
pasta_fake = os.path.join(pasta_base, "deepfakes")

saida_real = "data/dataset/real"
saida_fake = "data/dataset/fake"

max_videos = 200
frame_interval = 15

def extrair_frames(pasta_videos, pasta_saida, limite):
    os.makedirs(pasta_saida, exist_ok=True)

    videos = [v for v in os.listdir(pasta_videos) if v.endswith(".mp4")]
    videos = videos[:limite]

    print(f"Total de vídeos selecionados: {len(videos)}")
    total_frames_salvos = 0

    for i, video_nome in enumerate(videos):
        caminho_video = os.path.join(pasta_videos, video_nome)
        cap = cv2.VideoCapture(caminho_video)

        count = 0
        frame_id = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if count % frame_interval == 0:
                nome_frame = f"{video_nome[:-4]}_{frame_id}.jpg"
                caminho_saida = os.path.join(pasta_saida, nome_frame)
                cv2.imwrite(caminho_saida, frame)
                frame_id += 1
                total_frames_salvos += 1

            count += 1

        cap.release()
        print(f"[{i+1}/{len(videos)}] {video_nome} concluído")

    print(f"\nTotal de frames salvos: {total_frames_salvos}")

print("=== EXTRAINDO REAL ===")
extrair_frames(pasta_original, saida_real, max_videos)

print("\n=== EXTRAINDO FAKE ===")
extrair_frames(pasta_fake, saida_fake, max_videos)

print("\nFINALIZADO")


