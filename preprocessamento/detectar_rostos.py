from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def process_images(
    input_dir: Path,
    output_dir: Path,
    xml_path: Path,
    img_size: int,
    scale_factor: float,
    min_neighbors: int,
    min_size: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    face_cascade = cv2.CascadeClassifier(str(xml_path))
    if face_cascade.empty():
        raise RuntimeError(f'Não foi possível carregar o XML em: {xml_path}')

    image_paths = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}])
    saved = 0
    skipped = 0

    for idx, image_path in enumerate(image_paths, start=1):
        image = cv2.imread(str(image_path))
        if image is None:
            skipped += 1
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(min_size, min_size),
        )

        if len(faces) == 0:
            skipped += 1
            continue

        x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
        face_crop = image[y:y + h, x:x + w]
        face_crop = cv2.resize(face_crop, (img_size, img_size))
        cv2.imwrite(str(output_dir / image_path.name), face_crop)
        saved += 1

        if idx % 500 == 0:
            print(f'{idx}/{len(image_paths)} imagens processadas em {input_dir.name}')

    print(f'{input_dir} -> salvas: {saved} | sem rosto/erro: {skipped}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Detecta rostos nos frames e salva dataset processado.')
    parser.add_argument('--input-dir', default='data/interim/frames_interval_10')
    parser.add_argument('--output-dir', default='data/processed/faces_256_interval_10')#nome por conta do tamanho da imagem 
    parser.add_argument('--xml-path', default=str(Path(__file__).resolve().parent / 'haarcascade_frontalface_default.xml'))
    parser.add_argument('--img-size', type=int, default=256)#faces redimensionados para 256x256 pixels
    parser.add_argument('--scale-factor', type=float, default=1.1)
    parser.add_argument('--min-neighbors', type=int, default=5)
    parser.add_argument('--min-size', type=int, default=60)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    xml_path = Path(args.xml_path)

    process_images(
        input_dir / 'real',
        output_dir / 'real',
        xml_path,
        args.img_size,
        args.scale_factor,
        args.min_neighbors,
        args.min_size,
    )
    process_images(
        input_dir / 'fake',
        output_dir / 'fake',
        xml_path,
        args.img_size,
        args.scale_factor,
        args.min_neighbors,
        args.min_size,
    )


if __name__ == '__main__':
    main()
