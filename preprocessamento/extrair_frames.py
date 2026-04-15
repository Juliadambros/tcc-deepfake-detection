from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def extract_frames_from_folder(
    input_dir: Path,
    output_dir: Path,
    max_videos: int,
    frame_interval: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    videos = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in {'.mp4', '.avi', '.mov', '.mkv'}])
    videos = videos[:max_videos] if max_videos > 0 else videos

    print(f'Pasta: {input_dir}')
    print(f'Total de vídeos selecionados: {len(videos)}')

    total_saved = 0
    for idx, video_path in enumerate(videos, start=1):
        cap = cv2.VideoCapture(str(video_path))
        frame_count = 0
        saved_from_video = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                frame_name = f'{video_path.stem}_frame{saved_from_video:05d}.jpg'
                cv2.imwrite(str(output_dir / frame_name), frame)
                saved_from_video += 1
                total_saved += 1

            frame_count += 1

        cap.release()
        print(f'[{idx}/{len(videos)}] {video_path.name} -> {saved_from_video} frames')

    print(f'Total de frames salvos em {output_dir}: {total_saved}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Extrai frames dos vídeos reais e falsos.')
    parser.add_argument('--raw-dir', default='data/raw/FaceForensics++_C23')
    parser.add_argument('--real-subdir', default='original')
    parser.add_argument('--fake-subdir', default='deepfakes')
    parser.add_argument('--output-dir', default='data/interim/frames_interval_10')
    parser.add_argument('--max-videos', type=int, default=200) #200 primeiros 
    parser.add_argument('--frame-interval', type=int, default=10) #1 a cada 15 frames
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)

    extract_frames_from_folder(
        raw_dir / args.real_subdir,
        output_dir / 'real',
        args.max_videos,
        args.frame_interval,
    )
    extract_frames_from_folder(
        raw_dir / args.fake_subdir,
        output_dir / 'fake',
        args.max_videos,
        args.frame_interval,
    )


if __name__ == '__main__':
    main()
