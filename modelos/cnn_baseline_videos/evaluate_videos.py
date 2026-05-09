from __future__ import annotations

import argparse
import sys
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modelos.common.video_eval import evaluate_videos_from_faces
from modelos.cnn_baseline.model import CNNBaseline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Avalia vídeos com CNN Baseline por agregação de frames."
    )

    parser.add_argument(
        "--model-path",
        default="checkpoints/cnn/best_imagensCNNfase4.pt",
    )

    parser.add_argument(
        "--faces-dir",
        default="data/processed/faces_256_interval_10",
    )

    parser.add_argument(
        "--output-dir",
        default="reports/videos/cnn",
    )

    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dropout-fc", type=float, default=0.7)

    args = parser.parse_args()

    def build_model():
        return CNNBaseline(
            num_classes=2,
            dropout_fc=args.dropout_fc,
        )

    evaluate_videos_from_faces(
        model_name="cnn",
        model_builder=build_model,
        model_path=args.model_path,
        faces_dir=args.faces_dir,
        output_dir=args.output_dir,
        img_size=args.img_size,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()