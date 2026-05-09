from __future__ import annotations

import argparse
import sys
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modelos.common.video_eval import evaluate_videos_from_faces
from modelos.mesonet.model import Meso4


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Avalia vídeos com Meso-4 por agregação de frames."
    )

    parser.add_argument(
        "--model-path",
        default="checkpoints/mesonet_images/best_treinamento_imagens_4_fase1_1.pt",
    )

    parser.add_argument(
        "--faces-dir",
        default="data/processed/faces_256_interval_10",
    )

    parser.add_argument(
        "--output-dir",
        default="reports/videos/mesonet",
    )

    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dropout-conv", type=float, default=0.25)
    parser.add_argument("--dropout-fc", type=float, default=0.3)

    args = parser.parse_args()

    def build_model():
        return Meso4(
            num_classes=2,
            dropout_conv=args.dropout_conv,
            dropout_fc=args.dropout_fc,
        )

    evaluate_videos_from_faces(
        model_name="mesonet",
        model_builder=build_model,
        model_path=args.model_path,
        faces_dir=args.faces_dir,
        output_dir=args.output_dir,
        img_size=args.img_size,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()