from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


def clear_and_make(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def split_class_images(
    source_dir: Path,
    split_dir: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> None:
    if round(train_ratio + val_ratio + test_ratio, 5) != 1.0:
        raise ValueError('A soma de train/val/test precisa ser 1.0')

    rng = random.Random(seed)
    class_names = sorted([d.name for d in source_dir.iterdir() if d.is_dir()])

    for split_name in ['train', 'val', 'test']:
        for class_name in class_names:
            clear_and_make(split_dir / split_name / class_name)

    for class_name in class_names:
        files = [p for p in (source_dir / class_name).iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
        files = sorted(files)
        rng.shuffle(files)

        n = len(files)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val

        splits = {
            'train': files[:n_train],
            'val': files[n_train:n_train + n_val], #ajustar hiperparâmetros
            'test': files[n_train + n_val:n_train + n_val + n_test],
        }

        print(f'Classe {class_name}: total={n} train={n_train} val={n_val} test={n_test}')
        for split_name, split_files in splits.items():
            for file_path in split_files:
                shutil.copy2(file_path, split_dir / split_name / class_name / file_path.name)


def main() -> None:
    parser = argparse.ArgumentParser(description='Cria splits fixos de treino/val/teste.')
    parser.add_argument('--source-dir', default='data/processed/faces_256_interval_10')
    parser.add_argument('--output-dir', default='data/splits/split_01_interval_10')
    parser.add_argument('--train-ratio', type=float, default=0.7) #70% - treino
    parser.add_argument('--val-ratio', type=float, default=0.15) #15% - validação
    parser.add_argument('--test-ratio', type=float, default=0.15) #15% - teste
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    split_class_images(
        source_dir=Path(args.source_dir),
        split_dir=Path(args.output_dir),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
