from __future__ import annotations

from pathlib import Path
from typing import Tuple

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


def build_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])


def build_imagefolder_loaders(
    split_dir: str | Path,
    img_size: int,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    split_dir = Path(split_dir)
    transform = build_transform(img_size)

    train_dataset = datasets.ImageFolder(split_dir / 'train', transform=transform)
    val_dataset = datasets.ImageFolder(split_dir / 'val', transform=transform)
    test_dataset = datasets.ImageFolder(split_dir / 'test', transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader, train_dataset.classes
