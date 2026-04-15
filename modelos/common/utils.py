from __future__ import annotations

import csv
import json
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(path: str | Path, data: dict[str, Any]) -> None:
    ensure_dir(Path(path).parent)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def timestamp_now() -> str:
    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


def pick_device(device_name: str) -> torch.device:
    if device_name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_name)


def append_row_to_csv(csv_path: str | Path, row: dict[str, Any]) -> None:
    csv_path = Path(csv_path)
    ensure_dir(csv_path.parent)
    file_exists = csv_path.exists()
    with open(csv_path, 'a', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def copy_if_best(source_path: str | Path, target_path: str | Path) -> None:
    source_path = Path(source_path)
    target_path = Path(target_path)
    ensure_dir(target_path.parent)
    shutil.copy2(source_path, target_path)
