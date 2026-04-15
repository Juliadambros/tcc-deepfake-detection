from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DIR = DATA_DIR / 'raw'
INTERIM_DIR = DATA_DIR / 'interim'
PROCESSED_DIR = DATA_DIR / 'processed'
SPLITS_DIR = DATA_DIR / 'splits'
EXPERIMENTS_DIR = PROJECT_ROOT / 'experiments'
REPORTS_DIR = PROJECT_ROOT / 'reports'
CHECKPOINTS_DIR = PROJECT_ROOT / 'checkpoints'
HAAR_XML_PATH = PROJECT_ROOT / 'haarcascade_frontalface_default.xml'


def resolve_from_root(relative_path: str | Path) -> Path:
    path = Path(relative_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path
