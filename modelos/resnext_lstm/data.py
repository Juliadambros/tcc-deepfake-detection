from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class VideoFramesDataset(Dataset):
    def __init__(
        self,
        root_dir,
        seq_len=30,
        img_size=256,
        split="train",
        train_ratio=0.7,
        val_ratio=0.15,
        seed=42,
    ):
        self.root_dir = Path(root_dir)
        self.seq_len = seq_len
        self.img_size = img_size
        self.split = split

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.samples = self._build_samples()
        self.samples = self._split_samples(train_ratio, val_ratio, seed)

        print(f"[{split}] Total de vídeos: {len(self.samples)}")

    def _get_video_id(self, filename):
        """
        Exemplos:
        000_003_frame00000.jpg -> 000_003
        000_frame00000.jpg     -> 000
        """
        return filename.split("_frame")[0]

    def _build_samples(self):
        samples = []

        for class_name, label in [("real", 0), ("fake", 1)]:
            class_dir = self.root_dir / class_name

            if not class_dir.exists():
                raise FileNotFoundError(f"Pasta não encontrada: {class_dir}")

            videos = defaultdict(list)

            for img_path in sorted(class_dir.glob("*.jpg")):
                video_id = self._get_video_id(img_path.name)
                videos[video_id].append(img_path)

            for video_id, frames in videos.items():
                frames = sorted(frames)

                if len(frames) >= self.seq_len:
                    samples.append({
                        "video_id": video_id,
                        "frames": frames,
                        "label": label,
                        "class_name": class_name,
                    })

        return samples

    def _split_samples(self, train_ratio, val_ratio, seed):
        generator = torch.Generator().manual_seed(seed)

        indices = torch.randperm(len(self.samples), generator=generator).tolist()

        n_total = len(indices)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        if self.split == "train":
            selected = indices[:n_train]
        elif self.split == "val":
            selected = indices[n_train:n_train + n_val]
        elif self.split == "test":
            selected = indices[n_train + n_val:]
        else:
            raise ValueError("split deve ser: train, val ou test")

        return [self.samples[i] for i in selected]

    def _sample_frames(self, frames):
        """
        Seleciona seq_len frames distribuídos ao longo do vídeo.
        """
        total = len(frames)

        if total == self.seq_len:
            return frames

        indices = torch.linspace(0, total - 1, steps=self.seq_len).long().tolist()
        return [frames[i] for i in indices]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        selected_frames = self._sample_frames(item["frames"])

        images = []
        for frame_path in selected_frames:
            img = Image.open(frame_path).convert("RGB")
            img = self.transform(img)
            images.append(img)

        frames_tensor = torch.stack(images)  # (seq_len, 3, H, W)
        label = torch.tensor(item["label"], dtype=torch.long)

        return frames_tensor, label, item["video_id"]