from pathlib import Path
import shutil
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from src.config import (
    DATA_DIR,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    SEED,
    SPLITS_DIR,
    TEST_RATIO,
    VAL_RATIO,
)

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
except ImportError:
    torch = None
    Dataset = object
    DataLoader = None
    transforms = None


def scan_image_dataset(
    data_dir: Path = DATA_DIR,
    selected_classes: List[str] = None,
    max_per_class: int = 0,
) -> Tuple[List[Path], List[int], List[str]]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    all_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])
    if selected_classes:
        selected_set = {c.lower() for c in selected_classes}
        class_dirs = [d for d in all_dirs if d.name.lower() in selected_set]
    else:
        class_dirs = all_dirs

    class_names = [d.name for d in class_dirs]
    image_paths: List[Path] = []
    labels: List[int] = []

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for label_idx, cls_dir in enumerate(class_dirs):
        files = [f for f in sorted(cls_dir.iterdir()) if f.suffix.lower() in valid_exts]
        if max_per_class > 0:
            files = files[:max_per_class]
        for img_file in files:
            image_paths.append(img_file)
            labels.append(label_idx)

    return image_paths, labels, class_names


def stratified_holdout_split(
    image_paths: List[Path],
    labels: List[int],
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    seed: int = SEED,
) -> Dict[str, Tuple[List[Path], List[int]]]:
    paths_arr = np.array(image_paths)
    labels_arr = np.array(labels)

    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        paths_arr,
        labels_arr,
        test_size=test_ratio,
        stratify=labels_arr,
        random_state=seed,
    )

    relative_val_ratio = val_ratio / (1.0 - test_ratio)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths,
        train_val_labels,
        test_size=relative_val_ratio,
        stratify=train_val_labels,
        random_state=seed,
    )

    return {
        "train": (train_paths.tolist(), train_labels.tolist()),
        "val": (val_paths.tolist(), val_labels.tolist()),
        "test": (test_paths.tolist(), test_labels.tolist()),
    }


def materialize_splits_to_disk(
    splits: Dict[str, Tuple[List[Path], List[int]]],
    class_names: List[str],
    splits_dir: Path = SPLITS_DIR,
) -> Dict[str, Tuple[List[Path], List[int]]]:
    if splits_dir.exists():
        shutil.rmtree(splits_dir)

    new_splits: Dict[str, Tuple[List[Path], List[int]]] = {}
    for split_name, (paths, labels) in splits.items():
        new_paths: List[Path] = []
        for src_path, label_idx in zip(paths, labels):
            cls_name = class_names[label_idx]
            dest_dir = splits_dir / split_name / cls_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_file = dest_dir / src_path.name
            shutil.copy2(src_path, dest_file)
            new_paths.append(dest_file)
        new_splits[split_name] = (new_paths, labels)

    return new_splits


def load_existing_splits_from_disk(
    splits_dir: Path = SPLITS_DIR,
) -> Tuple[Dict[str, Tuple[List[Path], List[int]]], List[str]]:
    train_dir = splits_dir / "train"
    if not train_dir.exists():
        raise FileNotFoundError(f"Directory not found: {train_dir}")

    class_names = [d.name for d in sorted([d for d in train_dir.iterdir() if d.is_dir()])]
    splits: Dict[str, Tuple[List[Path], List[int]]] = {}
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for split_name in ["train", "val", "test"]:
        sp_dir = splits_dir / split_name
        paths: List[Path] = []
        labels: List[int] = []
        for label_idx, cls_name in enumerate(class_names):
            cls_folder = sp_dir / cls_name
            if not cls_folder.exists():
                continue
            for img_file in sorted(cls_folder.iterdir()):
                if img_file.suffix.lower() in valid_exts:
                    paths.append(img_file)
                    labels.append(label_idx)
        splits[split_name] = (paths, labels)

    return splits, class_names


def extract_color_texture_features(image_path: Path, bins: int = 8) -> np.ndarray:
    with Image.open(image_path) as img:
        img_rgb = img.convert("RGB").resize((128, 128))
        img_hsv = img_rgb.convert("HSV")

    rgb_arr = np.asarray(img_rgb, dtype=np.float32) / 255.0
    hsv_arr = np.asarray(img_hsv, dtype=np.float32) / 255.0

    features = []
    for channel in range(3):
        hist, _ = np.histogram(hsv_arr[:, :, channel], bins=bins, range=(0.0, 1.0), density=True)
        features.append(hist)

    rgb_mean = rgb_arr.mean(axis=(0, 1))
    rgb_std = rgb_arr.std(axis=(0, 1))
    hsv_mean = hsv_arr.mean(axis=(0, 1))
    hsv_std = hsv_arr.std(axis=(0, 1))

    features.extend([rgb_mean, rgb_std, hsv_mean, hsv_std])
    return np.concatenate(features).astype(np.float32)


def build_tabular_features(paths: List[Path]) -> np.ndarray:
    return np.vstack([extract_color_texture_features(p) for p in paths])


class PlantLeafDataset(Dataset):
    def __init__(self, image_paths: List[Path], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        with Image.open(img_path) as img:
            image = img.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label, str(img_path)


def get_transforms(image_size: int = IMAGE_SIZE, use_augmentation: bool = True):
    if transforms is None:
        raise ImportError("torchvision is required.")

    if use_augmentation:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_transform, eval_transform
