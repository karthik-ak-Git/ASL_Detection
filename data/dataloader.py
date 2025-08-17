"""Dataloaders and transforms for ASL dataset.

- Auto-detects train/test folders among:
  - dataset/train, dataset/test
  - dataset/asl_alphabet_train, dataset/asl_alphabet_test
- Uses 224px ImageNet-normalized transforms for ResNet18.
"""

from torchvision import transforms
import os
from typing import Tuple, Dict

import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

# Defaults
BATCH_SIZE = 32
VALID_SPLIT = 0.2


def _detect_dataset_roots(base_dir: str = "dataset") -> Tuple[str, str]:
    """Return (train_dir, test_dir) based on existing folders."""
    candidates = [
        (os.path.join(base_dir, "train"), os.path.join(base_dir, "test")),
        (os.path.join(base_dir, "asl_alphabet_train"),
         os.path.join(base_dir, "asl_alphabet_test")),
    ]
    for train_dir, test_dir in candidates:
        if os.path.isdir(train_dir):
            # test_dir may be missing; caller can handle
            return train_dir, test_dir
    raise FileNotFoundError(
        "Could not find dataset. Expected one of: 'dataset/train' or 'dataset/asl_alphabet_train'"
    )


def build_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


def _build_transforms():
    # For backward compatibility
    return build_transforms(train=True), build_transforms(train=False)


def _class_root(path: str) -> str:
    """Return the folder that directly contains class subfolders.

    If the given path has exactly one subdirectory, and that subdirectory
    contains multiple subdirectories, drill down one level. This handles
    archives that extract to a wrapping folder like 'asl_alphabet_train/asl_alphabet_train/'.
    """
    if not os.path.isdir(path):
        return path
    subs = [d for d in os.listdir(
        path) if os.path.isdir(os.path.join(path, d))]
    if len(subs) == 1:
        inner = os.path.join(path, subs[0])
        inner_subs = [d for d in os.listdir(
            inner) if os.path.isdir(os.path.join(inner, d))]
        if len(inner_subs) > 1:
            return inner
    return path


def get_dataloaders(batch_size: int = BATCH_SIZE, valid_split: float = VALID_SPLIT):
    train_dir, _ = _detect_dataset_roots()
    train_dir = _class_root(train_dir)
    train_transform, test_transform = _build_transforms()

    full_dataset = datasets.ImageFolder(
        root=train_dir, transform=train_transform)
    class_to_idx = full_dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # Split train/val
    val_size = int(valid_split * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size])

    # Use test/eval transform for val set to avoid leaking train augmentations
    val_dataset.dataset.transform = test_transform

    # Dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, class_to_idx, idx_to_class


def get_test_loader(batch_size: int = BATCH_SIZE, num_workers: int | None = None):
    _, test_dir = _detect_dataset_roots()
    test_dir = _class_root(test_dir)
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    _, test_transform = _build_transforms()
    test_dataset = datasets.ImageFolder(
        root=test_dir, transform=test_transform)
    if num_workers is None:
        num_workers = 0 if os.name == 'nt' else 2
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers, pin_memory=True)
    return test_loader


def get_class_mapping() -> Tuple[Dict[str, int], Dict[int, str]]:
    """Return (class_to_idx, idx_to_class) based on detected training root."""
    train_dir, _ = _detect_dataset_roots()
    train_dir = _class_root(train_dir)
    ds = datasets.ImageFolder(root=train_dir)
    class_to_idx = ds.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    return class_to_idx, idx_to_class
