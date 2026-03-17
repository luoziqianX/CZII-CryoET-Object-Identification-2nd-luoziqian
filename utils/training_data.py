"""
Shared data loading, transforms, and dataloaders for segmentation training.
"""
import os
from typing import List, Optional

import numpy as np
import torch
from monai.data import CacheDataset, DataLoader, Dataset
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    Orientationd,
    RandAffined,
    RandCropByLabelClassesd,
    RandFlipd,
    RandGaussianNoised,
    RandRotate90d,
    RandShiftIntensityd,
    RandStdShiftIntensityd,
)

from utils.patch_utils import extract_3d_patches_minimal_overlap


# Default data configuration
TRAIN_DATA_DIR = "./numpy-data-types-point-C"
TEST_DATA_DIR = "./data"
TRAIN_NAMES = ["TS_5_4", "TS_69_2", "TS_6_6", "TS_73_6", "TS_86_3", "TS_99_9"]
VALID_NAMES = ["TS_6_4"]
TOMO_TYPE_LIST = ["ctfdeconvolved", "denoised", "isonetcorrected", "wbp"]
TEST_TOMO_TYPE_LIST = ["denoised"]
DATA_TYPE = [""]
VAL_PATCH_SIZES = [128, 384, 384]


def load_data_files(
    train_data_dir: str = TRAIN_DATA_DIR,
    train_names: Optional[List[str]] = None,
    valid_names: Optional[List[str]] = None,
    tomo_type_list: Optional[List[str]] = None,
    test_tomo_type_list: Optional[List[str]] = None,
    data_type: Optional[List[str]] = None,
    use_tqdm: bool = True,
) -> tuple:
    """Load train and validation data files from numpy arrays."""
    train_names = train_names or TRAIN_NAMES
    valid_names = valid_names or VALID_NAMES
    tomo_type_list = tomo_type_list or TOMO_TYPE_LIST
    test_tomo_type_list = test_tomo_type_list or TEST_TOMO_TYPE_LIST
    data_type = data_type or DATA_TYPE

    if use_tqdm:
        from tqdm import tqdm
        name_iter_train = tqdm(train_names)
        name_iter_valid = tqdm(valid_names)
    else:
        name_iter_train = train_names
        name_iter_valid = valid_names

    train_files = []
    valid_files = []

    for tomo_type in tomo_type_list:
        for data_t in data_type:
            for name in name_iter_train:
                image = np.load(
                    f"{train_data_dir}/train_image_{name}_{tomo_type}{data_t}.npy"
                )
                label = np.load(
                    f"{train_data_dir}/train_label_{name}_{tomo_type}{data_t}.npy"
                )
                label[label == 2] = 0
                label[label > 2] -= 1
                train_files.append({"image": image, "label": label})

    for tomo_type in test_tomo_type_list:
        for data_t in data_type:
            for name in name_iter_valid:
                image = np.load(
                    f"{train_data_dir}/train_image_{name}_{tomo_type}{data_t}.npy"
                )
                label = np.load(
                    f"{train_data_dir}/train_label_{name}_{tomo_type}{data_t}.npy"
                )
                label[label == 2] = 0
                label[label > 2] -= 1
                valid_files.append({"image": image, "label": label})

    return train_files, valid_files


def get_transforms(
    val_patch_sizes: List[int] = VAL_PATCH_SIZES,
    num_samples: int = 1,
) -> tuple:
    """Get non-random, validation, and random transforms."""
    non_random_transforms = Compose(
        [
            EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
        ]
    )

    val_transforms = Compose(
        [
            EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
            NormalizeIntensityd(keys="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
        ]
    )

    random_transforms = Compose(
        [
            RandCropByLabelClassesd(
                keys=["image", "label"],
                label_key="label",
                spatial_size=val_patch_sizes,
                num_classes=6,
                num_samples=num_samples,
            ),
            NormalizeIntensityd(keys="image"),
            RandFlipd(keys=["image", "label"], prob=0.3, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.3, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.3, spatial_axis=2),
            RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.50),
            RandStdShiftIntensityd(keys=["image"], prob=0.5, factors=0.1),
            RandGaussianNoised(keys=["image"], prob=0.3, mean=0.0, std=0.1),
            RandRotate90d(
                keys=["image", "label"],
                prob=0.5,
                max_k=3,
                spatial_axes=[1, 2],
            ),
            RandAffined(
                keys=["image", "label"],
                prob=0.3,
                rotate_range=(0.1, 0.1, 0.1),
                scale_range=(0.1, 0.1, 0.1),
            ),
        ]
    )

    return non_random_transforms, val_transforms, random_transforms


def create_dataloaders(
    train_files: List[dict],
    valid_files: List[dict],
    val_patch_sizes: List[int] = VAL_PATCH_SIZES,
    num_samples: int = 1,
    train_batch_size: int = 1,
    valid_batch_size: int = 1,
    num_workers: int = 16,
) -> tuple:
    """Create train and validation dataloaders."""
    non_random_transforms, val_transforms, random_transforms = get_transforms(
        val_patch_sizes=val_patch_sizes, num_samples=num_samples
    )

    raw_train_ds = CacheDataset(
        data=train_files, transform=non_random_transforms, cache_rate=1.0
    )
    train_ds = Dataset(data=raw_train_ds, transform=random_transforms)
    train_loader = DataLoader(
        train_ds,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_images = [d["image"] for d in valid_files]
    val_labels = [d["label"] for d in valid_files]
    val_image_patches, _ = extract_3d_patches_minimal_overlap(
        val_images, val_patch_sizes
    )
    val_label_patches, _ = extract_3d_patches_minimal_overlap(
        val_labels, val_patch_sizes
    )
    val_patched_data = [
        {"image": img, "label": lbl}
        for img, lbl in zip(val_image_patches, val_label_patches)
    ]

    valid_ds = CacheDataset(
        data=val_patched_data, transform=val_transforms, cache_rate=1.0
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=valid_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, valid_loader
