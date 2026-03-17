import os
import warnings

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
from tqdm import tqdm

from .patches import extract_3d_patches_minimal_overlap

warnings.filterwarnings("ignore")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

TRAIN_DATA_DIR = "./numpy-data-types-point-C"
TEST_DATA_DIR = "./data"

DEFAULT_TRAIN_NAMES = ["TS_5_4", "TS_69_2", "TS_6_6", "TS_73_6", "TS_86_3", "TS_99_9"]
DEFAULT_VALID_NAMES = ["TS_6_4"]
DEFAULT_TOMO_TYPES = ["ctfdeconvolved", "denoised", "isonetcorrected", "wbp"]
DEFAULT_TEST_TOMO_TYPES = ["denoised"]
DEFAULT_VAL_PATCH_SIZES = [128, 384, 384]


def load_npy_files(
    data_dir=TRAIN_DATA_DIR,
    train_names=None,
    valid_names=None,
    tomo_type_list=None,
    test_tomo_type_list=None,
    data_type=None,
):
    """Load .npy image/label pairs for training and validation."""
    if train_names is None:
        train_names = DEFAULT_TRAIN_NAMES
    if valid_names is None:
        valid_names = DEFAULT_VALID_NAMES
    if tomo_type_list is None:
        tomo_type_list = DEFAULT_TOMO_TYPES
    if test_tomo_type_list is None:
        test_tomo_type_list = DEFAULT_TEST_TOMO_TYPES
    if data_type is None:
        data_type = [""]

    train_files = []
    valid_files = []

    for tomo_type in tomo_type_list:
        for data_t in data_type:
            for name in tqdm(train_names, desc=f"Loading train ({tomo_type})"):
                image = np.load(
                    f"{data_dir}/train_image_{name}_{tomo_type}{data_t}.npy"
                )
                label = np.load(
                    f"{data_dir}/train_label_{name}_{tomo_type}{data_t}.npy"
                )
                label[label == 2] = 0
                label[label > 2] -= 1
                train_files.append({"image": image, "label": label})

    for tomo_type in test_tomo_type_list:
        for data_t in data_type:
            for name in tqdm(valid_names, desc=f"Loading valid ({tomo_type})"):
                image = np.load(
                    f"{data_dir}/train_image_{name}_{tomo_type}{data_t}.npy"
                )
                label = np.load(
                    f"{data_dir}/train_label_{name}_{tomo_type}{data_t}.npy"
                )
                label[label == 2] = 0
                label[label > 2] -= 1
                valid_files.append({"image": image, "label": label})

    return train_files, valid_files


def build_transforms(val_patch_sizes=None, num_samples=1):
    """Build non-random (cache), validation, and random (training) transforms."""
    if val_patch_sizes is None:
        val_patch_sizes = DEFAULT_VAL_PATCH_SIZES

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
                keys=["image", "label"], prob=0.5, max_k=3, spatial_axes=[1, 2]
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


def build_dataloaders(
    train_files,
    valid_files,
    val_patch_sizes=None,
    num_samples=1,
    train_batch_size=1,
    valid_batch_size=1,
    num_workers=16,
):
    """Build train and validation DataLoaders with caching and patch extraction."""
    if val_patch_sizes is None:
        val_patch_sizes = DEFAULT_VAL_PATCH_SIZES

    non_random_transforms, val_transforms, random_transforms = build_transforms(
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
