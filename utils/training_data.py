from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np


DEFAULT_TRAIN_DATA_DIR = "./numpy-data-types-point-C"
DEFAULT_TEST_DATA_DIR = "./data"
DEFAULT_TRAIN_NAMES = [
    "TS_5_4",
    "TS_69_2",
    "TS_6_6",
    "TS_73_6",
    "TS_86_3",
    "TS_99_9",
]
DEFAULT_VALID_NAMES = ["TS_6_4"]
DEFAULT_TRAIN_TOMO_TYPES = ["ctfdeconvolved", "denoised", "isonetcorrected", "wbp"]
DEFAULT_TEST_TOMO_TYPES = ["denoised"]
DEFAULT_DATA_TYPES = [""]


def calculate_patch_starts(dimension_size: int, patch_size: int) -> List[int]:
    """Calculate patch start positions with minimal overlap."""
    if dimension_size <= patch_size:
        return [0]

    n_patches = np.ceil(dimension_size / patch_size)
    if n_patches == 1:
        return [0]

    total_overlap = (n_patches * patch_size - dimension_size) / (n_patches - 1)

    positions = []
    for i in range(int(n_patches)):
        pos = int(i * (patch_size - total_overlap))
        if pos + patch_size > dimension_size:
            pos = dimension_size - patch_size
        if pos not in positions:
            positions.append(pos)

    return positions


def extract_3d_patches_minimal_overlap(
    arrays: List[np.ndarray], patch_sizes: List[int]
) -> Tuple[List[np.ndarray], List[Tuple[int, int, int]]]:
    """Extract patches that tile a volume with minimal overlap."""
    patch_size_d, patch_size_h, patch_size_w = patch_sizes
    shape = arrays[0].shape
    depth, height, width = shape

    if not all(arr.shape == shape for arr in arrays):
        raise ValueError("All input arrays must have the same shape")
    if patch_size_d > depth or patch_size_h > height or patch_size_w > width:
        raise ValueError(
            f"patch_size ({patch_size_d, patch_size_h, patch_size_w}) must be smaller than shape {shape}"
        )

    x_starts = calculate_patch_starts(depth, patch_size_d)
    y_starts = calculate_patch_starts(height, patch_size_h)
    z_starts = calculate_patch_starts(width, patch_size_w)

    patches = []
    coordinates = []
    for arr in arrays:
        for x in x_starts:
            for y in y_starts:
                for z in z_starts:
                    patch = arr[
                        x : x + patch_size_d, y : y + patch_size_h, z : z + patch_size_w
                    ]
                    patches.append(patch)
                    coordinates.append((x, y, z))

    return patches, coordinates


def remap_labels(label: np.ndarray) -> np.ndarray:
    """Apply the shared label remapping used across training entrypoints."""
    label[label == 2] = 0
    label[label > 2] -= 1
    return label


def load_labeled_files(
    data_dir: str,
    names: Sequence[str],
    tomo_types: Sequence[str],
    data_types: Sequence[str],
    progress_factory: Optional[Callable[[Sequence[str]], Iterable[str]]] = None,
) -> List[dict]:
    """Load image/label pairs using the training scripts' naming convention."""
    files = []
    for tomo_type in tomo_types:
        for data_t in data_types:
            iterator = progress_factory(names) if progress_factory else names
            for name in iterator:
                image = np.load(f"{data_dir}/train_image_{name}_{tomo_type}{data_t}.npy")
                label = np.load(f"{data_dir}/train_label_{name}_{tomo_type}{data_t}.npy")
                files.append({"image": image, "label": remap_labels(label)})
    return files


def build_validation_patch_data(
    valid_files: Sequence[dict], patch_sizes: List[int]
) -> List[dict]:
    """Create the validation patch list expected by MONAI cache datasets."""
    val_images = [item["image"] for item in valid_files]
    val_labels = [item["label"] for item in valid_files]
    val_image_patches, _ = extract_3d_patches_minimal_overlap(val_images, patch_sizes)
    val_label_patches, _ = extract_3d_patches_minimal_overlap(val_labels, patch_sizes)
    return [
        {"image": image_patch, "label": label_patch}
        for image_patch, label_patch in zip(val_image_patches, val_label_patches)
    ]
