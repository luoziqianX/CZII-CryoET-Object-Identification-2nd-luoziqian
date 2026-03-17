"""
3D patch extraction utilities for segmentation training.
"""
from typing import List, Tuple

import numpy as np


def calculate_patch_starts(dimension_size: int, patch_size: int) -> List[int]:
    """
    Calculate the starting positions of patches along a single dimension
    with minimal overlap to cover the entire dimension.

    Parameters:
    -----------
    dimension_size : int
        Size of the dimension
    patch_size : int
        Size of the patch in this dimension

    Returns:
    --------
    List[int]
        List of starting positions for patches
    """
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
    """
    Extract 3D patches from multiple arrays with minimal overlap to cover the entire array.

    Parameters:
    -----------
    arrays : List[np.ndarray]
        List of input arrays, each with shape (m, n, l)
    patch_sizes : List[int]
        Patch sizes [D, H, W] for each dimension

    Returns:
    --------
    patches : List[np.ndarray]
        List of all patches from all input arrays
    coordinates : List[Tuple[int, int, int]]
        List of starting coordinates (x, y, z) for each patch
    """
    patch_size_d, patch_size_h, patch_size_w = patch_sizes
    shape = arrays[0].shape
    D, H, W = shape
    if not all(arr.shape == shape for arr in arrays):
        raise ValueError("All input arrays must have the same shape")
    if patch_size_d > D or patch_size_h > H or patch_size_w > W:
        raise ValueError(
            f"patch_size ({patch_size_d, patch_size_h, patch_size_w}) must be smaller than shape {shape}"
        )

    m, n, l = shape
    patches = []
    coordinates = []

    x_starts = calculate_patch_starts(m, patch_size_d)
    y_starts = calculate_patch_starts(n, patch_size_h)
    z_starts = calculate_patch_starts(l, patch_size_w)

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
