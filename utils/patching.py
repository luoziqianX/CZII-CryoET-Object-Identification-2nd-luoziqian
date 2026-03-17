from typing import List, Tuple

import numpy as np


def calculate_patch_starts(dimension_size: int, patch_size: int) -> List[int]:
    """
    Calculate the starting positions of patches along a single dimension
    with minimal overlap to cover the entire dimension.
    """
    if dimension_size <= patch_size:
        return [0]

    # Calculate number of patches needed.
    n_patches = np.ceil(dimension_size / patch_size)
    if n_patches == 1:
        return [0]

    # Calculate overlap.
    total_overlap = (n_patches * patch_size - dimension_size) / (n_patches - 1)

    # Generate starting positions.
    positions = []
    for i in range(int(n_patches)):
        pos = int(i * (patch_size - total_overlap))
        if pos + patch_size > dimension_size:
            pos = dimension_size - patch_size
        if pos not in positions:  # Avoid duplicates.
            positions.append(pos)

    return positions


def extract_3d_patches_minimal_overlap(
    arrays: List[np.ndarray], patch_sizes: List[int]
) -> Tuple[List[np.ndarray], List[Tuple[int, int, int]]]:
    """
    Extract 3D patches from multiple arrays with minimal overlap to cover
    the entire array.
    """
    patch_size_d, patch_size_h, patch_size_w = patch_sizes
    shape = arrays[0].shape
    d_size, h_size, w_size = shape
    if not all(arr.shape == shape for arr in arrays):
        raise ValueError("All input arrays must have the same shape")
    if patch_size_d > d_size or patch_size_h > h_size or patch_size_w > w_size:
        raise ValueError(
            f"patch_size ({patch_size_d, patch_size_h, patch_size_w}) must be smaller than shape {shape}"
        )

    patches = []
    coordinates = []

    # Calculate starting positions for each dimension.
    x_starts = calculate_patch_starts(d_size, patch_size_d)
    y_starts = calculate_patch_starts(h_size, patch_size_h)
    z_starts = calculate_patch_starts(w_size, patch_size_w)

    # Extract patches from each array.
    for arr in arrays:
        for x in x_starts:
            for y in y_starts:
                for z in z_starts:
                    patch = arr[
                        x : x + patch_size_d,
                        y : y + patch_size_h,
                        z : z + patch_size_w,
                    ]
                    patches.append(patch)
                    coordinates.append((x, y, z))

    return patches, coordinates
