from typing import List, Tuple

import numpy as np


def calculate_patch_starts(dimension_size: int, patch_size: int) -> List[int]:
    """
    Calculate the starting positions of patches along a single dimension
    with minimal overlap to cover the entire dimension.

    Parameters
    ----------
    dimension_size : int
        Size of the dimension
    patch_size : int
        Size of the patch in this dimension

    Returns
    -------
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

    Parameters
    ----------
    arrays : List[np.ndarray]
        List of input arrays, each with shape (D, H, W)
    patch_sizes : List[int]
        Patch size for each dimension [depth, height, width]

    Returns
    -------
    patches : List[np.ndarray]
        List of all patches from all input arrays
    coordinates : List[Tuple[int, int, int]]
        List of starting coordinates (d, h, w) for each patch
    """
    patch_size_d, patch_size_h, patch_size_w = patch_sizes
    shape = arrays[0].shape
    D, H, W = shape

    if not all(arr.shape == shape for arr in arrays):
        raise ValueError("All input arrays must have the same shape")
    if patch_size_d > D or patch_size_h > H or patch_size_w > W:
        raise ValueError(
            f"patch_size ({patch_size_d}, {patch_size_h}, {patch_size_w}) "
            f"must be smaller than shape {shape}"
        )

    patches = []
    coordinates = []

    d_starts = calculate_patch_starts(D, patch_size_d)
    h_starts = calculate_patch_starts(H, patch_size_h)
    w_starts = calculate_patch_starts(W, patch_size_w)

    for arr in arrays:
        for d in d_starts:
            for h in h_starts:
                for w in w_starts:
                    patch = arr[
                        d : d + patch_size_d,
                        h : h + patch_size_h,
                        w : w + patch_size_w,
                    ]
                    patches.append(patch)
                    coordinates.append((d, h, w))

    return patches, coordinates
