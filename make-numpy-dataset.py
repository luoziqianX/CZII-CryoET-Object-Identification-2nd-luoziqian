# %%
import pdb

config_blob = """{
    "name": "czii_cryoet_mlchallenge_2024",
    "description": "2024 CZII CryoET ML Challenge training data.",
    "version": "1.0.0",

    "pickable_objects": [
        {
            "name": "apo-ferritin",
            "is_particle": true,
            "pdb_id": "4V1W",
            "label": 1,
            "color": [  0, 117, 220, 128],
            "radius": 80,
            "map_threshold": 0.0418
        },
        {
            "name": "beta-amylase",
            "is_particle": true,
            "pdb_id": "1FA2",
            "label": 2,
            "color": [153,  63,   0, 128],
            "radius": 65,
            "map_threshold": 0.035
        },
        {
            "name": "beta-galactosidase",
            "is_particle": true,
            "pdb_id": "6X1Q",
            "label": 3,
            "color": [ 76,   0,  92, 128],
            "radius": 90,
            "map_threshold": 0.0578
        },
        {
            "name": "ribosome",
            "is_particle": true,
            "pdb_id": "6EK0",
            "label": 4,
            "color": [  0,  92,  49, 128],
            "radius": 150,
            "map_threshold": 0.0374
        },
        {
            "name": "thyroglobulin",
            "is_particle": true,
            "pdb_id": "6SCJ",
            "label": 5,
            "color": [ 43, 206,  72, 128],
            "radius": 120,
            "map_threshold": 0.0278
        },
        {
            "name": "virus-like-particle",
            "is_particle": true,
            "pdb_id": "6N4V",            
            "label": 6,
            "color": [255, 204, 153, 128],
            "radius": 150,
            "map_threshold": 0.201
        }
    ],

    "overlay_root": "./working/overlay",

    "overlay_fs_args": {
        "auto_mkdir": true
    },

    "static_root": "./data/train/static"
}"""

copick_config_path = "./working/copick.config"
output_overlay = "./working/overlay"
import os

os.makedirs("./working/", exist_ok=True)
with open(copick_config_path, "w") as f:
    f.write(config_blob)

# %%
# Update the overlay
# Define source and destination directories
source_dir = "./data/train/overlay"
destination_dir = "./working/overlay"

# %%
# Make a copick project
import os
import shutil

def with_curation_prefix(filename):
    return filename if filename.startswith("curation_0_") else f"curation_0_{filename}"


def copy_overlay_tree(source_dir, destination_dir):
    for root, _, files in os.walk(source_dir):
        relative_path = os.path.relpath(root, source_dir)
        target_dir = os.path.join(destination_dir, relative_path)
        os.makedirs(target_dir, exist_ok=True)

        for file in files:
            source_file = os.path.join(root, file)
            destination_file = os.path.join(target_dir, with_curation_prefix(file))
            shutil.copy2(source_file, destination_file)
            print(f"Copied {source_file} to {destination_file}")


copy_overlay_tree(source_dir, destination_dir)

# %%
import os
import numpy as np
from pathlib import Path
import torch
import torchinfo
import zarr, copick
from tqdm import tqdm
from monai.data import DataLoader, Dataset, CacheDataset, decollate_batch
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    Orientationd,
    AsDiscrete,
    RandFlipd,
    RandRotate90d,
    NormalizeIntensityd,
    RandCropByLabelClassesd,
)
from monai.networks.nets import UNet
from monai.losses import DiceLoss, FocalLoss, TverskyLoss
from monai.metrics import DiceMetric, ConfusionMatrixMetric
import mlflow
import mlflow.pytorch

# %%
tomo_type_list = ["ctfdeconvolved", "denoised", "isonetcorrected", "wbp"]

# %%
# root = copick.from_file(copick_config_path)

copick_user_name = "copickUtils"
copick_segmentation_name = "paintedPicks"
voxel_size = 10
# tomo_type = "denoised"

# %%
from copick_utils.segmentation import segmentation_from_picks
import copick_utils.writers.write as write
from collections import defaultdict


# Just do this once
generate_masks = True


def build_data_record(run, tomo_type, segmentation_name, user_name, voxel_size):
    tomogram = run.get_voxel_spacing(voxel_size).get_tomogram(tomo_type).numpy()
    segmentation = run.get_segmentations(
        name=segmentation_name,
        user_id=user_name,
        voxel_size=voxel_size,
        is_multilabel=True,
    )[0].numpy()
    return {
        "name": run.name,
        "image": tomogram,
        "image-rot90": np.rot90(tomogram, axes=(1, 2)),
        "label": segmentation,
        "label-rot90": np.rot90(segmentation, axes=(1, 2)),
        "tomo_type": tomo_type,
    }


def save_numpy_record(record, output_dir):
    variants = [
        ("image", "image", ""),
        ("image-rot90", "image", "-rot90"),
        ("label", "label", ""),
        ("label-rot90", "label", "-rot90"),
    ]
    for record_key, file_prefix, suffix in variants:
        output_path = (
            f"{output_dir}/train_{file_prefix}_{record['name']}_{record['tomo_type']}{suffix}.npy"
        )
        with open(output_path, "wb") as f:
            np.save(f, record[record_key])


for tomo_type in tomo_type_list:
    root = copick.from_file(copick_config_path)
    if generate_masks:
        target_objects = defaultdict(dict)
        for object in root.pickable_objects:
            if object.is_particle:
                target_objects[object.name]["label"] = object.label
                target_objects[object.name]["radius"] = object.radius

        for run in tqdm(root.runs):
            tomo = run.get_voxel_spacing(10)
            tomo = tomo.get_tomogram(tomo_type).numpy()
            target = np.zeros(tomo.shape, dtype=np.uint8)
            for pickable_object in root.pickable_objects:
                pick = run.get_picks(
                    object_name=pickable_object.name, user_id="curation"
                )
                # pdb.set_trace()
                if len(pick):
                    # pdb.set_trace()
                    target = segmentation_from_picks.from_picks(
                        pick[0],
                        target,
                        target_objects[pickable_object.name]["radius"] * 0.4,
                        target_objects[pickable_object.name]["label"],
                    )
            write.segmentation(
                run, target, copick_user_name, name=copick_segmentation_name
            )

    data_dicts = []
    dir_name = "./numpy-data-types-point-C"
    os.makedirs(dir_name, exist_ok=True)
    for run in tqdm(root.runs):
        data_dicts.append(
            build_data_record(
                run,
                tomo_type,
                copick_segmentation_name,
                copick_user_name,
                voxel_size,
            )
        )

    for record in data_dicts:
        save_numpy_record(record, dir_name)
