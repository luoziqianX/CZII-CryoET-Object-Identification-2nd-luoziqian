import os
import shutil

import copick
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from copick_utils.segmentation import segmentation_from_picks
import copick_utils.writers.write as write


# ---- Copick project configuration ----

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

os.makedirs("./working/", exist_ok=True)
with open(copick_config_path, "w") as f:
    f.write(config_blob)


# ---- Copy overlay files with curation prefix ----

source_dir = "./data/train/overlay"
destination_dir = "./working/overlay"

for root, dirs, files in os.walk(source_dir):
    relative_path = os.path.relpath(root, source_dir)
    target_dir = os.path.join(destination_dir, relative_path)
    os.makedirs(target_dir, exist_ok=True)

    for file in files:
        new_filename = file if file.startswith("curation_0_") else f"curation_0_{file}"
        source_file = os.path.join(root, file)
        destination_file = os.path.join(target_dir, new_filename)
        shutil.copy2(source_file, destination_file)
        print(f"Copied {source_file} to {destination_file}")


# ---- Generate numpy dataset from copick data ----

tomo_type_list = ["ctfdeconvolved", "denoised", "isonetcorrected", "wbp"]
copick_user_name = "copickUtils"
copick_segmentation_name = "paintedPicks"
voxel_size = 10
dir_name = "./numpy-data-types-point-C"

generate_masks = True

for tomo_type in tomo_type_list:
    root = copick.from_file(copick_config_path)

    if generate_masks:
        target_objects = defaultdict(dict)
        for obj in root.pickable_objects:
            if obj.is_particle:
                target_objects[obj.name]["label"] = obj.label
                target_objects[obj.name]["radius"] = obj.radius

        for run in tqdm(root.runs, desc=f"Generating masks ({tomo_type})"):
            tomo = run.get_voxel_spacing(10)
            tomo = tomo.get_tomogram(tomo_type).numpy()
            target = np.zeros(tomo.shape, dtype=np.uint8)
            for pickable_object in root.pickable_objects:
                pick = run.get_picks(
                    object_name=pickable_object.name, user_id="curation"
                )
                if len(pick):
                    target = segmentation_from_picks.from_picks(
                        pick[0],
                        target,
                        target_objects[pickable_object.name]["radius"] * 0.4,
                        target_objects[pickable_object.name]["label"],
                    )
            write.segmentation(
                run, target, copick_user_name, name=copick_segmentation_name
            )

    os.makedirs(dir_name, exist_ok=True)
    data_dicts = []
    for run in tqdm(root.runs, desc=f"Saving numpy ({tomo_type})"):
        tomogram = run.get_voxel_spacing(voxel_size).get_tomogram(tomo_type).numpy()
        tomogram_rot90 = np.rot90(tomogram, axes=(1, 2))
        segmentation = run.get_segmentations(
            name=copick_segmentation_name,
            user_id=copick_user_name,
            voxel_size=voxel_size,
            is_multilabel=True,
        )[0].numpy()
        segmentation_rot90 = np.rot90(segmentation, axes=(1, 2))
        data_dicts.append(
            {
                "name": run.name,
                "image": tomogram,
                "image-rot90": tomogram_rot90,
                "label": segmentation,
                "label-rot90": segmentation_rot90,
                "tomo_type": tomo_type,
            }
        )

    for i in range(len(data_dicts)):
        name = data_dicts[i]["name"]
        tt = data_dicts[i]["tomo_type"]

        np.save(f"{dir_name}/train_image_{name}_{tt}.npy", data_dicts[i]["image"])
        np.save(f"{dir_name}/train_image_{name}_{tt}-rot90.npy", data_dicts[i]["image-rot90"])
        np.save(f"{dir_name}/train_label_{name}_{tt}.npy", data_dicts[i]["label"])
        np.save(f"{dir_name}/train_label_{name}_{tt}-rot90.npy", data_dicts[i]["label-rot90"])
