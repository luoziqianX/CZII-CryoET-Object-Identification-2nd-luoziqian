# %%
import numpy as np
import torch
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

# %%
import os
from tqdm import tqdm

from monai.transforms import (
    Compose,
    Orientationd,
    RandFlipd,
    RandShiftIntensityd,
    RandRotate90d,
)
from monai.data import (
    CacheDataset,
    decollate_batch,
)
from monai.metrics import DiceMetric
from monai.losses import DiceCELoss
import torch
import warnings
from utils.patching import extract_3d_patches_minimal_overlap


warnings.filterwarnings("ignore")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Note: I should probably averge the overlapping areas,
# but here they are just overwritten by the most recent one.


# %%
# TRAIN_DATA_DIR = "./numpy-data-types-point"
TRAIN_DATA_DIR = "./numpy-data-types-point-C"
TEST_DATA_DIR = "./data"


# %%
train_names = ["TS_5_4", "TS_69_2", "TS_6_6", "TS_73_6", "TS_86_3", "TS_99_9"]
valid_names = ["TS_6_4"]
tomo_type_list = ["ctfdeconvolved"] + ["denoised"] + ["isonetcorrected"] + ["wbp"]
test_tomo_type_list = ["denoised"]
data_type = [""]

train_files = []
valid_files = []

for tomo_type in tomo_type_list:
    for data_t in data_type:
        for name in tqdm(train_names):
            image = np.load(
                f"{TRAIN_DATA_DIR}/train_image_{name}_{tomo_type}{data_t}.npy"
            )
            label = np.load(
                f"{TRAIN_DATA_DIR}/train_label_{name}_{tomo_type}{data_t}.npy"
            )
            label[label == 2] = 0
            label[label > 2] -= 1

            train_files.append({"image": image, "label": label})

for tomo_type in test_tomo_type_list:
    for data_t in data_type:
        for name in tqdm(valid_names):
            image = np.load(
                f"{TRAIN_DATA_DIR}/train_image_{name}_{tomo_type}{data_t}.npy"
            )
            label = np.load(
                f"{TRAIN_DATA_DIR}/train_label_{name}_{tomo_type}{data_t}.npy"
            )
            label[label == 2] = 0
            label[label > 2] -= 1

            valid_files.append({"image": image, "label": label})


# %% [markdown]
# # Trainiing Dataset and Dataloader

# %%
# Non-random transforms to be cached
non_random_transforms = Compose(
    [
        EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
        # NormalizeIntensityd(keys="image"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
    ]
)

val_random_transforms = Compose(
    [
        EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
        NormalizeIntensityd(keys="image"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
    ]
)

raw_train_ds = CacheDataset(
    data=train_files, transform=non_random_transforms, cache_rate=1.0
)

my_num_samples = 1
train_batch_size = 1
# val_patch_sizes = [128, 384, 384]
val_patch_sizes = [128, 384, 384]
from monai.transforms import (
    RandAffined,
    RandGaussianNoised,
    RandStdShiftIntensityd,
)

# Random transforms to be applied during training
random_transforms = Compose(
    [
        # NormalizeIntensityd(keys="image"),
        RandCropByLabelClassesd(
            keys=["image", "label"],
            label_key="label",
            spatial_size=val_patch_sizes,
            num_classes=6,
            num_samples=my_num_samples,
        ),
        NormalizeIntensityd(keys="image"),
        RandFlipd(keys=["image", "label"], prob=0.3, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.3, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.3, spatial_axis=2),
        RandShiftIntensityd(
            keys=["image"],
            offsets=0.10,
            prob=0.50,
        ),
        RandStdShiftIntensityd(
            keys=["image"],
            prob=0.5,
            factors=0.1,
        ),
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

train_ds = Dataset(data=raw_train_ds, transform=random_transforms)


# DataLoader remains the same
train_loader = DataLoader(
    train_ds,
    batch_size=train_batch_size,
    shuffle=True,
    num_workers=16,
    pin_memory=torch.cuda.is_available(),
)

# %% [markdown]
# # val Dataset and Dataloader

# %%
val_images, val_labels = [dcts["image"] for dcts in valid_files], [
    dcts["label"] for dcts in valid_files
]

val_image_patches, _ = extract_3d_patches_minimal_overlap(val_images, val_patch_sizes)
val_label_patches, _ = extract_3d_patches_minimal_overlap(val_labels, val_patch_sizes)

val_patched_data = [
    {"image": img, "label": lbl}
    for img, lbl in zip(val_image_patches, val_label_patches)
]


valid_ds = CacheDataset(
    data=val_patched_data, transform=val_random_transforms, cache_rate=1.0
)


valid_batch_size = 1
# DataLoader remains the same
valid_loader = DataLoader(
    valid_ds,
    batch_size=valid_batch_size,
    shuffle=False,
    num_workers=16,
    pin_memory=torch.cuda.is_available(),
)
from torch import nn

# %% [markdown]
# # pytorch-lighting training pipeline

# %%
import lightning.pytorch as pl

from monai.networks.nets import SegResNet
from monai.losses import TverskyLoss
from monai.metrics import DiceMetric


# from PMFSNetV5 import PMFSNet


class Model2D(pl.LightningModule):

    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 6,
        dropout_prob=0.3,
        beta: float = 0.5,
        alpha: float = 0.5,
        lr: float = 1e-3,
        upsample_mode="deconv",
    ):

        super().__init__()
        self.save_hyperparameters()
        model = SegResNet(
            in_channels=self.hparams.in_channels,
            out_channels=self.hparams.out_channels,
            dropout_prob=dropout_prob,
            upsample_mode=upsample_mode,
        )
        self.model = model

        self.loss_fn = TverskyLoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True,
            sigmoid=False,
            reduction="none",
        )  # softmax=True for multiclass
        self.loss_fn_1 = DiceCELoss(
            to_onehot_y=True,
            softmax=True,
            sigmoid=False,
        )
        self.loss_fn_background_1 = DiceCELoss(
            to_onehot_y=True,
            softmax=True,
            sigmoid=False,
        )
        self.loss_fn_background = TverskyLoss(
            include_background=True,
            to_onehot_y=True,
            softmax=True,
            sigmoid=False,
            reduction="none",
            beta=self.hparams.beta,
            alpha=self.hparams.alpha,
        )
        self.metric_fn = DiceMetric(
            include_background=False, reduction="mean", ignore_empty=True
        )
        self.metric_fn_background = DiceMetric(
            include_background=True,
            reduction="mean",
            ignore_empty=True,
        )
        self.die = DiceMetric(
            include_background=False, reduction="mean", ignore_empty=True
        )
        self.cross_entropy_loss_fn = nn.CrossEntropyLoss()
        weight = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
        weight /= weight.sum()
        self.weight = weight
        weight_bg = torch.tensor([0.1, 1.0, 1.0, 1.0, 1.0, 1.0])
        weight_bg /= weight_bg.sum()
        self.weight_bg = weight_bg

        self.train_loss = 0
        self.train_loss_background = 0
        self.train_loss_background_list = [0] * 6
        self.val_loss_background_list = [0] * 6
        self.val_metric = 0
        self.val_metric_background = 0
        self.val_die = 0
        self.train_metric = 0
        self.train_metric_background = 0
        self.num_train_batch = 0
        self.num_val_batch = 0
        self.val_loss = 0
        self.val_loss_background = 0
        self.train_ce_loss = 0
        self.val_ce_loss = 0

    def forward(self, image):
        label_pre = self.model(image)
        return label_pre

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        y_one_hot = torch.nn.functional.one_hot(y[:, 0].long(), num_classes=6).permute(
            0, 4, 1, 2, 3
        )  # N,6,H,W,D
        y_one_hot = y_one_hot.to(x.dtype)
        y_hat = self(x)
        weight_bg = self.weight_bg.to(x.device).to(x.dtype)
        loss_background = self.loss_fn_background(y_hat, y)
        ce_loss = self.cross_entropy_loss_fn(y_hat, y[:, 0].long()).mean()
        self.train_ce_loss += ce_loss.mean()
        self.train_loss_background += loss_background.mean()

        loss_background_list = loss_background.mean(dim=0).tolist()
        self.train_loss_background_list = [
            i + j for i, j in zip(self.train_loss_background_list, loss_background_list)
        ]

        self.num_train_batch += 1
        loss_background = (loss_background @ weight_bg).mean()
        if ce_loss < 0.05:  # 0.02
            alpha = 0.05 / ce_loss.detach().item()
            ce_loss = ce_loss * alpha  # make sure the ce_loss is not too small

        torch.cuda.empty_cache()
        return +loss_background + ce_loss

    def on_train_epoch_end(self):
        loss_per_epoch_background = self.train_loss_background / self.num_train_batch
        self.log("train_loss_background", loss_per_epoch_background, prog_bar=True)
        train_metric_per_epoch = self.train_metric / self.num_train_batch
        self.log("train_metric", train_metric_per_epoch, prog_bar=False)
        for i in range(6):
            idx = i if i <= 1 else i + 1
            background_loss = self.train_loss_background_list[i] / self.num_train_batch
            self.log(f"train_loss_background_{idx}", background_loss, prog_bar=True)

        ce_loss_per_epoch = self.train_ce_loss / self.num_train_batch
        self.log("train_ce_loss", ce_loss_per_epoch, prog_bar=True)

        self.train_loss = 0
        self.train_metric = 0
        self.num_train_batch = 0
        self.train_loss_background = 0
        self.train_loss_background_list = [0] * 6
        self.train_metric_background = 0
        self.train_ce_loss = 0

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():  # This ensures that gradients are not stored in memory
            x, y = batch["image"], batch["label"]
            y_one_hot = torch.nn.functional.one_hot(
                y[:, 0].long(), num_classes=6
            ).permute(
                0, 4, 1, 2, 3
            )  # N,6,H,W,D
            y_one_hot = y_one_hot.to(x.dtype)
            # y_hat = self(x)
            y_hat = self(x)
            metric_val_outputs = [
                AsDiscrete(argmax=True, to_onehot=self.hparams.out_channels)(i)
                for i in decollate_batch(y_hat)
            ]
            metric_val_labels = [
                AsDiscrete(to_onehot=self.hparams.out_channels)(i)
                for i in decollate_batch(y)
            ]
            ce_loss = self.cross_entropy_loss_fn(y_hat, y[:, 0].long()).mean()
            self.val_ce_loss += ce_loss.mean()

            # compute metric for current iteration
            self.metric_fn(y_pred=metric_val_outputs, y=metric_val_labels)
            metrics = self.metric_fn.aggregate(reduction="mean_batch")
            val_die = torch.mean(metrics)
            val_loss = self.loss_fn(y_hat, y)
            val_loss_background = self.loss_fn_background(y_hat, y)
            val_loss_background_list = val_loss_background.mean(dim=0).tolist()

            self.val_loss_background_list = [
                i + j
                for i, j in zip(self.val_loss_background_list, val_loss_background_list)
            ]

            val_metric_loss = self.loss_fn_1(y_hat, y)
            val_metric_loss_background = self.loss_fn_background_1(y_hat, y)
            val_metric = 1 - val_metric_loss
            val_metric_background = 1 - val_metric_loss_background
            # val_f1_score = self.f1(y_hat, y)
            self.val_metric += val_metric
            self.val_metric_background += val_metric_background
            self.num_val_batch += 1
            self.val_die += val_die
            self.val_loss += val_loss.mean()
            self.val_loss_background += val_loss_background.mean()
        torch.cuda.empty_cache()
        return {"val_metric": val_metric}

    def on_validation_epoch_end(self):
        metric_per_epoch = self.val_metric / self.num_val_batch
        metric_per_epoch_background = self.val_metric_background / self.num_val_batch
        self.log(
            "val_metric", metric_per_epoch, prog_bar=True, sync_dist=False
        )  # sync_dist=True for distributed training
        val_die = self.val_die / self.num_val_batch
        self.log("val_die", val_die, prog_bar=True, sync_dist=False)
        self.log(
            "val_metric_background",
            metric_per_epoch_background,
            prog_bar=True,
            sync_dist=False,
        )
        loss_per_epoch = self.val_loss / self.num_val_batch
        loss_per_epoch_background = self.val_loss_background / self.num_val_batch
        self.log("val_loss", loss_per_epoch, prog_bar=True, sync_dist=False)
        self.log(
            "val_loss_background",
            loss_per_epoch_background,
            prog_bar=True,
            sync_dist=False,
        )
        for i in range(6):
            idx = i if i <= 1 else i + 1
            background_loss = self.val_loss_background_list[i] / self.num_val_batch
            self.log(f"val_loss_background_{idx}", background_loss, prog_bar=True)

        ce_loss_per_epoch = self.val_ce_loss / self.num_val_batch
        self.log("val_ce_loss", ce_loss_per_epoch, prog_bar=True)

        self.val_ce_loss = 0
        self.val_loss_background_list = [0] * 6
        self.val_loss = 0
        self.val_loss_background = 0
        self.val_metric = 0
        self.val_metric_background = 0
        self.val_die = 0
        self.num_val_batch = 0

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return optimizer


# %%
# learning_rate = 5e-4
learning_rate = 1e-3
# learning_rate = 5e-3
num_epochs = 1000

model = Model2D(
    in_channels=1,
    out_channels=6,
    dropout_prob=0.1,
    upsample_mode="deconv",
    lr=learning_rate,
    beta=0.5,
    alpha=0.5,
)


# %%
trainer = pl.Trainer(
    max_epochs=num_epochs,
    accelerator="gpu",
    devices=[0],
    num_nodes=1,
    log_every_n_steps=5,
    check_val_every_n_epoch=5,
    enable_checkpointing=True,
    enable_progress_bar=True,
    callbacks=[
        pl.callbacks.ModelCheckpoint(
            every_n_epochs=5,
            save_top_k=100,
            monitor="epoch",
            mode="max",
            save_last=True,
            save_on_train_epoch_end=True,
            filename="{epoch}-{val_loss:.2f}-{val_metric:.2f}-{step}",
        ),
        pl.callbacks.EarlyStopping(monitor="val_metric", patience=20, mode="max"),
    ],
)

# %%
trainer.fit(model, train_loader, valid_loader)

# %%
