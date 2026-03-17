# %%
import os
import warnings

import lightning.pytorch as pl
import torch

from models.lightning_module import SegmentationLightningModule
from models.model2 import Net
from utils.training_data import (
    TRAIN_DATA_DIR,
    VAL_PATCH_SIZES,
    create_dataloaders,
    load_data_files,
)

warnings.filterwarnings("ignore")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
train_files, valid_files = load_data_files(
    train_data_dir=TRAIN_DATA_DIR,
    use_tqdm=True,
)

train_loader, valid_loader = create_dataloaders(
    train_files=train_files,
    valid_files=valid_files,
    val_patch_sizes=VAL_PATCH_SIZES,
    num_samples=1,
    train_batch_size=1,
    valid_batch_size=1,
    num_workers=16,
)

# %%
model_backend = Net(
    out_channels=6,
    arch="resnet18d",
    decoder_dim=[80, 80, 64, 32, 16],
    pretrained=True,
)

model = SegmentationLightningModule(
    model=model_backend,
    out_channels=6,
    lr=1e-3,
    beta=0.5,
    alpha=0.5,
    dice_ce_sigmoid=False,
)

# %%
num_epochs = 1000
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
