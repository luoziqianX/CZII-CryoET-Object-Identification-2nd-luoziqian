from typing import List, Tuple, Union

import lightning.pytorch as pl
from monai.networks.nets import UNet

from models.base_model import BaseModel2D
from utils.data import load_npy_files, build_dataloaders


class UNet3DModel(BaseModel2D):

    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 6,
        channels: Union[Tuple[int, ...], List[int]] = (48, 64, 80, 80),
        strides: Union[Tuple[int, ...], List[int]] = (2, 2, 1),
        num_res_units: int = 1,
        beta: float = 0.5,
        alpha: float = 0.5,
        lr: float = 1e-3,
    ):
        super().__init__(out_channels=out_channels, beta=beta, alpha=alpha, lr=lr)
        self.save_hyperparameters()
        self.model = UNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
        )


if __name__ == "__main__":
    learning_rate = 1e-3
    num_epochs = 1000
    val_patch_sizes = [128, 384, 384]

    train_files, valid_files = load_npy_files()
    train_loader, valid_loader = build_dataloaders(
        train_files,
        valid_files,
        val_patch_sizes=val_patch_sizes,
        num_samples=2,
    )

    model = UNet3DModel(
        in_channels=1,
        out_channels=6,
        lr=learning_rate,
        beta=0.5,
        alpha=0.5,
    )

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator="gpu",
        devices=[0],
        num_nodes=1,
        log_every_n_steps=2,
        check_val_every_n_epoch=2,
        enable_checkpointing=True,
        enable_progress_bar=True,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                every_n_epochs=2,
                save_top_k=100,
                monitor="epoch",
                mode="max",
                save_last=True,
                save_on_train_epoch_end=True,
                filename="{epoch}-{val_loss:.2f}-{val_metric:.2f}-{step}",
            ),
            pl.callbacks.EarlyStopping(
                monitor="val_metric", patience=20, mode="max"
            ),
        ],
    )

    trainer.fit(model, train_loader, valid_loader)
