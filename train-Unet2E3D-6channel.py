from typing import List

import lightning.pytorch as pl

from models.base_model import BaseModel2D
from models.model2 import Net
from utils.data import load_npy_files, build_dataloaders


class UNet2E3DModel(BaseModel2D):

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 6,
        arch: str = "resnet34d",
        decoder_dim: List[int] = [256, 128, 64, 32, 16],
        beta: float = 0.5,
        alpha: float = 0.5,
        lr: float = 1e-3,
    ):
        super().__init__(out_channels=out_channels, beta=beta, alpha=alpha, lr=lr)
        self.save_hyperparameters()
        self.model = Net(
            out_channels=out_channels,
            arch=arch,
            decoder_dim=decoder_dim,
            pretrained=True,
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
        num_samples=1,
    )

    model = UNet2E3DModel(
        in_channels=1,
        out_channels=6,
        arch="resnet18d",
        decoder_dim=[80, 80, 64, 32, 16],
        lr=learning_rate,
        beta=0.5,
        alpha=0.5,
    )

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
            pl.callbacks.EarlyStopping(
                monitor="val_metric", patience=20, mode="max"
            ),
        ],
    )

    trainer.fit(model, train_loader, valid_loader)
