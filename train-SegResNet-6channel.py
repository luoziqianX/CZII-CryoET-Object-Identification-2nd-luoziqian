import lightning.pytorch as pl
from monai.networks.nets import SegResNet

from models.base_model import BaseModel2D
from utils.data import load_npy_files, build_dataloaders


class SegResNetModel(BaseModel2D):

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 6,
        dropout_prob: float = 0.3,
        upsample_mode: str = "deconv",
        beta: float = 0.5,
        alpha: float = 0.5,
        lr: float = 1e-3,
    ):
        super().__init__(out_channels=out_channels, beta=beta, alpha=alpha, lr=lr)
        self.save_hyperparameters()
        self.model = SegResNet(
            in_channels=in_channels,
            out_channels=out_channels,
            dropout_prob=dropout_prob,
            upsample_mode=upsample_mode,
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

    model = SegResNetModel(
        in_channels=1,
        out_channels=6,
        dropout_prob=0.1,
        upsample_mode="deconv",
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
