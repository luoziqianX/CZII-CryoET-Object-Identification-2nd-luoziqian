import torch
import torch.nn as nn
import lightning.pytorch as pl
from monai.data import decollate_batch
from monai.losses import DiceCELoss, TverskyLoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete


class BaseModel2D(pl.LightningModule):
    """
    Base Lightning module for 3D segmentation with shared loss functions,
    metrics, training/validation logic, and optimizer configuration.

    Subclasses must set ``self.model`` in their ``__init__`` before calling
    ``super().__init__()`` or assign it afterwards.
    """

    def __init__(
        self,
        out_channels: int = 6,
        beta: float = 0.5,
        alpha: float = 0.5,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.loss_fn = TverskyLoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True,
            sigmoid=False,
            reduction="none",
        )
        self.loss_fn_1 = DiceCELoss(
            to_onehot_y=True, softmax=True, sigmoid=False
        )
        self.loss_fn_background_1 = DiceCELoss(
            to_onehot_y=True, softmax=True, sigmoid=False
        )
        self.loss_fn_background = TverskyLoss(
            include_background=True,
            to_onehot_y=True,
            softmax=True,
            sigmoid=False,
            reduction="none",
            beta=beta,
            alpha=alpha,
        )
        self.metric_fn = DiceMetric(
            include_background=False, reduction="mean", ignore_empty=True
        )
        self.metric_fn_background = DiceMetric(
            include_background=True, reduction="mean", ignore_empty=True
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

        self._reset_train_accumulators()
        self._reset_val_accumulators()

    def _reset_train_accumulators(self):
        self.train_loss = 0
        self.train_loss_background = 0
        self.train_loss_background_list = [0] * 6
        self.train_metric = 0
        self.train_metric_background = 0
        self.num_train_batch = 0
        self.train_ce_loss = 0

    def _reset_val_accumulators(self):
        self.val_loss_background_list = [0] * 6
        self.val_metric = 0
        self.val_metric_background = 0
        self.val_die = 0
        self.num_val_batch = 0
        self.val_loss = 0
        self.val_loss_background = 0
        self.val_ce_loss = 0

    def forward(self, image):
        return self.model(image)

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        y_hat = self(x)
        weight_bg = self.weight_bg.to(x.device).to(x.dtype)

        loss_background = self.loss_fn_background(y_hat, y)
        ce_loss = self.cross_entropy_loss_fn(y_hat, y[:, 0].long()).mean()

        self.train_ce_loss += ce_loss.mean()
        self.train_loss_background += loss_background.mean()

        loss_background_list = loss_background.mean(dim=0).tolist()
        self.train_loss_background_list = [
            a + b
            for a, b in zip(self.train_loss_background_list, loss_background_list)
        ]
        self.num_train_batch += 1

        loss_background = (loss_background @ weight_bg).mean()
        if ce_loss < 0.05:
            alpha = 0.05 / ce_loss.detach().item()
            ce_loss = ce_loss * alpha

        torch.cuda.empty_cache()
        return loss_background + ce_loss

    def on_train_epoch_end(self):
        n = self.num_train_batch
        self.log("train_loss_background", self.train_loss_background / n, prog_bar=True)
        self.log("train_metric", self.train_metric / n, prog_bar=False)
        for i in range(6):
            idx = i if i <= 1 else i + 1
            self.log(
                f"train_loss_background_{idx}",
                self.train_loss_background_list[i] / n,
                prog_bar=True,
            )
        self.log("train_ce_loss", self.train_ce_loss / n, prog_bar=True)
        self._reset_train_accumulators()

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            x, y = batch["image"], batch["label"]
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

            self.metric_fn(y_pred=metric_val_outputs, y=metric_val_labels)
            metrics = self.metric_fn.aggregate(reduction="mean_batch")
            val_die = torch.mean(metrics)

            val_loss = self.loss_fn(y_hat, y)
            val_loss_background = self.loss_fn_background(y_hat, y)
            val_loss_background_list = val_loss_background.mean(dim=0).tolist()
            self.val_loss_background_list = [
                a + b
                for a, b in zip(
                    self.val_loss_background_list, val_loss_background_list
                )
            ]

            val_metric_loss = self.loss_fn_1(y_hat, y)
            val_metric_loss_background = self.loss_fn_background_1(y_hat, y)
            val_metric = 1 - val_metric_loss
            val_metric_background = 1 - val_metric_loss_background

            self.val_metric += val_metric
            self.val_metric_background += val_metric_background
            self.num_val_batch += 1
            self.val_die += val_die
            self.val_loss += val_loss.mean()
            self.val_loss_background += val_loss_background.mean()

        torch.cuda.empty_cache()
        return {"val_metric": val_metric}

    def on_validation_epoch_end(self):
        n = self.num_val_batch
        self.log("val_metric", self.val_metric / n, prog_bar=True, sync_dist=False)
        self.log("val_die", self.val_die / n, prog_bar=True, sync_dist=False)
        self.log(
            "val_metric_background",
            self.val_metric_background / n,
            prog_bar=True,
            sync_dist=False,
        )
        self.log("val_loss", self.val_loss / n, prog_bar=True, sync_dist=False)
        self.log(
            "val_loss_background",
            self.val_loss_background / n,
            prog_bar=True,
            sync_dist=False,
        )
        for i in range(6):
            idx = i if i <= 1 else i + 1
            self.log(
                f"val_loss_background_{idx}",
                self.val_loss_background_list[i] / n,
                prog_bar=True,
            )
        self.log("val_ce_loss", self.val_ce_loss / n, prog_bar=True)
        self._reset_val_accumulators()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
