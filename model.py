import torch
import torch.nn as nn
import torch.optim as optim

import lightning as L


class MLP(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.channels = channels
        self.layers = []
        for i in range(len(channels) - 1):
            self.layers.append(nn.Linear(channels[i], channels[i + 1]))
            self.layers.append(nn.ReLU())
        self.layers = self.layers[:-1]
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)


class LitMLP(L.LightningModule):

    def __init__(self, args):
        super().__init__()

        self.args = args

        self.model = MLP(args.channels)
        self.loss = nn.MSELoss()

        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        iid, area, dpw, sleak, gleak, gt_area, gt_pw = batch
        x = torch.stack([area, dpw, sleak, gleak], dim=1)
        output = self(x)
        pw_hat, area_hat = output[:, 0], output[:, 1]
        loss = self.loss(pw_hat, gt_pw) + self.loss(area_hat, gt_area)
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        iid, area, dpw, sleak, gleak, gt_area, gt_pw = batch
        x = torch.stack([area, dpw, sleak, gleak], dim=1)
        output = self(x)
        pw_hat, area_hat = output[:, 0], output[:, 1]
        loss = self.loss(pw_hat, gt_pw) + self.loss(area_hat, gt_area)
        self.log("val_loss", loss, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=self.args.patience
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }


if __name__ == "__main__":
    pass
