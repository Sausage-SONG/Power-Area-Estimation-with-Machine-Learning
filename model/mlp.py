import torch
import torch.nn as nn
import torch.optim as optim
from einops import rearrange

import lightning as L


class MLP(nn.Module):

    def __init__(self, channels, arch_sizes, arch_embed_dim):
        super().__init__()

        self.channels = channels
        self.layers = []
        for i in range(len(channels) - 1):
            if i == 0:
                self.layers.append(nn.Linear(channels[i]*2, channels[i + 1]))
            else:
                self.layers.append(nn.Linear(channels[i], channels[i + 1]))
            self.layers.append(nn.ReLU())
        self.layers = self.layers[:-1]
        self.layers = nn.Sequential(*self.layers)

        self.arch_embed = []
        for arch_size in arch_sizes:
            self.arch_embed.append(nn.Embedding(arch_size, arch_embed_dim))
        self.arch_embed = nn.ModuleList(self.arch_embed)
        self.arch_output = nn.Linear(arch_embed_dim * len(arch_sizes), channels[0])

    def forward(self, x, arch):
        arch_feature = []
        for i in range(len(self.arch_embed)):
            arch_feature.append(self.arch_embed[i](arch[:, i]))
        arch_feature = torch.stack(arch_feature, dim=1)
        arch_feature = rearrange(arch_feature, "b n d -> b (n d)")
        arch_feature = self.arch_output(arch_feature)

        input = torch.cat([x, arch_feature], dim=1)
        return self.layers(input)


class LitMLP(L.LightningModule):

    def __init__(self, args):
        super().__init__()

        self.args = args

        self.model = MLP(args.channels, args.arch_sizes, args.arch_embed_dim)
        self.loss = nn.MSELoss()
        # self.mae = nn.L1Loss()

        self.save_hyperparameters()

    def mae(self, x1, x2):
        return (x1 - x2).abs().sum() / len(x1)

    def forward(self, x, arch):
        return self.model(x, arch)

    def training_step(self, batch, batch_idx):
        iid, area, dpw, sleak, gleak, arch, gt_pw, gt_area, gt_pw_err, gt_area_err = batch
        x = torch.stack([area, dpw, sleak, gleak], dim=1)
        output = self(x, arch)
        pw_hat, area_hat = output[:, 0], output[:, 1]
        
        loss = self.loss(pw_hat, gt_pw) + self.loss(area_hat, gt_area)
        area_mae = self.mae(area_hat, gt_area) / self.args.alpha[-1]
        pw_mae = self.mae(pw_hat, gt_pw) / self.args.alpha[-2]
        mae = area_mae + pw_mae
        
        self.log("train_loss", loss, sync_dist=True)
        self.log("train_power_mae", pw_mae, sync_dist=True)
        self.log("train_area_mae", area_mae, sync_dist=True)
        # self.log("train_mae", mae, sync_dist=True)
        self.log("train_gt_pw_err", gt_pw_err.mean(), sync_dist=True)
        self.log("train_gt_area_err", gt_area_err.mean(), sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        iid, area, dpw, sleak, gleak, arch, gt_pw, gt_area, gt_pw_err, gt_area_err = batch
        x = torch.stack([area, dpw, sleak, gleak], dim=1)
        output = self(x, arch)
        pw_hat, area_hat = output[:, 0], output[:, 1]
        
        loss = self.loss(pw_hat, gt_pw) + self.loss(area_hat, gt_area)
        area_mae = self.mae(area_hat, gt_area) / self.args.alpha[-1]
        pw_mae = self.mae(pw_hat, gt_pw) / self.args.alpha[-2]
        mae = area_mae + pw_mae
        
        self.log("val_loss", loss, sync_dist=True)
        self.log("val_power_mae", pw_mae, sync_dist=True)
        self.log("val_area_mae", area_mae, sync_dist=True)
        # self.log("val_mae", mae, sync_dist=True)
        self.log("val_gt_pw_err", gt_pw_err.mean(), sync_dist=True)
        self.log("val_gt_area_err", gt_area_err.mean(), sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        iid, area, dpw, sleak, gleak, arch, gt_pw, gt_area, gt_pw_err, gt_area_err = batch
        x = torch.stack([area, dpw, sleak, gleak], dim=1)
        output = self(x, arch)
        pw_hat, area_hat = output[:, 0], output[:, 1]
        
        loss = self.loss(pw_hat, gt_pw) + self.loss(area_hat, gt_area)
        area_mae = self.mae(area_hat, gt_area) / self.args.alpha[-1]
        pw_mae = self.mae(pw_hat, gt_pw) / self.args.alpha[-2]
        mae = area_mae + pw_mae
        
        self.log("test_loss", loss, sync_dist=True)
        self.log("test_power_mae", pw_mae, sync_dist=True)
        self.log("test_area_mae", pw_mae, sync_dist=True)
        # self.log("test_mae", mae, sync_dist=True)
        self.log("test_gt_pw_err", gt_pw_err.mean(), sync_dist=True)
        self.log("test_gt_area_err", gt_area_err.mean(), sync_dist=True)
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
