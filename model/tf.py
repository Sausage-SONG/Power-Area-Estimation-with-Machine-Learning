import torch
import torch.nn as nn
import torch.optim as optim

import lightning as L


class TF(nn.Module):

    def __init__(self, d_model, n_head, dim_feedforward, norm_first, num_layer, in_channel, out_channel):
        super().__init__()
        
        self.embed = nn.Linear(in_channel, d_model)
        self.output = nn.Linear(d_model, out_channel)

        d_layer = nn.TransformerDecoderLayer(d_model, n_head, dim_feedforward, batch_first=True, norm_first=norm_first)
        self.tf = nn.TransformerDecoder(d_layer, num_layer)

    def forward(self, x):
        x = self.embed(x)
        x = self.tf(x, x)
        return self.output(x)


class LitTF(L.LightningModule):

    def __init__(self, args):
        super().__init__()

        self.args = args

        self.model = TF(args.d_model, args.n_head, args.dim_feedforward, args.norm_first, args.num_layer, args.in_channel, args.out_channel)
        self.loss = nn.MSELoss()

        self.save_hyperparameters()

    def mae(self, x1, x2):
        return (x1 - x2).abs().sum() / len(x1)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        iid, area, dpw, sleak, gleak, gt_pw, gt_area, gt_pw_err, gt_area_err = batch
        x = torch.stack([area, dpw, sleak, gleak], dim=1)
        output = self(x)
        pw_hat, area_hat = output[:, 0], output[:, 1]
        
        loss = self.loss(pw_hat, gt_pw) + self.loss(area_hat, gt_area)
        area_mae = self.mae(area_hat, gt_area) / self.args.alpha[-1]
        pw_mae = self.mae(pw_hat, gt_pw) / self.args.alpha[-2]
        mae = area_mae + pw_mae
        
        self.log("train_loss", loss, sync_dist=True)
        self.log("train_power_mae", pw_mae, sync_dist=True)
        self.log("train_area_mae", area_mae, sync_dist=True)
        self.log("train_gt_pw_err", gt_pw_err.mean(), sync_dist=True)
        self.log("train_gt_area_err", gt_area_err.mean(), sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        iid, area, dpw, sleak, gleak, gt_pw, gt_area, gt_pw_err, gt_area_err = batch
        x = torch.stack([area, dpw, sleak, gleak], dim=1)
        output = self(x)
        pw_hat, area_hat = output[:, 0], output[:, 1]
        
        loss = self.loss(pw_hat, gt_pw) + self.loss(area_hat, gt_area)
        area_mae = self.mae(area_hat, gt_area) / self.args.alpha[-1]
        pw_mae = self.mae(pw_hat, gt_pw) / self.args.alpha[-2]
        mae = area_mae + pw_mae
        
        self.log("val_loss", loss, sync_dist=True)
        self.log("val_power_mae", pw_mae, sync_dist=True)
        self.log("val_area_mae", area_mae, sync_dist=True)
        self.log("val_gt_pw_err", gt_pw_err.mean(), sync_dist=True)
        self.log("val_gt_area_err", gt_area_err.mean(), sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        iid, area, dpw, sleak, gleak, gt_pw, gt_area, gt_pw_err, gt_area_err = batch
        x = torch.stack([area, dpw, sleak, gleak], dim=1)
        output = self(x)
        pw_hat, area_hat = output[:, 0], output[:, 1]
        
        loss = self.loss(pw_hat, gt_pw) + self.loss(area_hat, gt_area)
        area_mae = self.mae(area_hat, gt_area) / self.args.alpha[-1]
        pw_mae = self.mae(pw_hat, gt_pw) / self.args.alpha[-2]
        mae = area_mae + pw_mae
        
        self.log("test_loss", loss, sync_dist=True)
        self.log("test_power_mae", pw_mae, sync_dist=True)
        self.log("test_area_mae", pw_mae, sync_dist=True)
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
