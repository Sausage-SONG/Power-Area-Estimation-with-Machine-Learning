import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import lightning as L
import pandas as pd

from pathlib import Path


class ICCAD(Dataset):

    def __init__(self, csv_file, split, alpha):
        super().__init__()

        self.csv_file = Path(csv_file)
        df = pd.read_csv(self.csv_file, dtype='float32').iloc[split]
        df = df.drop(columns=["gt_pf", "gt_t"])
        df.reset_index(inplace=True, drop=True)
        df = torch.tensor(df.values)
        gt_pw_err = (df[:, 2:5].sum(dim=1) - df[:, -2]).abs().unsqueeze(-1)
        gt_area_err = (df[:, 1] - df[:, -1]).abs().unsqueeze(-1)

        norm = (df - df.mean(dim=0)) / df.std(dim=0)
        self.data = torch.hstack([norm[:, :-2], df[:, [-2]] * alpha[-2], df[:, [-1]]*alpha[-1], gt_pw_err, gt_area_err])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        iid, area, dpw, sleak, gleak, gt_pw, gt_area, gt_pw_err, gt_area_err = self.data[idx]
        return iid, area, dpw, sleak, gleak, gt_pw, gt_area, gt_pw_err, gt_area_err

class LitICCAD(L.LightningDataModule):

    def __init__(self, args, trn_split, val_split):
        super().__init__()
        self.args = args
        self.trn_split = trn_split
        self.val_split = val_split

    def setup(self, stage=None):
        self.train_ds = ICCAD(self.args.csv_file, self.trn_split, self.args.alpha)
        self.val_ds = ICCAD(self.args.csv_file, self.val_split, self.args.alpha)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.args.batch_size,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.args.batch_size, pin_memory=True)


if __name__ == "__main__":
    dataset = ICCAD("data/data.csv", list(range(15000)), [10, 1e-7])
    res = dataset[0]
