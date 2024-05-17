import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import lightning as L
import pandas as pd

from pathlib import Path


class ICCAD(Dataset):

    def __init__(self, csv_file, split):
        super().__init__()

        self.csv_file = Path(csv_file)
        self.data = pd.read_csv(self.csv_file, dtype='float32').iloc[split]
        self.data.reset_index(inplace=True, drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        iid = self.data.at[idx, "id"]
        area = self.data.at[idx, "area"]
        dpw = self.data.at[idx, "dynamic"]
        sleak = self.data.at[idx, "sleak"]
        gleak = self.data.at[idx, "gleak"]
        gt_area = self.data.at[idx, "gt_area"]
        gt_pw = self.data.at[idx, "gt_pw"]
        return iid, area, dpw, sleak, gleak, gt_area, gt_pw 


class LitICCAD(L.LightningDataModule):

    def __init__(self, args, trn_split, val_split):
        super().__init__()
        self.args = args
        self.trn_split = trn_split
        self.val_split = val_split

    def setup(self, stage=None):
        self.train_ds = ICCAD(self.args.csv_file, self.trn_split)
        self.val_ds = ICCAD(self.args.csv_file, self.val_split)

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
    dataset = ICCAD("data/data.csv", [0, 100])
    res = dataset[0]
    breakpoint()
