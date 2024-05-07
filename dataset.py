import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import lightning as L

from pathlib import Path


class ICCAD(Dataset):

    def __init__(self, csv_file, split):
        super().__init__()

        self.csv_file = Path(csv_file)
        with self.csv_file.open("r") as f:
            lines = f.readlines()
        lines = list(map(lambda x: x.strip().split("    "), lines))
        lines = [[float(x) for x in row] for row in lines]
        self.data = torch.tensor(lines, dtype=torch.float)[split]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        arch = row[:-4]
        pfm = row[-4]
        pw = row[-3]
        area = row[-2]
        tvf = row[-1]
        return arch, pfm, pw, area, tvf


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
    dataset = ICCAD("contest.csv", [0, 100])
