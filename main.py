import os
os.environ["SLURM_JOB_NAME"] = "bash"

import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import (
    DeviceStatsMonitor,
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping,
)
from pathlib import Path
from argparse import ArgumentParser
import yaml

from dataset import LitICCAD
from model import * 

parser = ArgumentParser()
parser.add_argument("-c", "--config", type=str, default="configs/mlp.yaml")
parser.add_argument("--debug", action='store_true')
args = parser.parse_args()
with open(args.config, "r") as f:
    config = yaml.safe_load(f)
args.__dict__.update(config)

with open(args.csv_file, "r") as f:
    data_size = len(f.readlines()) - 1
for fold in range(args.k_fold):

    trn_split = list(range(0, fold * data_size // args.k_fold)) + list(range((fold + 1) * data_size // args.k_fold, data_size))
    val_split = list(range(fold * data_size // args.k_fold, (fold + 1) * data_size // args.k_fold))

    model = eval(args.model_name)(args)
    dm = LitICCAD(args, trn_split, val_split)

    device_stats = DeviceStatsMonitor()
    lr_monitor = LearningRateMonitor(logging_interval="step")
    ckpt_callback = ModelCheckpoint(
        filename="{epoch}-{val_loss:.4f}",
        save_last=True,
        save_top_k=3,
        mode="min",
        monitor="val_loss",
        # save_on_train_epoch_end=True,
    )
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=args.patience * 10,
        mode="min",
    )
    callbacks = [device_stats, lr_monitor, ckpt_callback, early_stopping] if not args.debug else []
    logger = TensorBoardLogger(Path(__file__).parent / "tf_logs", name=args.name, version=fold) if not args.debug else False 

    trainer = L.Trainer(
        accelerator="gpu",
        devices=-1,
        strategy="ddp",
        callbacks=callbacks,
        max_epochs=-1,
        logger=logger,
        log_every_n_steps=1,
        # check_val_every_n_epoch=5000,
        # num_sanity_val_steps=0,
    )
    trainer.fit(model, datamodule=dm)
