from models.GRU_atten import GRUAttention
from joblib import load
from config import config
from utils.utils import inverse_norm
from torchmetrics.functional import mean_absolute_percentage_error
from pytorch_lightning.callbacks import ModelCheckpoint
import torch.optim as optim
import pytorch_lightning as pl
import os
import torch
import torch.nn as nn


class CombinedModel(pl.LightningModule):
    def __init__(self):
        super(CombinedModel, self).__init__()
        GRUatten_ckpt, lgb_ckpt = None, None
        for file in os.listdir(config.mdoel_saving_dir):
            if file.startswith('GRUatten'):
                GRUatten_ckpt = file
            else:
                lgb_ckpt = file
        self.GRU_atten = GRUAttention()
        if GRUatten_ckpt is not None:
            self.GRU_atten.load_from_checkpoint(os.path.join(config.mdoel_saving_dir, GRUatten_ckpt))
            # self.GRU_atten.freeze()
        if lgb_ckpt is not None:
            self.LGB = load(os.path.join(config.mdoel_saving_dir, lgb_ckpt))
        self.fc = nn.Linear(in_features=2, out_features=1)

    def forward(self, x):
        # (64, 3)
        atten_pred = self.GRU_atten(x).squeeze(2)
        lgb_pred = torch.from_numpy(self.LGB.predict(x.cpu().squeeze())).cuda().float().unsqueeze(1)
        comined = torch.cat([atten_pred, lgb_pred], dim=1)
        out = self.fc(comined)
        return out

    def training_step(self, batch, batch_idx):
        X, y_true = batch
        y_pred = self(X)
        return mean_absolute_percentage_error(y_pred, y_true)

    def validation_step(self, batch, batch_idx):
        X, y_true = batch
        y_pred = self(X)
        y_true, y_pred = inverse_norm(y_true), inverse_norm(y_pred)
        mape = mean_absolute_percentage_error(y_pred, y_true)
        self.log(name='val_mape', value=mape, prog_bar=True)

    def test_step(self, batch, batch_idx):
        X, y_true = batch
        y_pred = self(X)
        y_true, y_pred = inverse_norm(y_true), inverse_norm(y_pred)
        mape = mean_absolute_percentage_error(y_pred, y_true)
        self.log(name='test_mape', value=mape)

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=config.atten_lr)
        return opt

    def configure_callbacks(self):
        checkpoint = ModelCheckpoint(monitor='val_mape',
                                     mode='min',
                                     dirpath='final_ckpt',
                                     filename='CominedNet-{epoch}-{step}-{val_mape:.5f}')
        return checkpoint




