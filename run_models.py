import os
import torch
import matplotlib.pyplot as plt
from datasets.elec_dataset import ElectricityDataModule
from models.normal import NormalGRU
from models.GRU_atten import GRUAttention
import pytorch_lightning as pl
from lightgbm import LGBMRegressor
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
from joblib import dump
from utils.utils import get_array_data, inverse_norm
from config import config


def run_normal_GRU():
    dm = ElectricityDataModule()
    normal_GRU = NormalGRU()

    trainer = pl.Trainer(max_epochs=config.max_epochs,
                         min_epochs=config.min_epochs,
                         logger=[TensorBoardLogger("../lightning_logs", name="NormalGRU")],
                         gpus=1)
    trainer.fit(normal_GRU, datamodule=dm)
    test_results = trainer.test(normal_GRU, datamodule=dm, verbose=False)
    return test_results[0]['test_mape']


def run_shallow_model(model: str, show:bool=False):
    train_arr, val_arr, test_arr = get_array_data(decom_method=config.decom_method, dim=2)
    X_train, y_train, X_val, y_val = train_arr[:, :-1], train_arr[:, -1], val_arr[:, :-1], val_arr[:, -1]
    X_test, y_test = test_arr[:, :-1], inverse_norm(test_arr[:, -1])
    if model == 'lightgbm':
        regressor = LGBMRegressor()
    elif model == 'knn':
        regressor = KNeighborsRegressor()
    elif model == 'rf':
        regressor = RandomForestRegressor()
    else:
        raise ValueError(f"No {model} implementation!")

    regressor.fit(X_train, y_train)
    y_pred = inverse_norm(regressor.predict(X_test))
    if show:
        plt.plot(range(240), y_test[:240], label='y_true')
        plt.plot(range(240), y_pred[:240], label='y_pred')
        plt.title(model)
        plt.show()
    dump(regressor, os.path.join(config.mdoel_saving_dir, model+'.joblib'))
    return mean_absolute_percentage_error(y_test, y_pred)


def run_GRU_atten():
    dm = ElectricityDataModule()
    net = GRUAttention()

    def initialize_weights(x):
        if hasattr(x, 'weight') and x.weight.dim() > 1:
            torch.nn.init.xavier_uniform_(x.weight.data)

    net.apply(initialize_weights)
    trainer = pl.Trainer(max_epochs=config.max_epochs,
                         min_epochs=config.min_epochs,
                         logger=[TensorBoardLogger("lightning_logs", name="GRUatten")],
                         gpus=1)
    trainer.fit(net, datamodule=dm)
    test_results = trainer.test(net, datamodule=dm, verbose=False)
    return test_results[0]['test_mape']



# print(f"{config.decom_method}: (MAPE) "
#       f"GRU: {run_normal_GRU()}, "
#       f"LGB: {run_shallow_model('lightgbm')}, "
#       f"KNN: {run_shallow_model('knn')}, "
#       f"RF: {run_shallow_model('rf')}")

