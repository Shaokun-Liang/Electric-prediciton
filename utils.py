import os
import numpy as np
import torch
from typing import Literal, Union
from config import config


def get_array_data(decom_method: str = Literal['prophet+', 'prophet', None], dim: int = Literal[2, 3]):
    convert = {'prophet+': 'data_cleaned_decomposed', None: 'lag_j=3', 'prophet': 'data_cleaned_prophet_minmax'}
    data_dir = convert[decom_method]
    if dim == 2:
        data_dir += '_2D'
    train_dir = os.path.join(config.data_dir, data_dir, 'train_dataset.npy')
    val_dir = os.path.join(config.data_dir, data_dir, 'validation_dataset.npy')
    test_dir = os.path.join(config.data_dir, data_dir, 'test_dataset.npy')
    train_arr, val_arr, test_arr = np.load(train_dir), np.load(val_dir), np.load(test_dir)
    np.random.shuffle(train_arr)
    return train_arr, val_arr, test_arr


def inverse_norm(data: Union[torch.Tensor, np.ndarray]):
    if config.decom_method is None:
        return data * (24739 - 9581) + 9581
    elif config.decom_method == 'prophet+':
        if isinstance(data, torch.Tensor):
            return torch.exp(data)
        elif isinstance(data, np.ndarray):
            return np.exp(data)
        else:
            raise TypeError(f"Input data can only be tensor or array, got {type(data)}.")
    else:
        return data * (24739 - 9581) + 9581
