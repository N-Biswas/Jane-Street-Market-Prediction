import pickle
import numpy as np
import pandas as pd
from numba import njit

@njit
def fast_fillna(array, values):
    if np.isnan(array.sum()):
        array = np.where(np.isnan(array), values, array)
    return array


def save_pickle(dic, save_path):
    with open(save_path, 'wb') as f:
        # with gzip.open(save_path, 'wb') as f:
        pickle.dump(dic, f)


def load_pickle(load_path):
    with open(load_path, 'rb') as f:
        # with gzip.open(load_path, 'rb') as f:
        message_dict = pickle.load(f)
    return message_dict


