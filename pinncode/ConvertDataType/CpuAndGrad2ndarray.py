import torch
import numpy as np

def cpu_and_grad2ndarray(data):
    data_ndarray = dict(data)
    for key, value in data_ndarray.items():
        data_ndarray[key] = value.detach().cpu().numpy()
    return data_ndarray