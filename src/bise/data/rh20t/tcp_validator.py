import numpy as np


def inspect_tcp_base(file_path: str):
    data = np.load(file_path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.shape == ():
        data = data.item()
    return data
