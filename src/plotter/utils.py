import jsons
import numpy as np


def import_data(path):
    with open(path, 'r') as f:
        to_return = f.read()
    to_return = jsons.loads(to_return)
    return to_return


def moving_average(data, average_window):
    averaged = np.convolve(data, np.ones((average_window,))/average_window, mode='valid')
    return np.arange(len(data), step=len(data)/averaged.size), averaged
