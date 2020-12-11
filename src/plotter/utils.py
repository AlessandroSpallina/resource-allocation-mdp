import jsons
import numpy as np


def import_data(path):
    with open(path, 'r') as f:
        to_return = f.read()
    to_return = jsons.loads(to_return)
    return to_return
