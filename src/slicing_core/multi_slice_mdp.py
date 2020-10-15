import numpy as np

from state import State


# multi-slice support for different kind of slices
# TODO: support for more than two slices (look at _generate_states, fix it!)
class MultiSliceMDP:
    def __init__(self, slices):
        self._slices = slices

    def _generate_states(self):
        # see https://www.kite.com/python/answers/how-to-get-all-element-combinations-of-two-numpy-arrays-in-python
        mesh = np.array(np.meshgrid(self._slices[0], self._slices[1]))
        return mesh.T.reshape(-1, 2)


