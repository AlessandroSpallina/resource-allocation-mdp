import numpy as np


class Agent(object):
    # Random policy agent, selects a direction at random then moves
    def __init__(self, grid):
        self.grid = grid

    def getAction(self, state, i):  # Random action selector
        return np.random.randint(1, 5)

    def learn(self, experienceTuples):
        return
