import numpy as np

from res.old.state import State

"""
Design and return the grid world
                parameters:
                xDim: (int)Height of the gridworld
                yDIm: (int)width of the grid world
                numBlocks: (int) number of arbitrarily set blocks in the grid
                reward: (float) reward for all normal states in the block. Initially set for all states
                special states can then be modified with state.setReward(reward)
"""


def design_grid_world(x_dim, y_dim, num_blocks=3, reward=-1):  # Design the grid world given its length and width
    """
    Create an ndArray of state objects of size [xDim, yDim]
    """
    grid_world = np.array([[State(i, j, reward) for j in range(y_dim)] for i in range(x_dim)], dtype=object)
    print("Grid world created")
    grid_world[-1, 0].setAsTerminal()  # set The grid at the bottom left as terminal
    print("Goal Location set as [{},{}]".format(grid_world[-1, 0].x, grid_world[-1, 0].y))
    grid_world[-1, 0].reward = 5  # Goal location
    grid_world[-1, -1].reward = -3  # Fire location, Also terminal
    grid_world[-1, -1].setAsTerminal()
    print("Fire Location set as [{},{}]".format(grid_world[-1, -1].x, grid_world[-1, -1].y))
    count = 0
    while True:
        """Set numBlocks number of blocks arbitrarily in the grid space. 
        the state.blockable() property of the state helps to avoid already set blocks, terminal blocks
        starting blocks and the immediately viable position from the starting position
        """
        locx = np.random.randint(x_dim)
        locy = np.random.randint(y_dim)
        if not grid_world[locx, locy].blockable():
            continue
        grid_world[locx, locy].block()
        count += 1
        print("Block number {} created at [{},{}]".format(count, locx, locy))

        if count == num_blocks:
            break
    return grid_world

    # grid_world[]


# grid = designGridWorld(3,3,3)
# grid[2,2].isTerminal()


class Environment(object):
    """
    The environment is the interface between the agent and the grid world, it creates and consists of the grid
    world as well as all underlying transition rules between states. It keeps track of the present state
    receives actions and generates feedback to the agent and controls transition between states
    properties:
        self.xDim, yDim, numBlocks ==> See designGridWorld function
        self.transitionProb: (float) between 0 and 1. defines the probability of moving to the desired
        location. It introduces stochasticity to the environment where the same action could produce
        different reactions from the environment
        self.initState: (state) the starting position (0,0)
        self.actionDict: (dictionary) of all actions


    """

    def __init__(self, xDim, yDim, numBlocks, transitProb):
        self.xDim = xDim
        self.yDim = yDim
        self.numBlocks = numBlocks
        self.transitProb = transitProb
        self.grid = design_grid_world(self.xDim, self.yDim, self.numBlocks)
        self.initState = self.grid[0, 0]
        self.state = self.initState
        self.reward = 0
        self.action_dict = {0: "remained in place", 1: "Moved up", 2: "Moved down", 3: "Moved left", 4: "Moved right "}

    def goalAchieved(self):  # returns whether the goal has been reached
        return self.state == self.grid[-1, 0]

    def move(self, action):  # The movement produced by an action.
        # The new transition is controlled by this parameter and it introduces uncertainity to the movement
        rand = np.random.rand()
        if rand <= self.transitProb:
            return action
        else:
            return np.random.randint(5)

    def reset(self):  # Restart and set to the intial State
        self.state = self.initState
        print("Grid world reset")
        print("Position: ({}, {})".format(self.state.x, self.state.y))

    def nextStep(self, action):  # The Rules following the agents selection of an action
        action = self.move(action)  # The environment returns a stochastic map from the action to the movement
        if action == 0:
            self.nextState = self.state  # Remain in place
        if action == 1:  # Move up if not at the top of the grid, else remain in place
            if self.state.x == 0:
                action = 0
                self.nextState = self.state
            else:
                self.nextState = self.grid[self.state.x - 1, self.state.y]

        elif action == 2:  # Go down if not at the bottom , remain in place otherwise
            if self.state.x == self.xDim - 1:
                print("bottom")
                action = 0
                self.nextState = self.state
            else:
                self.nextState = self.grid[self.state.x + 1, self.state.y]

        elif action == 3:  # If at the left border, remain in place, otherwise move left
            if self.state.y == 0:
                action = 0
                self.nextState = self.state
            else:
                self.nextState = self.grid[self.state.x, self.state.y - 1]

        elif action == 4:  # If at the right border, remain in place, otherwise move right
            if self.state.y == self.yDim - 1:
                action = 0
                self.nextState = self.state
            else:
                self.nextState = self.grid[self.state.x, self.state.y + 1]
        if not self.nextState.isReachable:  # If the chosen state is blocked, remain in place
            action = 0
            print("oops, you hit an obstacle")
            self.nextState = self.state
        self.state = self.nextState  # The next state becomes the present state
        print(self.action_dict[action])
        print("New position: ({}, {})".format(self.state.x, self.state.y))
        return self.state
