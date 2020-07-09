class State(object):
    """
    The state class which is used to construct a state object which in this instance represents a
    position in the grid defined by its x and y co-ordinates.
    The class contains information such
    self.x: (int)The x co-ordinate of the state on the grid
    self.y: (int)The y co-ordinate of the state on the grid
    self.loc: (tuple) the actual position of the state on the grid (x,y)
    self.reward: (float) Reward associated with the state
    self.isReachable: (boolean) True if the state is reachable. Unreachable states are blocked by obstacles
    self.isTerminalState: (boolean) True if the the goal location or the fire location
    self.isStartLocation: (boolean) True if at the initial position
    self.freeLocs: (tuple) representing positions not to be blocked such as right and down from the starting
    point to allow initial movement
    """

    def __init__(self, x, y, reward=-1, is_reachable=True, is_terminal_state=False):
        self.x = x
        self.y = y
        self.loc = (x, y)
        self.reward = reward
        self.is_reachable = is_reachable
        self.is_terminal_state = is_terminal_state
        self.is_start_location = self.is_start_loc()
        self.free_locs = [(0, 1), (1, 0)]

    def blockable(self):  # Return true if blockable. Terminal and freeLocs are not blockable
        #return self.playable() and not self.loc in self.free_locs
        return self.playable() and self.loc not in self.free_locs

    def playable(self):  # playable states are Reachable non terminal states
        return self.is_reachable and not self.is_terminal_state

    def is_start_loc(self):  # Return true if at the starting location
        if self.loc == (0, 0):
            return True
        return False

    def set_reward(self, reward):  # To set reward associated with each state
        self.reward = reward

    def is_accessible(self):  # Return true if the state is reachable
        return self.is_reachable

    def block(self):  # Add a block to a particular state
        self.is_reachable = False

    def get_reward(self):  # Return reward for state
        return self.reward

    def set_as_terminal(self):  # Set a particular state as terminal
        self.is_terminal_state = True

    def is_terminal(self):  # Return True if state is terminal
        return self.is_terminal_state
