from state import State


class Agent:
    def __init__(self, states, policy, environment):
        self._states = states
        self._policy = policy
        self._environment = environment

        # useful for the first timeslot of simulation
        self._current_timeslot = 0
        self._current_state = State(0, 0)
        self._action = self._get_action()
        self._current_state = self._environment.simulate_timeslot(self._action)

    def _get_action(self):
        try:
            if len(self._policy[0]) > 0:  # if we are here the policy is a matrix (fh)
                return self._policy[:, self._current_timeslot][self._states.index(self._current_state)]
        except TypeError:
            return self._policy[self._states.index(self._current_state)]

    def control_environment(self):
        for i in range(1, self._environment.simulation_time):
            self._current_timeslot = i
            self._action = self._get_action()
            self._current_state = self._environment.simulate_timeslot(self._action)

        return self._environment.get_statistics()
