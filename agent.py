class Agent:
    def __init__(self, states, policy, environment):
        self._states = states
        self._policy = policy
        self._environment = environment

    def control_environment(self):
        action = 0
        for i in range(self._environment.simulation_time):
            current_state = self._environment.simulate_timeslot(action)
            action = self._policy[self._states.index(current_state)]

        return self._environment.get_statistics()
