from slice_simulator import ServerMinCapError, ServerMaxCapError


class Agent:
    def __init__(self, states, policy, environment, verbose=False):
        self._verbose = verbose
        self._states = states
        self._policy = policy
        self._environment = environment

    def control_environment(self):
        action = 0
        for i in range(self._environment.simulation_time):
            try:
                current_state = self._environment.simulate_timeslot(action)
            except ServerMinCapError:
                if self._verbose:
                    print(f"Agent {self._policy} tried to deallocate in an invalid state")
            except ServerMaxCapError:
                if self._verbose:
                    print(f"Agent {self._policy} tried to allocate in an invalid state")
            action = self._policy[self._states.index(current_state)]

        return self._environment.get_statistics()
