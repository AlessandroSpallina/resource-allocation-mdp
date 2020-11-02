from copy import copy


class MultiSliceSimulator:
    def __init__(self, simulations, actions):
        self._simulations = simulations
        self._actions = actions

    @property
    def incoming_jobs(self):
        return [self._simulations[0].incoming_jobs, self._simulations[1].incoming_jobs]

    @property
    def simulation_time(self):
        return self._simulations[0].simulation_time

    def simulate_timeslot(self, action_id):
        action = self._actions[action_id]

        state_0 = self._simulations[0].simulate_timeslot(action[0])
        state_1 = self._simulations[1].simulate_timeslot(action[1])

        return [state_0, state_1]

    def get_statistics(self):
        stat_0 = self._simulations[0].get_statistics()
        stat_1 = self._simulations[1].get_statistics()

        return [stat_0, stat_1]
