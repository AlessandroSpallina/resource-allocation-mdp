import abc
import numpy as np

from src.slicing_core.environment import MultiSliceSimulator


class Agent:
    @property
    @abc.abstractmethod
    def history(self):
        pass

    @abc.abstractmethod
    def start_automatic_control(self):
        pass


# TODO: this NetworkOperator needs a little refactoring in order to be usable in a real scenario (i.e. timeslots, costs)
class NetworkOperator(Agent):
    def __init__(self, policy, environment, control_timeslot_duration):
        self._policy = policy
        self._environment = environment
        self._control_timeslot_duration = control_timeslot_duration

        self._current_timeslot = 0
        self._current_state = self._environment.current_state

        # statistics
        self._history = []

    @property
    def history(self):
        return self._history

    def start_automatic_control(self):
        for i in range(self._control_timeslot_duration):
            self._current_timeslot = i
            self._current_state = self._environment.current_state
            action_to_do = self._policy.get_action_from_policy(self._current_state, self._current_timeslot)
            self._history.append(self._environment.next_timeslot(action_to_do))


def add_real_costs_to_stats(environment_history, slices_paramethers):
    # C = alpha * C_k * num of jobs in the queue + beta * C_n * num of server + gamma * C_l * num of lost jobs
    to_return = []
    for ts in environment_history:
        ts_tmp = []
        multislice_states = ts['state']
        lost_jobs = ts['lost_jobs']
        for i in range(len(slices_paramethers)):
            cost1 = slices_paramethers[i].alpha * slices_paramethers[i].c_job * multislice_states[i].k
            cost2 = slices_paramethers[i].beta * slices_paramethers[i].c_server * multislice_states[i].n
            cost3 = slices_paramethers[i].gamma * slices_paramethers[i].c_lost * lost_jobs[i]
            ts_tmp.append(cost1 + cost2 + cost3)
        to_return.append(ts)
        to_return[-1]['cost'] = ts_tmp
    return to_return


class NetworkOperatorSimulator(Agent):
    """
    NetworkOperatorSimulator is a NetworkOperator with the capability to run several simulations
    with the same simulation config, this is done in order to better extract statistics for
    the policy evaluation avoiding misunderstanding due to lucky/unlucky inside the simulator.
    The history property contain the average between all simulations done.
    """
    def __init__(self, policy, simulation_config):
        self._policy = policy
        self._simulation_conf = simulation_config
        self._history = []
        self._init_agents()

    @property
    def history(self):
        return self._history

    def start_automatic_control(self):
        history_tmp = []
        for agent in self._agents:
            agent.start_automatic_control()
            history_tmp.append(add_real_costs_to_stats(agent.history, self._simulation_conf.slices))

        self._history = history_tmp

        # self._history = self._averaged_history()

    def _init_agents(self):
        self._agents = []
        for i in range(self._simulation_conf.runs):
            self._agents.append(NetworkOperator(self._policy,
                                                MultiSliceSimulator(self._simulation_conf),
                                                self._simulation_conf.timeslots))

    # def _average_history(self, history_to_average):
    #     to_ret = []
    #
    #     lost_jobs_to_average = []
    #     processed_jobs_






