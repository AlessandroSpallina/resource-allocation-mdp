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


# TODO: this NetworkOperator needs a refactoring in order to be usable in a real scenario (i.e. timeslots, costs)
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


def _add_real_costs_to_stats(environment_history, slices_paramethers):
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


# TODO: This class does processing (i.e. wait time in the system/queue), this should not stay here in the future
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
            # TODO: this add_real_costs_to_stats can be moved inside the NetworkOperator object (useful with real scen.)
            history_tmp.append(_add_real_costs_to_stats(agent.history, self._simulation_conf.slices))

        self._history = self._average_history(history_tmp)

    def _init_agents(self):
        self._agents = []
        for _ in range(self._simulation_conf.runs):
            self._agents.append(NetworkOperator(self._policy,
                                                MultiSliceSimulator(self._simulation_conf),
                                                self._simulation_conf.timeslots))

    # TODO: this can be written better and independent of what the environment returns
    def _average_history(self, history_to_average):
        active_servers_average = []
        jobs_in_queue_average = []
        incoming_jobs_average = []
        lost_jobs_average = []
        processed_jobs_average = []
        wait_time_in_the_queue_average = []
        wait_time_in_the_system_average = []
        cost_average = []

        for i in range(self._simulation_conf.runs):
            active_servers_average.append([d['active_servers'] for d in history_to_average[i]])
            jobs_in_queue_average.append([d['jobs_in_queue'] for d in history_to_average[i]])
            incoming_jobs_average.append([d['incoming_jobs'] for d in history_to_average[i]])
            lost_jobs_average.append([d['lost_jobs'] for d in history_to_average[i]])
            processed_jobs_average.append([d['processed_jobs'] for d in history_to_average[i]])
            wait_time_in_the_queue_average.append([d['wait_time_in_the_queue'] for d in history_to_average[i]])
            wait_time_in_the_system_average.append([d['wait_time_in_the_system'] for d in history_to_average[i]])
            cost_average.append([d['cost'] for d in history_to_average[i]])

        active_servers_average = np.average(np.array(active_servers_average), axis=0).tolist()
        jobs_in_queue_average = np.average(np.array(jobs_in_queue_average), axis=0).tolist()
        incoming_jobs_average = np.average(np.array(incoming_jobs_average), axis=0).tolist()
        lost_jobs_average = np.average(np.array(lost_jobs_average), axis=0).tolist()
        processed_jobs_average = np.average(np.array(processed_jobs_average), axis=0).tolist()
        cost_average = np.average(np.array(cost_average), axis=0).tolist()

        # timing statistics need before to be processed in order to have histograms
        wait_time_in_the_queue_average = \
            np.average(
                [self._histogram_from_feature(sim, (np.array(wait_time_in_the_queue_average, dtype=object).max()[0] + 1))
                 for sim in wait_time_in_the_queue_average], axis=0)
        wait_time_in_the_system_average = \
            np.average(
                [self._histogram_from_feature(sim, (np.array(wait_time_in_the_system_average, dtype=object).max()[0] + 1))
                 for sim in wait_time_in_the_system_average], axis=0)

        return {
            "active_servers": active_servers_average,
            "jobs_in_queue": jobs_in_queue_average,
            "incoming_jobs": incoming_jobs_average,
            "lost_jobs": lost_jobs_average,
            "processed_jobs": processed_jobs_average,
            "cost": cost_average,
            "wait_time_in_the_queue": wait_time_in_the_queue_average,
            "wait_time_in_the_system": wait_time_in_the_system_average
        }

    def _histogram_from_feature(self, feature, output_dimension):
        # splitting feature per subslice
        feature_per_subslice = [list() for _ in range(self._simulation_conf.slice_count)]

        histogram = np.zeros((self._simulation_conf.slice_count, output_dimension))
        for elem in feature:  # elem contains a time-slot timing rilevation
            for i in range(len(elem)):
                if len(elem[i]) > 0:
                    for e in elem[i]:
                        feature_per_subslice[i].append(e)

        # counting the occurrencies of times
        # i.e. [999, 90, 1] means 999 jobs in 0 ts, 90 in 1 ts and so on
        for s in range(len(feature_per_subslice)):
            if len(feature_per_subslice[s]) > 0:
                for i in range(max(feature_per_subslice[s]) + 1):
                    histogram[s][i] = feature_per_subslice[s].count(i)

        # and now calculate the percentage!
        for s in range(len(histogram)):
            if len(feature_per_subslice[s]) > 0:
                slice_sum = histogram[s].sum()
                for i in range(len(histogram[s])):
                    histogram[s][i] /= slice_sum

        return histogram.T













