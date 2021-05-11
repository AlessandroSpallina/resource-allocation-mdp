import abc
import numpy as np
import multiprocessing
from copy import copy

from src.slicing_core.environment import MultiSliceSimulator
from src.slicing_core.utils import _SimulationCache


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
    # C = alpha * C_k * num of jobs in the queue + beta * C_n * num of server + gamma * C_l * num of lost jobs + alloc...dealloc
    to_return = []
    for ts_index in range(len(environment_history)):
        ts_tmp = []
        ts_tmp2 = []
        multislice_states = environment_history[ts_index]['state']
        lost_jobs = environment_history[ts_index]['lost_jobs']
        for i in range(len(slices_paramethers)):
            ts_tmp2.append([])

            # TODO: remeber that here we don't have alpha,beta,gamma..
            cost1 = slices_paramethers[i].c_job * multislice_states[i].k
            ts_tmp2[i].append(cost1)

            cost2 = slices_paramethers[i].c_server * multislice_states[i].n
            ts_tmp2[i].append(cost2)

            cost3 = slices_paramethers[i].c_lost * lost_jobs[i]
            ts_tmp2[i].append(cost3)

            if ts_index > 0:
                previous_state = environment_history[ts_index-1]['state']
                cost4 = \
                    slices_paramethers[i].c_alloc * \
                    (0 if (multislice_states[i].n - previous_state[i].n) <= 0 else (multislice_states[i].n - previous_state[i].n))
                cost5 = \
                    slices_paramethers[i].c_dealloc * \
                    (0 if (multislice_states[i].n - previous_state[i].n) >= 0 else ((multislice_states[i].n - previous_state[i].n)*-1))
            else:
                cost4 = 0
                cost5 = 0

            ts_tmp2[i].append(cost4)
            ts_tmp2[i].append(cost5)

            ts_tmp.append(cost1 + cost2 + cost3 + cost4 + cost5)
        to_return.append(environment_history[ts_index])
        to_return[-1]['cost'] = ts_tmp
        to_return[-1]['cost_component'] = ts_tmp2
    return to_return


def _run_and_cache_simulation(policy, conf, cache):
    agent = NetworkOperator(policy, MultiSliceSimulator(conf), conf.timeslots)

    agent.start_automatic_control()
    # TODO: this add_real_costs_to_stats can be moved inside the NetworkOperator object (useful with real scen.)
    to_store = _add_real_costs_to_stats(
        agent.history,
        conf.slices)
    cache.store(to_store)


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
        self._history_std = []
        self._history_raw = []
        self._caches = []
        # self._init_agents()

    @property
    def history(self):
        return self._history

    @property
    def history_std(self):
        return self._history_std

    # @property
    # def history_raw(self):
    #     return self._history_raw

    def start_automatic_control(self):
        history_tmp = []
        processes = []
        self._caches = [_SimulationCache(self._simulation_conf, f'{i}') for i in range(self._simulation_conf.runs)]

        for index in range(self._simulation_conf.runs):

            processes.append(
                multiprocessing.Process(target=_run_and_cache_simulation,
                                        args=(self._policy, self._simulation_conf, self._caches[index]))
            )
            processes[-1].start()

        for p in processes:
            p.join()

        for cache in self._caches:
            history_tmp.append(cache.load(blocking=True))
            cache.remove()

        # self._history_raw = self._raw_history(history_tmp)
        self._history, self._history_std = self._average_history(history_tmp)

    # def _init_agents(self):
    #     self._agents = []
    #     for _ in range(self._simulation_conf.runs):
    #         self._agents.append(NetworkOperator(self._policy,
    #                                             MultiSliceSimulator(self._simulation_conf),
    #                                             self._simulation_conf.timeslots))

    # def _raw_history(self, raw):
    #     tmp_active_servers = []
    #     tmp_jobs_in_queue = []
    #     tmp_jobs_in_system = []
    #     tmp_incoming_jobs = []
    #     tmp_lost_jobs = []
    #     tmp_processed_jobs = []
    #     tmp_cost = []
    #
    #     for i in range(self._simulation_conf.runs):
    #         tmp_active_servers.append([d['active_servers'] for d in raw[i]])
    #         tmp_jobs_in_queue.append([d['jobs_in_queue'] for d in raw[i]])
    #         tmp_jobs_in_system.append([d['jobs_in_system'] for d in raw[i]])
    #         tmp_incoming_jobs.append([d['incoming_jobs'] for d in raw[i]])
    #         tmp_lost_jobs.append([d['lost_jobs'] for d in raw[i]])
    #         tmp_processed_jobs.append([d['processed_jobs'] for d in raw[i]])
    #         tmp_cost.append([d['cost'] for d in raw[i]])
    #
    #     return {
    #         "active_servers": tmp_active_servers,
    #         "jobs_in_queue": tmp_jobs_in_queue,
    #         "jobs_in_system": tmp_jobs_in_system,
    #         "incoming_jobs": tmp_incoming_jobs,
    #         "lost_jobs": tmp_lost_jobs,
    #         "processed_jobs": tmp_processed_jobs,
    #         "cost": tmp_cost,
    #     }

    # TODO: this can be written better and independent of what the environment returns
    def _average_history(self, history_to_average):
        wait_time_in_the_queue_average = []
        wait_time_in_the_system_average = []

        tmp_active_servers_average = []
        tmp_jobs_in_queue_average = []
        tmp_jobs_in_system_average = []
        tmp_incoming_jobs_average = []
        tmp_lost_jobs_average = []
        tmp_processed_jobs_average = []
        tmp_cost_average = []
        tmp_cost_component_average = []

        for i in range(self._simulation_conf.runs):
            tmp_active_servers_average.append([d['active_servers'] for d in history_to_average[i]])
            tmp_jobs_in_queue_average.append([d['jobs_in_queue'] for d in history_to_average[i]])
            tmp_jobs_in_system_average.append([d['jobs_in_system'] for d in history_to_average[i]])
            tmp_incoming_jobs_average.append([d['incoming_jobs'] for d in history_to_average[i]])
            tmp_lost_jobs_average.append([d['lost_jobs'] for d in history_to_average[i]])
            tmp_processed_jobs_average.append([d['processed_jobs'] for d in history_to_average[i]])
            tmp_cost_average.append([d['cost'] for d in history_to_average[i]])
            tmp_cost_component_average.append([d['cost_component'] for d in history_to_average[i]])

            wait_time_in_the_queue_average.append([d['wait_time_in_the_queue'] for d in history_to_average[i]])
            wait_time_in_the_system_average.append([d['wait_time_in_the_system'] for d in history_to_average[i]])

        active_servers_average = np.average(np.array(tmp_active_servers_average), axis=0).tolist()
        jobs_in_queue_average = np.average(np.array(tmp_jobs_in_queue_average), axis=0).tolist()
        jobs_in_system_average = np.average(np.array(tmp_jobs_in_system_average), axis=0).tolist()
        incoming_jobs_average = np.average(np.array(tmp_incoming_jobs_average), axis=0).tolist()
        lost_jobs_average = np.average(np.array(tmp_lost_jobs_average), axis=0).tolist()
        processed_jobs_average = np.average(np.array(tmp_processed_jobs_average), axis=0).tolist()
        cost_average = np.average(np.array(tmp_cost_average), axis=0).tolist()
        cost_component_average = np.average(np.array(tmp_cost_component_average), axis=0).tolist()

        active_servers_average_std = np.std(np.array(tmp_active_servers_average), axis=0).tolist()
        jobs_in_queue_average_std = np.std(np.array(tmp_jobs_in_queue_average), axis=0).tolist()
        jobs_in_system_average_std = np.std(np.array(tmp_jobs_in_system_average), axis=0).tolist()
        incoming_jobs_average_std = np.std(np.array(tmp_incoming_jobs_average), axis=0).tolist()
        lost_jobs_average_std = np.std(np.array(tmp_lost_jobs_average), axis=0).tolist()
        processed_jobs_average_std = np.std(np.array(tmp_processed_jobs_average), axis=0).tolist()
        cost_average_std = np.std(np.array(tmp_cost_average), axis=0).tolist()
        cost_component_std = np.std(np.array(tmp_cost_component_average), axis=0).tolist()

        # timing statistics need before to be processed in order to have histograms
        wait_time_in_the_queue_average = \
            np.average(
                [self._histogram_from_feature(sim, (np.array(wait_time_in_the_queue_average, dtype=object).max()[0] + 1))
                 for sim in wait_time_in_the_queue_average], axis=0)
        wait_time_in_the_system_average = \
            np.average(
                [self._histogram_from_feature(sim, (np.array(wait_time_in_the_system_average, dtype=object).max()[0] + 1))
                 for sim in wait_time_in_the_system_average], axis=0)

        return ({
            "active_servers": active_servers_average,
            "jobs_in_queue": jobs_in_queue_average,
            "jobs_in_system": jobs_in_system_average,
            "incoming_jobs": incoming_jobs_average,
            "lost_jobs": lost_jobs_average,
            "processed_jobs": processed_jobs_average,
            "cost": cost_average,
            "cost_component": cost_component_average,
            "wait_time_in_the_queue": wait_time_in_the_queue_average,
            "wait_time_in_the_system": wait_time_in_the_system_average
        }, {
            "active_servers": active_servers_average_std,
            "jobs_in_queue": jobs_in_queue_average_std,
            "jobs_in_system": jobs_in_system_average_std,
            "incoming_jobs": incoming_jobs_average_std,
            "lost_jobs": lost_jobs_average_std,
            "processed_jobs": processed_jobs_average_std,
            "cost": cost_average_std,
            "cost_component": cost_component_std,
        })

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













