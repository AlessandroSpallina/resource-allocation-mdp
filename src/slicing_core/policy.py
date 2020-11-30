import abc
import numpy as np
import mdptoolbox
import math
import pickle
import multiprocessing
from copy import copy

from src.slicing_core.state import SingleSliceState
from src.slicing_core.config import POLICY_CACHE_FILES_PATH


class _Cache:
    def __init__(self, config, file_extension):
        self._path = f"{POLICY_CACHE_FILES_PATH}{config.hash}.{file_extension}"

    def load(self):
        try:
            loaded = pickle.load(open(self._path, "rb"))
        except FileNotFoundError:
            loaded = None
        return loaded

    def store(self, policy):
        pickle.dump(policy, open(self._path, "wb"))
        return self._path


class Policy(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def policy(self):
        pass

    @property
    @abc.abstractmethod
    def states(self):
        pass

    def __init__(self, policy_config):
        self._config = policy_config

    @abc.abstractmethod
    def init(self):
        pass

    @abc.abstractmethod
    def calculate_policy(self):
        pass

    @abc.abstractmethod
    def get_action_from_policy(self, current_state, current_timeslot):
        pass


class CachedPolicy(Policy):
    def __init__(self, config, policy_class):
        super().__init__(config)
        self._cache = _Cache(config, policy_class(config).__class__.__name__)
        cached = self._cache.load()
        if cached is not None:
            self.obj = cached
            self._is_cached = True
        else:
            self.obj = policy_class(config)
            self._is_cached = False

    @property
    def policy(self):
        return self.obj.policy

    @property
    def states(self):
        return self.obj.states

    def init(self):
        if not self._is_cached:
            self.obj.init()

    def calculate_policy(self):
        if not self._is_cached:
            self.obj.calculate_policy()
            self._cache.store(self.obj)

    def get_action_from_policy(self, current_state, current_timeslot):
        return self.obj.get_action_from_policy(current_state, current_timeslot)


class SingleSliceMdpPolicy(Policy):
    @property
    def transition_matrix(self):
        return self._transition_matrix

    @property
    def reward_matrix(self):
        return self._reward_matrix

    @property
    def states(self):
        return self._states

    @property
    def policy(self):
        return self._policy

    def init(self):
        self._generate_states()
        self._generate_actions()
        self._generate_transition_matrix()
        self._generate_reward_matrix()

    def calculate_policy(self):
        if self._config.algorithm == 'vi':
            self._policy = list(self._run_value_iteration(self._config.discount_factor))
            # translating action id in the policy table with the real action
            for i in range(len(self._policy)):
                self._policy[i] = self._actions[self._policy[i]]

        elif self._config.algorithm == 'fh':
            policy = self._run_finite_horizon(self._config.discount_factor)
            self._policy = np.empty_like(policy, dtype=object)
            # translating action id in the policy table with the real action
            for i in range(len(self._policy)):
                for j in range(len(self._policy[i])):
                    self._policy[i][j] = self._actions[policy[i][j]]

    def get_action_from_policy(self, current_state, current_timeslot):
        try:
            if len(self._policy[0]) > 0:  # if we are here the policy is a matrix (fh)
                return self._policy[:, current_timeslot][self._states.index(current_state)]
        except TypeError:
            return self._policy[self._states.index(current_state)]

    def _generate_states(self):
        self._states = []
        for i in range(self._config.server_max_cap + 1):
            for j in range(self._config.queue_size + 1):
                self._states.append(SingleSliceState(j, i))

    def _generate_actions(self):
        self._actions = [i for i in range(self._config.server_max_cap + 1)]

    def _generate_transition_matrix(self):
        self._transition_matrix = np.zeros((self._config.server_max_cap + 1, len(self._states), len(self._states)))

        # lets iterate the trans matrix and fill with correct probabilities
        for a in range(len(self._transition_matrix)):
            for i in range(len(self._states)):
                for j in range(len(self._states)):
                    self._transition_matrix[a][i][j] = \
                        self._calculate_transition_probability(self._states[i], self._states[j], a)

    def _calculate_transition_probability(self, from_state, to_state, action_id):
        if not self._config.immediate_action:
            h_d = self._calculate_h_d(from_state)
        else:
            h_d = self._calculate_h_d(to_state)

        diff = to_state - from_state

        tmp = 0
        tmp2 = 0

        for x in range(max(0, diff.k), self._config.queue_size - from_state.k + 1):
            p_proc = 0

            try:
                p_arr = self._config.arrivals_histogram[x]
            except IndexError:
                p_arr = 0

            if from_state.k + x - to_state.k == \
                    ((from_state.k + x) if self._config.arrival_processing_phase else from_state.k):
                p_proc = sum(h_d[from_state.k + x - to_state.k:])
            elif from_state.k + x - to_state.k < \
                    ((from_state.k + x) if self._config.arrival_processing_phase else from_state.k):
                try:
                    p_proc = h_d[from_state.k + x - to_state.k]
                except IndexError:
                    pass
            else:
                p_proc = 0

            tmp += p_arr * p_proc

        for x in range(self._config.queue_size - from_state.k + 1,
                       len(self._config.arrivals_histogram) + 1):
            p_proc = 0

            try:
                p_arr = self._config.arrivals_histogram[x]
            except IndexError:
                p_arr = 0

            if self._config.queue_size - to_state.k == self._config.queue_size:
                p_proc = sum(h_d[self._config.queue_size - to_state.k:])
            elif self._config.queue_size - to_state.k < self._config.queue_size:
                try:
                    p_proc = h_d[self._config.queue_size - to_state.k]
                except IndexError:
                    pass
            else:
                p_proc = 0

            tmp2 += p_arr * p_proc

        transition_probability = tmp + tmp2

        return self._filter_transition_probability_by_action(transition_probability, to_state, action_id)

    def _calculate_h_d(self, state):
        h_d = [1.]  # default H_d value for S = 0
        if state.n > 0:
            h_d = self._config.server_capacity_histogram
            for i in range(1, state.n):
                h_d = np.convolve(h_d, self._config.server_capacity_histogram)
        return h_d

    def _filter_transition_probability_by_action(self, transition_probability, to_state, action_id):
        if to_state.n != action_id:
            return 0
        return transition_probability

    def _generate_reward_matrix(self):
        self._reward_matrix = np.zeros((self._config.server_max_cap + 1, len(self._states), len(self._states)))

        for a in range(len(self._reward_matrix)):
            for i in range(len(self._states)):
                for j in range(len(self._states)):
                    if self._transition_matrix[a][i][j] > 0:
                        self._reward_matrix[a][i][j] = self._calculate_transition_reward(self._states[j])

    def _calculate_transition_reward(self, to_state):
        # C = alpha * C_k * num of jobs + beta * C_n * num of server + gamma * C_l * E(num of lost jobs)
        cost1 = self._config.c_job * to_state.k
        cost2 = self._config.c_server * to_state.n
        cost3 = 0

        # expected value of lost packets
        for i in range(len(self._config.arrivals_histogram)):
            if to_state.k + i > self._config.queue_size:
                cost3 += self._config.arrivals_histogram[i] * i * self._config.c_lost

        return - (self._config.alpha * cost1 +
                  self._config.beta * cost2 +
                  self._config.gamma * cost3)

    def _run_value_iteration(self, discount):
        vi = mdptoolbox.mdp.ValueIteration(self._transition_matrix, self._reward_matrix, discount)
        vi.run()
        return vi.policy

    def _run_finite_horizon(self, discount):
        vi = mdptoolbox.mdp.FiniteHorizon(self._transition_matrix, self._reward_matrix,
                                          discount, self._config.timeslots)
        vi.run()
        return vi.policy


class MultiSliceMdpPolicy(SingleSliceMdpPolicy):
    def init(self):
        self._init_slices()
        self._generate_states()
        self._generate_actions()
        self._generate_transition_matrix()
        self._generate_reward_matrix()

    def _init_slices(self):
        self._slices = []

        for i in range(self._config.slice_count):
            self._slices.append(SingleSliceMdpPolicy(self._config.slice(i)))
            self._slices[-1].init()

    def _generate_states(self):
        slices_states = [s.states for s in self._slices]
        mesh = np.array(np.meshgrid(*slices_states))

        to_filter = mesh.T.reshape(-1, len(slices_states)).tolist()

        self._states = []

        for multislice_state in to_filter:
            if sum([singleslice_state.n for singleslice_state in multislice_state]) <= self._config.server_max_cap:
                self._states.append(multislice_state)

    def _generate_actions(self):
        tmp = [list(range(self._config.server_max_cap + 1))] * self._config.slice_count

        mesh = np.array(np.meshgrid(*tmp))
        to_filter = mesh.T.reshape(-1, len(tmp)).tolist()

        self._actions = []
        for i in to_filter:
            if sum(i) <= tmp[0][-1]:
                self._actions.append(i)

    def _generate_transition_matrix(self):
        self._transition_matrix = np.zeros((len(self._actions), len(self._states), len(self._states)))

        for a in range(len(self._transition_matrix)):
            for i in range(len(self._states)):
                for j in range(len(self._states)):
                    self._transition_matrix[a][i][j] = \
                        self._calculate_transition_probability(self._states[i], self._states[j], self._actions[a])

    def _calculate_transition_probability(self, from_state, to_state, action):
        transition_probability = \
            math.prod([self._slices[i].transition_matrix[action[i]]
                       [self._slices[i].states.index(from_state[i])]
                       [self._slices[i].states.index(to_state[i])] for i in range(self._config.slice_count)])
        return transition_probability

    def _generate_reward_matrix(self):
        self._reward_matrix = np.zeros((len(self._actions), len(self._states), len(self._states)))

        for a in range(len(self._reward_matrix)):
            for i in range(len(self._states)):
                for j in range(len(self._states)):
                    if self._transition_matrix[a][i][j] > 0:
                        self._reward_matrix[a][i][j] = self._calculate_transition_reward(self._states[j])

    def _calculate_transition_reward(self, to_state):
        # C = alpha * C_k * num of jobs + beta * C_n * num of server + gamma * C_l * E(num of lost jobs)
        cost1 = [self._config.slices[i].c_job * to_state[i].k for i in range(self._config.slice_count)]
        cost2 = [self._config.slices[i].c_server * to_state[i].n for i in range(self._config.slice_count)]
        cost3 = [0] * self._config.slice_count

        # expected value of lost packets
        for single_slice_index in range(len(self._slices)):
            for i in range(len(self._config.slices[single_slice_index].arrivals_histogram)):
                if to_state[single_slice_index].k + i > self._config.slices[single_slice_index].queue_size:
                    cost3[single_slice_index] += self._config.slices[single_slice_index].arrivals_histogram[i] * i * \
                                                 self._config.slices[single_slice_index].c_lost

        normalized_cost1 = [self._config.slices[i].alpha * cost1[i] for i in range(self._config.slice_count)]
        normalized_cost2 = [self._config.slices[i].beta * cost2[i] for i in range(self._config.slice_count)]
        normalized_cost3 = [self._config.slices[i].gamma * cost3[i] for i in range(self._config.slice_count)]

        return - (sum(normalized_cost1) + sum(normalized_cost2) + sum(normalized_cost3))


# target function for multiprocessing in PriorityMultiSliceMdpPolicy
def _run_subslices(slice_conf):
    subslices = []
    for i in range(slice_conf.server_max_cap + 1):
        subconf = copy(slice_conf)
        subconf.server_max_cap = i
        subslices.append(CachedPolicy(subconf, SingleSliceMdpPolicy))
        subslices[-1].init()
        subslices[-1].calculate_policy()
    return subslices


# Order matter! slice with index 0 is the highest priority ans so on..
class PriorityMultiSliceMdpPolicy(MultiSliceMdpPolicy):
    def init(self):
        self._init_slices()
        self._generate_states()
        self._generate_actions()

    def calculate_policy(self):
        self._policy = []

        for state in self._states:  # @ for each state
            servers_left = 0
            if self._config.algorithm == 'vi':
                servers_left = self._config.server_max_cap
            elif self._config.algorithm == 'fh':
                servers_left = np.array([self._config.server_max_cap] * self._config.timeslots)
            multislice_action = []
            for i in range(self._config.slice_count):  # @ for each singleslice in the multislice
                slice_action = []
                if self._config.algorithm == 'vi':
                    if state[i].n > servers_left:
                        # we are here when state[i] is a not possible state due highest priority slice allocations
                        # i.e. state (0,2) when servers_left are only 1
                        # this will never happen when the system start from [(0,0) for all slices], but we have to
                        # be robust so if for any reason the low priority slice is in a (0,2) state, the policy have to
                        # handle this
                        state[i].n = servers_left

                    slice_action = \
                        self._slices[i][servers_left].policy[self._slices[i][servers_left].states.index(state[i])]
                    servers_left -= slice_action

                elif self._config.algorithm == 'fh':
                    if state[i].n > min(servers_left):
                        # the same as above, but for finite horizon mdp algo
                        state[i].n = min(servers_left)

                    slice_action = \
                        self._slices[i][min(servers_left)].policy[self._slices[i][min(servers_left)].states.index(state[i])]

                    servers_left = servers_left - slice_action
                    slice_action = slice_action.tolist()

                multislice_action.append(slice_action)
            self._policy.append(multislice_action)

    def _init_slices(self):
        """ Preparing multiprocessing stuff """
        processes = []
        for i in range(self._config.slice_count):
            processes.append(multiprocessing.Process(target=_run_subslices, args=(self._config.slice(i),)))
            processes[-1].start()
        for process in processes:
            process.join()

        # when i am here my multiprocesses already cached slices policies, i can just pick all of these
        self._slices = [_run_subslices(self._config.slice(i)) for i in range(self._config.slice_count)]

    def _generate_states(self):
        max_servers = self._config.server_max_cap
        slices_with_maxservers = [s[max_servers] for s in self._slices]

        slices_states = [s.states for s in slices_with_maxservers]
        mesh = np.array(np.meshgrid(*slices_states))

        to_filter = mesh.T.reshape(-1, len(slices_states)).tolist()

        self._states = []

        for multislice_state in to_filter:
            if sum([singleslice_state.n for singleslice_state in multislice_state]) <= self._config.server_max_cap:
                self._states.append(multislice_state)
