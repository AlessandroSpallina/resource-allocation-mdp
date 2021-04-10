import abc
import numpy as np
import mdptoolbox
import math
import multiprocessing
from copy import copy

from src.slicing_core.state import SingleSliceState
from src.slicing_core.utils import _Cache

import src.slicing_core.src.slicing_core.cpolicy as cpolicy


class Policy(metaclass=abc.ABCMeta):
    def __init__(self, policy_config):
        self._config = policy_config
        self._policy = []
        self._states = []

    @property
    @abc.abstractmethod
    def policy(self):
        pass

    @property
    @abc.abstractmethod
    def states(self):
        pass

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

    @property
    def json_states(self):
        to_ret = []
        for s in self.obj.states:
            to_ret.append([s_i.json for s_i in s])
        return to_ret

    def init(self, partial_initialization=False):
        if not self._is_cached:
            if partial_initialization:
                self.obj.init(partial_initialization)
            else:
                self.obj.init()

    def calculate_policy(self):
        if not self._is_cached:
            self.obj.calculate_policy()
            self._cache.store(self.obj)

    def get_action_from_policy(self, current_state, current_timeslot):
        return self.obj.get_action_from_policy(current_state, current_timeslot)


# outdated, use cpolicy.FastInitSingleSliceMdpPolicy instead!
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

        if self._config.algorithm == 'rvi':
            self._policy = list(self._run_relative_value_iteration())
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

            self._policy = self._policy.tolist()

    def get_action_from_policy(self, current_state, current_timeslot):
        for i in range(len(current_state)):
            current_state[i].k = int(current_state[i].k / self._config.queue_scaling) * self._config.queue_scaling

        if self._config.algorithm == 'fh':
            return self._policy[self._states.index(current_state)][current_timeslot]
        else:
            return self._policy[self._states.index(current_state)]

    def _generate_states(self):
        self._states = []
        for i in range(self._config.server_max_cap + 1):
            for j in range(0, self._config.queue_size + 1, self._config.queue_scaling):
                self._states.append(SingleSliceState(j, i))

    def _generate_actions(self):
        self._actions = [i for i in range(self._config.server_max_cap + 1)]

    def _generate_transition_matrix(self):
        self._transition_matrix = np.zeros((self._config.server_max_cap + 1, len(self._states), len(self._states)))

        # lets iterate the trans matrix and fill with correct probabilities
        for a in range(len(self._transition_matrix)):
            for i in range(len(self._states)):
                for j in range(len(self._states)):

                    if self._config.queue_scaling > 1:
                        # example: if queue_scaling is 5:
                        #   Pr[(0,X)->(5,X)] is the same as without the scaling
                        #   but Pr[(0,X)->(0,X)] is bigger because of the sum of the prob. of the hidden states
                        #   Pr[(0,X)->(0,X)] = Pr[(0,X)->(0,X)] + Pr[(0,X)->(1,X)] + Pr[(0,X)->(2,X)] ... etc

                        for i_from in range(self._config.queue_scaling):
                            hidden_from = copy(self._states[i])
                            hidden_from.k += i_from

                            for j_to in range(self._config.queue_scaling):
                                hidden_to = copy(self._states[j])
                                hidden_to.k += j_to
                                if hidden_to.k <= self._config.queue_size:
                                    prob_tmp = self._calculate_transition_probability(hidden_from, hidden_to, a)
                                    self._transition_matrix[a][i][j] += prob_tmp
                    else:
                        self._transition_matrix[a][i][j] = \
                            self._calculate_transition_probability(self._states[i], self._states[j], a)

                self._transition_matrix[a][i] /= self._transition_matrix[a][i].sum()

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
            for _ in range(1, state.n):
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
                        if self._config.queue_scaling > 1:
                            # is the same as _generate_transition_matrix but for the reward!
                            hidden_to = copy(self._states[j])
                            for j_to in range(self._config.queue_scaling):
                                hidden_to.k += j_to
                                if hidden_to.k <= self._config.queue_size:
                                    rew_tmp = self._calculate_transition_reward(hidden_to)
                                    self._reward_matrix[a][i][j] += rew_tmp
                        else:
                            self._reward_matrix[a][i][j] = self._calculate_transition_reward(self._states[j])

        if self._config.normalize_reward_matrix:
            # normalize the reward matrix
            min_value = - self._reward_matrix.min()
            self._reward_matrix /= min_value

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

    def _run_relative_value_iteration(self):
        rvi = mdptoolbox.mdp.RelativeValueIteration(self._transition_matrix, self._reward_matrix)
        rvi.run()
        return rvi.policy

    def _run_value_iteration(self, discount):
        vi = mdptoolbox.mdp.ValueIteration(self._transition_matrix, self._reward_matrix, discount)
        vi.run()
        return vi.policy

    def _run_finite_horizon(self, discount):
        fh = mdptoolbox.mdp.FiniteHorizon(self._transition_matrix, self._reward_matrix,
                                          discount, self._config.timeslots)
        fh.run()
        return fh.policy


class MultiSliceMdpPolicy(cpolicy.SingleSliceMdpPolicy):
    def init(self):
        self._init_slices()
        self._generate_states(self._slices)
        self._generate_actions()
        self._generate_transition_matrix()
        self._generate_reward_matrix()

    def _init_slices(self):
        self._slices = []

        for i in range(self._config.slice_count):
            self._slices.append(SingleSliceMdpPolicy(self._config.slice(i)))
            self._slices[-1].init()

    def _generate_states(self, from_slices=[]):
        slices_states = [s.states for s in from_slices]
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


def _get_subconfs_from_singleslice(singleslice_conf):
    subconfs = {}
    for i in range(singleslice_conf.server_max_cap + 1):
        subconf = copy(singleslice_conf)
        subconf.server_max_cap = i
        subconfs[i] = subconf
    return subconfs


def _get_range_subconfs_from_singleslice(singleslice_conf, start, stop):
    subconfs = {}
    for i in range(start, stop + 1):
        subconf = copy(singleslice_conf)
        subconf.server_max_cap = i
        subconfs[i] = subconf
    return subconfs


def _get_balanced_confs(confs_dict):
    keys = list(confs_dict.keys())
    keys.sort()
    balanced = []
    if len(keys) % 2 > 0:  # if odd
        balanced.append([confs_dict[keys[-1]]])
        keys.remove(keys[-1])
    for i in range(int(len(keys) / 2)):
        balanced.append([confs_dict[i], confs_dict[max(keys) - i]])
    return balanced


# target function for multiprocessing
def _run_singleslice_from_confs(slice_index, slice_confs, partial_initialization):
    for slice_conf in slice_confs:
        p = CachedPolicy(slice_conf, cpolicy.FastInitSingleSliceMdpPolicy)
        p.init(partial_initialization)
        p.calculate_policy()
        print(f"Policy of slice-{slice_index} with {slice_conf.server_max_cap} servers done", flush=True)


def _run_subslices(slice_conf):
    subslices = []
    for i in range(slice_conf.server_max_cap + 1):
        subconf = copy(slice_conf)
        subconf.server_max_cap = i
        subslices.append(CachedPolicy(subconf, cpolicy.FastInitSingleSliceMdpPolicy))
        subslices[-1].init()
        subslices[-1].calculate_policy()
    return subslices


def _run_range_subslices(slice_conf, start, stop):
    subslices = []
    for i in range(start, stop + 1):
        subconf = copy(slice_conf)
        subconf.server_max_cap = i
        subslices.append(CachedPolicy(subconf, cpolicy.FastInitSingleSliceMdpPolicy))
        subslices[-1].init()
        subslices[-1].calculate_policy()
    return subslices


# Order matter! slice with index 0 is the highest priority ans so on..
class PriorityMultiSliceMdpPolicy(MultiSliceMdpPolicy):
    def init(self):
        self._init_slices()
        self._generate_states([s[self._config.server_max_cap] for s in self._slices])
        self._generate_actions()

    def calculate_policy(self):
        self._policy = []

        for state in self._states:  # @ for each state
            multislice_action = []

            if self._config.algorithm == 'fh':
                servers_left = np.array([self._config.server_max_cap] * self._config.timeslots)
                for i in range(self._config.slice_count):  # @ for each singleslice in the multislice
                    i_th_state = copy(state[i])
                    if i_th_state.n > min(servers_left):
                        # we are here when state[i] is a not possible state due highest priority slice allocations
                        # i.e. state (0,2) when servers_left are only 1
                        # this will never happen when the system start from [(0,0) for all slices], but we have to
                        # be robust so if for any reason the low priority slice is in a (0,2) state, the policy have to
                        # handle this

                        i_th_state.n = min(servers_left)

                    slice_action = []
                    for j in range(len(servers_left)):
                        slice_action.append\
                            (self._slices[i][servers_left[j]].policy[
                                 self._slices[i][servers_left[j]].states.index(i_th_state)][j])

                    servers_left = servers_left - slice_action
                    multislice_action.append(slice_action)

                self._policy.append(np.column_stack(multislice_action).tolist())

            else:
                servers_left = self._config.server_max_cap
                for i in range(self._config.slice_count):  # @ for each singleslice in the multislice
                    i_th_state = copy(state[i])
                    if i_th_state.n > servers_left:
                        # the same as above, but for finite horizon mdp algo
                        i_th_state.n = servers_left

                    slice_action = \
                        self._slices[i][servers_left].policy[self._slices[i][servers_left].states.index(i_th_state)]
                    servers_left -= slice_action
                    multislice_action.append(slice_action)

                self._policy.append(multislice_action)

    def _init_slices(self):  # THIS HAVE MORE PARALLELISM, BUT SEEMS WORSE
        """ Preparing multiprocessing stuff """
        processes = []

        for i in range(self._config.slice_count):
            #subconfs = _get_balanced_confs(_get_subconfs_from_singleslice(self._config.slice(i)))
            subconds_dicts = _get_subconfs_from_singleslice(self._config.slice(i))
            subconfs = [[subconds_dicts[e]] for e in subconds_dicts]

            for s in subconfs:
                processes.append(multiprocessing.Process(target=_run_singleslice_from_confs, args=(i, s,)))
                processes[-1].start()

        for process in processes:
            process.join()

        # when i am here my multiprocesses already cached slices policies, i can just pick all of these
        self._slices = [_run_subslices(self._config.slice(i)) for i in range(self._config.slice_count)]


class SimplifiedPriorityMultiSliceMdpPolicy(MultiSliceMdpPolicy):
    def init(self):
        self._init_slices()
        self._generate_states(self._slices)
        self._generate_actions()

    def _init_slices(self):
        self._slices = []
        servers_left = self._config.server_max_cap

        for s_i in range(self._config.slice_count):
            conf = self._config.slice(s_i)
            conf.server_max_cap = servers_left
            self._slices.append(CachedPolicy(conf, cpolicy.FastInitSingleSliceMdpPolicy))
            self._slices[-1].init()
            self._slices[-1].calculate_policy()
            servers_left -= int(np.array(self._slices[-1].policy).max())

    def calculate_policy(self):
        self._policy = []

        for state in self._states:  # @ for each state
            multislice_action = []

            if self._config.algorithm == 'fh':
                raise NotImplementedError('add simplifiedprioritymdppolicy.calculate_policy() for fh')
            else:
                for i in range(self._config.slice_count):  # @ for each singleslice in the multislice
                    i_th_state = copy(state[i])

                    slice_action = \
                        self._slices[i].policy[self._slices[i].states.index(i_th_state)]
                    multislice_action.append(slice_action)

                self._policy.append(multislice_action)


# TODO: this only works with 2 slices, need to be updated!
class SequentialPriorityMultiSliceMdpPolicy(PriorityMultiSliceMdpPolicy):
    def init(self):
        slice_1_max = self._init_slices()
        self._generate_states([self._slices[0][self._config.server_max_cap], self._slices[1][slice_1_max]])
        self._generate_actions()

    def _init_slices(self):
        """ Preparing multiprocessing stuff """
        processes = []
        self._slices = [list() for _ in range(self._config.slice_count)]

        # run highest priority slice
        _run_singleslice_from_confs(0, [self._config.slice(0)], False)
        self._slices[0] = [list() for _ in range(self._config.slice(0).server_max_cap + 1)]
        self._slices[0][self._config.slice(0).server_max_cap] = \
            CachedPolicy(self._config.slice(0), cpolicy.FastInitSingleSliceMdpPolicy)

        self._slices[1] = [list() for _ in range(self._config.server_max_cap + 1)]
        slice_0_min = min(self._slices[0][self._config.slice(0).server_max_cap].policy)
        slice_0_max = max(self._slices[0][self._config.slice(0).server_max_cap].policy)

        slice_1_min = self._config.server_max_cap - slice_0_max
        slice_1_max = self._config.server_max_cap - slice_0_min

        subconds_dicts = _get_range_subconfs_from_singleslice(self._config.slice(1), slice_1_min, slice_1_max)
        subconfs = [[subconds_dicts[e]] for e in subconds_dicts]

        for s in range(len(subconfs)):
            processes.append(multiprocessing.Process(target=_run_singleslice_from_confs,
                                                     args=(1, subconfs[s], True if s > 0 else False, )))
            processes[-1].start()

        for process in processes:
            process.join()

        to_add = _run_range_subslices(self._config.slice(1), slice_1_min, slice_1_max)
        to_add.reverse()

        for i in range(slice_1_min, slice_1_max + 1):
            self._slices[1][i] = to_add.pop()

        return slice_1_max


class SingleSliceStaticPolicy(Policy):
    @property
    def policy(self):
        return self._policy

    @property
    def states(self):
        return self._states

    def init(self):
        self._generate_states()

    def calculate_policy(self):
        self._policy = [self._config.allocation] * len(self._states)

    def get_action_from_policy(self, current_state, current_timeslot):
        return self._policy[self._states.index(current_state)]

    def _generate_states(self):
        self._states = []
        for i in range(self._config.allocation + 1):
            for j in range(self._config.queue_size + 1):
                self._states.append(SingleSliceState(j, i))


class MultiSliceStaticPolicy(Policy):
    @property
    def policy(self):
        return self._policy

    @property
    def states(self):
        return self._states

    def init(self):
        self._init_slices()
        self._generate_states(self._slices)

    def calculate_policy(self):
        self._policy = []
        for s in self._states:
            action = []
            for i in range(len(s)):
                action.append(self._slices[i].get_action_from_policy(s[i], 0))
            self._policy.append(action)

    def get_action_from_policy(self, current_state, current_timeslot):
        return self._policy[self._states.index(current_state)]


    def _init_slices(self):
        self._slices = []

        for i in range(self._config.slice_count):
            slice_conf = self._config.slice(i)
            self._slices.append(SingleSliceStaticPolicy(slice_conf))
            self._slices[-1].init()
            self._slices[-1].calculate_policy()

    def _generate_states(self, from_slices=[]):
        slices_states = [s.states for s in from_slices]
        mesh = np.array(np.meshgrid(*slices_states))

        to_filter = mesh.T.reshape(-1, len(slices_states)).tolist()

        self._states = []

        for multislice_state in to_filter:
            if sum([singleslice_state.n for singleslice_state in multislice_state]) <= self._config.server_max_cap:
                self._states.append(multislice_state)
