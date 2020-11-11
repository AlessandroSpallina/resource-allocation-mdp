import abc
import numpy as np
import mdptoolbox
import math

from src.slicing_core.state import SingleSliceState


class Policy(metaclass=abc.ABCMeta):
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


class SingleSliceMdpPolicy(Policy):
    def __init__(self, policy_config, slice_id):
        self._id = slice_id
        self._config = policy_config

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
        self._generate_transition_matrix()
        self._generate_reward_matrix()

    def calculate_policy(self):
        if self._config.algorithm == 'vi':
            self._policy = self._run_value_iteration(self._config.discount_factors)
        elif self._config.algorithm == 'fh':
            self._policy = self._run_finite_horizon(self._config.discount_factors)

    def get_action_from_policy(self, current_state, current_timeslot):
        try:
            if len(self._policy[0]) > 0:  # if we are here the policy is a matrix (fh)
                return self._policy[:, current_timeslot][self._states.index(current_state)]
        except TypeError:
            return self._policy[self._states.index(current_state)]

    def _generate_states(self):
        self._states = []
        for i in range(self._config.server_max_cap + 1):
            for j in range(self._config.slices[self._id].queue_size + 1):
                self._states.append(SingleSliceState(j, i))

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

        for x in range(max(0, diff.k), self._config.slices[self._id].queue_size - from_state.k + 1):
            p_proc = 0

            try:
                p_arr = self._config.slices[self._id].arrivals_histogram[x]
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

        for x in range(self._config.slices[self._id].queue_size - from_state.k + 1,
                       len(self._config.slices[self._id].arrivals_histogram) + 1):
            p_proc = 0

            try:
                p_arr = self._config.slices[self._id].arrivals_histogram[x]
            except IndexError:
                p_arr = 0

            if self._config.slices[self._id].queue_size - to_state.k == self._config.slices[self._id].queue_size:
                p_proc = sum(h_d[self._config.slices[self._id].queue_size - to_state.k:])
            elif self._config.slices[self._id].queue_size - to_state.k < self._config.slices[self._id].queue_size:
                try:
                    p_proc = h_d[self._config.slices[self._id].queue_size - to_state.k]
                except IndexError:
                    pass
            else:
                p_proc = 0

            tmp2 += p_arr * p_proc

        transition_probability = tmp + tmp2

        return self._filter_transition_probability_by_action(transition_probability, from_state, to_state, action_id)

    def _calculate_h_d(self, state):
        h_d = [1.]  # default H_d value for S = 0
        if state.n > 0:
            h_d = self._config.slices[self._id].server_capacity_histogram
            for i in range(1, state.n):
                h_d = np.convolve(h_d, self._config.slices[self._id].server_capacity_histogram)
        return h_d

    def _filter_transition_probability_by_action(self, transition_probability, from_state, to_state, action_id):
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
        cost1 = self._config.slices[self._id].c_job * to_state.k
        cost2 = self._config.slices[self._id].c_server * to_state.n
        cost3 = 0

        # expected value of lost packets
        for i in range(len(self._config.slices[self._id].arrivals_histogram)):
            if to_state.k + i > self._config.slices[self._id].queue_size:
                cost3 += self._config.slices[self._id].arrivals_histogram[i] * i * self._config.slices[self._id].c_lost

        return - (self._config.slices[self._id].alpha * cost1 +
                  self._config.slices[self._id].beta * cost2 +
                  self._config.slices[self._id].gamma * cost3)

    def _run_value_iteration(self, discount):
        if type(discount) == list:
            to_return = []
            for i in discount:
                vi = mdptoolbox.mdp.ValueIteration(self._transition_matrix, self._reward_matrix, i)
                vi.run()
                to_return.append(vi.policy)
            return to_return

        vi = mdptoolbox.mdp.ValueIteration(self._transition_matrix, self._reward_matrix, discount)
        vi.run()
        return vi.policy

    def _run_finite_horizon(self, discount):
        if type(discount) == list:
            to_return = []
            for i in discount:
                vi = mdptoolbox.mdp.FiniteHorizon(self._transition_matrix, self._reward_matrix,
                                                  i, self._config.timeslots)
                vi.run()
                to_return.append(vi.policy)
            return to_return

        vi = mdptoolbox.mdp.FiniteHorizon(self._transition_matrix, self._reward_matrix,
                                          discount, self._config.timeslots)
        vi.run()
        return vi.policy


class MultiSliceMdpPolicy(Policy):
    def __init__(self, policy_config):
        self._config = policy_config

    @property
    def policy(self):
        return self._policy

    @property
    def states(self):
        return self._states

    @property
    def transition_matrix(self):
        return self._transition_matrix

    @property
    def reward_matrix(self):
        return self._reward_matrix

    def init(self):
        self._init_slices()
        self._generate_states()
        self._generate_actions()
        self._generate_transition_matrix()
        self._generate_reward_matrix()

    def calculate_policy(self):
        if self._config.algorithm == 'vi':
            self._policy = list(self._run_value_iteration(self._config.discount_factor))
            # translating action id in the policy table with the real action
            # es. of real action is [0, 5] which means: set 0 servers to slice0 and 5 servers to slice1
            for i in range(len(self._policy)):
                self._policy[i] = self._actions[self._policy[i]].tolist()

        elif self._config.algorithm == 'fh':
            self._policy = self._run_finite_horizon(self._config.discount_factor)
            # translating action id in the policy table with the real action
            for i in range(len(self._policy)):
                for j in range(len(self._policy[i])):
                    self._policy[i][j] = self._actions[self._policy[i][j]]

    def get_action_from_policy(self, current_state, current_timeslot):
        try:
            if len(self._policy[0]) > 0:  # if we are here the policy is a matrix (fh)
                return self._policy[:, current_timeslot][self._states.index(current_state)]
        except TypeError:
            return self._policy[self._states.index(current_state)]

    def _init_slices(self):
        self._slices = []

        for i in range(self._config.slice_count):
            self._slices.append(SingleSliceMdpPolicy(self._config, i))
            self._slices[-1].init()

        # pool = Pool(self._config.slice_count)
        # # pool.map([slice.init() for slice in self._slices])
        #
        # def multiprocess_slice_init(s):
        #     s.init()
        #     return s
        #
        # self._slices = pool.map(multiprocess_slice_init, self._slices)
        # pool.close()
        # # pool.join()
        # # print("d")

    def _generate_states(self):
        slices_states = [s.states for s in self._slices]
        mesh = np.array(np.meshgrid(*slices_states))

        # TODO: abbiamo stati inutili (es. se max cap=5 -> ((0,5),(0,5)) che è impossibile), ottimizzare
        # to_filter = mesh.T.reshape(-1, len(slices_states)).tolist()
        # self._states = []
        # for multislice_state in to_filter:
        #     if sum([singleslice_state.n for singleslice_state in multislice_state]) <= self._config.server_max_cap:
        #         self._states.append(multislice_state)

        self._states = mesh.T.reshape(-1, len(slices_states)).tolist()

    def _generate_actions(self):
        tmp = []
        for single_slice_state in self._states[-1]:
            tmp.append(list(range(single_slice_state.n + 1)))

        mesh = np.array(np.meshgrid(tmp[0], tmp[1]))
        to_filter = mesh.T.reshape(-1, 2)

        self._actions = []
        for i in to_filter:
            if sum(i) <= tmp[0][-1]:
                self._actions.append(i)

    def _calculate_transition_probability(self, from_state, to_state, action):
        transition_probability = \
            math.prod([self._slices[i].transition_matrix[action[i]]
                       [self._slices[i].states.index(from_state[i])]
                       [self._slices[i].states.index(to_state[i])] for i in range(self._config.slice_count)])
        return transition_probability

    def _generate_transition_matrix(self):
        self._transition_matrix = np.zeros((len(self._actions), len(self._states), len(self._states)))

        for a in range(len(self._transition_matrix)):
            for i in range(len(self._states)):
                for j in range(len(self._states)):
                    self._transition_matrix[a][i][j] = \
                        self._calculate_transition_probability(self._states[i], self._states[j], self._actions[a])

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

    def _run_value_iteration(self, discount):
        if type(discount) == list:
            to_return = []
            for i in discount:
                vi = mdptoolbox.mdp.ValueIteration(self._transition_matrix, self._reward_matrix, i)
                vi.run()
                to_return.append(vi.policy)
            return to_return

        vi = mdptoolbox.mdp.ValueIteration(self._transition_matrix, self._reward_matrix, discount)
        vi.run()
        return vi.policy

    def _run_finite_horizon(self, discount):
        if type(discount) == list:
            to_return = []
            for i in discount:
                vi = mdptoolbox.mdp.FiniteHorizon(self._transition_matrix, self._reward_matrix,
                                                  i, self._config.timeslots)
                vi.run()
                to_return.append(vi.policy)
            return to_return

        vi = mdptoolbox.mdp.FiniteHorizon(self._transition_matrix, self._reward_matrix,
                                          discount, self._config.timeslots)
        vi.run()
        return vi.policy


# class SingleSliceConservativePolicy(Policy):
#     def __init__(self, policy_config, slice_id):
#         self._id = slice_id
#         self._config = policy_config
#
#     @property
#     def policy(self):
#         return self._policy
#
#     def calculate_policy(self):
#         self._policy = []
#         for state in self._states:
#             self._policy.append(state.k)
#
#     def get_action_from_policy(self, current_state, current_timeslot):
#         return self._policy[self._states.index(current_state)]
#
#     def _generate_states(self):
#         self._states = []
#         for i in range(self._config.server_max_cap + 1):
#             for j in range(self._config.slices[self._id].queue_size + 1):
#                 self._states.append(SingleSliceState(j, i))
#
#
# class MultiSliceConservativePolicy(Policy):
#     def __init__(self, policy_config):
#         self._config = policy_config
#
#     def calculate_policy(self):
#         pass
#
#     def get_action_from_policy(self, current_state):
#         pass
#
