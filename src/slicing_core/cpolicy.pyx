import numpy as np
cimport numpy as np
import mdptoolbox
from copy import copy
import multiprocessing

from src.slicing_core.state import SingleSliceState
from src.slicing_core.utils import _Cache


cdef class CachedMatrix:
    pass


cdef class Policy:
    cdef public object _config
    cdef public object _policy
    cdef public object _states

    def __init__(self, policy_config):
        self._config = policy_config
        self._policy = []
        self._states = []

    @property
    def policy(self):
        pass

    @property
    def states(self):
        pass

    def init(self):
        pass

    def calculate_policy(self):
        pass

    def get_action_from_policy(self, current_state, current_timeslot):
        pass


cdef class SingleSliceMdpPolicy(Policy):
    cdef public object _transition_matrix
    cdef public object _reward_matrix
    cdef public object _actions

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

    cpdef void init(self):
        self._generate_states()
        self._generate_actions()
        self._generate_transition_matrix()
        self._generate_reward_matrix()

    cpdef void calculate_policy(self):
        cdef int i
        cdef int j

        if self._config.algorithm == 'vi':
            self._policy = list(self._run_value_iteration(self._config.discount_factor))
            # translating action id in the policy table with the real action
            # cdef int i
            for i in range(len(self._policy)):
                self._policy[i] = self._actions[self._policy[i]]

        if self._config.algorithm == 'rvi':
            self._policy = list(self._run_relative_value_iteration())
            # translating action id in the policy table with the real action
            # cdef int i
            for i in range(len(self._policy)):
                self._policy[i] = self._actions[self._policy[i]]

        elif self._config.algorithm == 'fh':
            policy = self._run_finite_horizon(self._config.discount_factor)
            self._policy = np.empty_like(policy, dtype=object)
            # translating action id in the policy table with the real action
            # cdef int i
            for i in range(len(self._policy)):
                # cdef int j
                for j in range(len(self._policy[i])):
                    self._policy[i][j] = self._actions[policy[i][j]]

            self._policy = self._policy.tolist()

    cpdef list get_action_from_policy(self, object current_state, int current_timeslot):
        cdef int i
        for i in range(len(current_state)):
            current_state[i].k = int(current_state[i].k / self._config.queue_scaling) * self._config.queue_scaling

        if self._config.algorithm == 'fh':
            return self._policy[self._states.index(current_state)][current_timeslot]
        else:
            return self._policy[self._states.index(current_state)]

    cdef void _generate_states(self):
        self._states = []

        cdef int i, j
        for i in range(self._config.server_max_cap + 1):
            # cdef int j
            for j in range(0, self._config.queue_size + 1, self._config.queue_scaling):
                self._states.append(SingleSliceState(j, i))

    cdef void _generate_actions(self):
        self._actions = [i for i in range(self._config.server_max_cap + 1)]

    cdef void _generate_transition_matrix(self):
        cdef np.ndarray transition_matrix = np.zeros((self._config.server_max_cap + 1, len(self._states), len(self._states)))

        # lets iterate the trans matrix and fill with correct probabilities
        cdef int a, i, j, i_from, j_to
        cdef object hidden_from, hidden_to
        cdef double prob_tmp

        for a in range(len(transition_matrix)):
            for i in range(len(self._states)):
                for j in range(len(self._states)):

                    if self._config.queue_scaling > 1:
                        # example: if queue_scaling is 5:
                        #   Pr[(0,X)->(5,X)] is the same as without the scaling
                        #   but Pr[(0,X)->(0,X)] is bigger because of the sum of the prob. of the hidden states
                        #   Pr[(0,X)->(0,X)] = Pr[(0,X)->(0,X)] + Pr[(0,X)->(1,X)] + Pr[(0,X)->(2,X)] ... etc

                        # cdef int i_from
                        for i_from in range(self._config.queue_scaling):
                            hidden_from = copy(self._states[i])
                            hidden_from.k += i_from

                            # cdef int j_to
                            for j_to in range(self._config.queue_scaling):
                                hidden_to = copy(self._states[j])
                                hidden_to.k += j_to
                                if hidden_to.k <= self._config.queue_size:
                                    prob_tmp = self._calculate_transition_probability(hidden_from, hidden_to, a)
                                    transition_matrix[a][i][j] += prob_tmp
                    else:
                        transition_matrix[a][i][j] = \
                            self._calculate_transition_probability(self._states[i], self._states[j], a)

                transition_matrix[a][i] /= transition_matrix[a][i].sum()

        self._transition_matrix = transition_matrix


    cdef double _calculate_transition_probability(self, object from_state, object to_state, int action_id):
        cdef list h_d

        if not self._config.immediate_action:
            h_d = self._calculate_h_d(from_state)
        else:
            h_d = self._calculate_h_d(to_state)

        cdef object diff = to_state - from_state

        cdef double tmp = 0
        cdef double tmp2 = 0

        cdef int x
        cdef double p_proc, p_arr

        for x in range(max(0, diff.k), self._config.queue_size - from_state.k + 1):
            p_proc = 0


            if x <= (len(self._config.arrivals_histogram) - 1):
                p_arr = self._config.arrivals_histogram[x]
            else:
                p_arr = 0
            # try:
            #     p_arr = self._config.arrivals_histogram[x]
            # except IndexError:
            #     p_arr = 0

            if from_state.k + x - to_state.k == \
                    ((from_state.k + x) if self._config.arrival_processing_phase else from_state.k):
                p_proc = sum(h_d[from_state.k + x - to_state.k:])
            elif from_state.k + x - to_state.k < \
                    ((from_state.k + x) if self._config.arrival_processing_phase else from_state.k):

                if (from_state.k + x - to_state.k) <= (len(h_d) - 1):
                    p_proc = h_d[from_state.k + x - to_state.k]
                # try:
                #     p_proc = h_d[from_state.k + x - to_state.k]
                # except IndexError:
                #     pass
            else:
                p_proc = 0

            tmp += p_arr * p_proc

        for x in range(self._config.queue_size - from_state.k + 1,
                       len(self._config.arrivals_histogram) + 1):
            p_proc = 0

            if x <= (len(self._config.arrivals_histogram) - 1):
                p_arr = self._config.arrivals_histogram[x]
            else:
                p_arr = 0
            # try:
            #     p_arr = self._config.arrivals_histogram[x]
            # except IndexError:
            #     p_arr = 0

            if self._config.queue_size - to_state.k == self._config.queue_size:
                p_proc = sum(h_d[self._config.queue_size - to_state.k:])
            elif self._config.queue_size - to_state.k < self._config.queue_size:

                if (self._config.queue_size - to_state.k) <= (len(h_d) - 1):
                    p_proc = h_d[self._config.queue_size - to_state.k]
                # try:
                #     p_proc = h_d[self._config.queue_size - to_state.k]
                # except IndexError:
                #     pass
            else:
                p_proc = 0

            tmp2 += p_arr * p_proc

        cdef double transition_probability = tmp + tmp2

        return self._filter_transition_probability_by_action(transition_probability, to_state, action_id)

    cdef list _calculate_h_d(self, object state):
        cdef list h_d = [1.]  # default H_d value for S = 0
        if state.n > 0:
            h_d = self._config.server_capacity_histogram
            for _ in range(1, state.n):
                h_d = np.convolve(h_d, self._config.server_capacity_histogram).tolist()
        return h_d

    cdef double _filter_transition_probability_by_action(self, double transition_probability, object to_state, int action_id):
        if to_state.n != action_id:
            return 0
        return transition_probability

    cdef void _generate_reward_matrix(self):
        cdef np.ndarray reward_matrix = np.zeros((self._config.server_max_cap + 1, len(self._states), len(self._states)))

        cdef int a, i, j, j_to
        cdef object hidden_to
        cdef double rew_tmp, min_value

        for a in range(len(reward_matrix)):
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
                                    reward_matrix[a][i][j] += rew_tmp
                        else:
                            reward_matrix[a][i][j] = self._calculate_transition_reward(self._states[j])

        if self._config.normalize_reward_matrix:
            # normalize the reward matrix
            min_value = - reward_matrix.min()
            reward_matrix /= min_value

        self._reward_matrix = reward_matrix

    cdef double _calculate_transition_reward(self, object to_state):
        # C = alpha * C_k * num of jobs + beta * C_n * num of server + gamma * C_l * E(num of lost jobs)
        cdef double cost1 = self._config.c_job * to_state.k
        cdef double cost2 = self._config.c_server * to_state.n
        cdef double cost3 = 0

        # expected value of lost packets
        cdef int i
        for i in range(len(self._config.arrivals_histogram)):
            if to_state.k + i > self._config.queue_size:
                cost3 += self._config.arrivals_histogram[i] * i * self._config.c_lost

        return - (self._config.alpha * cost1 +
                  self._config.beta * cost2 +
                  self._config.gamma * cost3)

    cdef list _run_relative_value_iteration(self):
        rvi = mdptoolbox.mdp.RelativeValueIteration(self._transition_matrix, self._reward_matrix)
        rvi.run()
        return list(rvi.policy)

    cdef list _run_value_iteration(self, float discount):
        vi = mdptoolbox.mdp.ValueIteration(self._transition_matrix, self._reward_matrix, discount)
        vi.run()
        return list(vi.policy)

    cdef list _run_finite_horizon(self, float discount):
        fh = mdptoolbox.mdp.FiniteHorizon(self._transition_matrix, self._reward_matrix,
                                          discount, self._config.timeslots)
        fh.run()
        return list(fh.policy)

# ---------------------------------------------------------------------------------------------------------

cpdef void _calculate_single_trans_matrix(int action_num, object mdp_object):
    cdef np.ndarray transition_matrix = np.zeros((len(mdp_object._states), len(mdp_object._states)))

    cdef int i, j, i_from, j_to
    cdef object hidden_from, hidden_to
    cdef double prob_tmp
    cdef object cache, base_conf

    base_conf = copy(mdp_object._config)
    del base_conf.server_max_cap
    cache = _Cache(base_conf, f"{action_num}.single_trans_matrix")

    if cache.load() is None:
        print(f"{action_num}.single_trans_matrix not yet cached: calculating")
        for i in range(len(mdp_object._states)):
            for j in range(len(mdp_object._states)):

                if mdp_object._config.queue_scaling > 1:
                    # example: if queue_scaling is 5:
                    #   Pr[(0,X)->(5,X)] is the same as without the scaling
                    #   but Pr[(0,X)->(0,X)] is bigger because of the sum of the prob. of the hidden states
                    #   Pr[(0,X)->(0,X)] = Pr[(0,X)->(0,X)] + Pr[(0,X)->(1,X)] + Pr[(0,X)->(2,X)] ... etc

                    # cdef int i_from
                    for i_from in range(mdp_object._config.queue_scaling):
                        hidden_from = copy(mdp_object._states[i])
                        hidden_from.k += i_from

                        # cdef int j_to
                        for j_to in range(mdp_object._config.queue_scaling):
                            hidden_to = copy(mdp_object._states[j])
                            hidden_to.k += j_to
                            if hidden_to.k <= mdp_object._config.queue_size:
                                prob_tmp = mdp_object._calculate_transition_probability(hidden_from, hidden_to, action_num)
                                transition_matrix[i][j] += prob_tmp
                else:
                    transition_matrix[i][j] = \
                        mdp_object._calculate_transition_probability(mdp_object._states[i], mdp_object._states[j], action_num)

            transition_matrix[i] /= transition_matrix[i].sum()
        cache.store(transition_matrix)


# this version have a faster init phase because of matrix caching and multiprocessing
cdef class FastInitSingleSliceMdpPolicy(SingleSliceMdpPolicy):
    cdef void _generate_transition_matrix(self):
        cdef int a, i
        cdef object processes = []
        cdef object cached, base_cached_conf
        cdef np.ndarray transition_matrix

        for a in range(self._config.server_max_cap + 1):
            processes.append(multiprocessing.Process(target=_calculate_single_trans_matrix, args=(a, self)))
            processes[-1].start()

        for process in processes:
            process.join()

        # loading cached matrices and composing the
        transition_matrix = np.zeros(
            (self._config.server_max_cap + 1, len(self._states), len(self._states)))

        base_cached_conf = copy(self._config)
        del base_cached_conf.server_max_cap

        for i in range(self._config.server_max_cap + 1):
            cached = _Cache(base_cached_conf, f"{i}.single_trans_matrix")
            transition_matrix[i] = cached.load()

        self._transition_matrix = transition_matrix



    cpdef double _calculate_transition_probability(self, object from_state, object to_state, int action_id):
        return SingleSliceMdpPolicy._calculate_transition_probability(self, from_state, to_state, action_id)



# ---------------------------------------------------------------------------------------------------------







# # this version have a faster init phase because of matrix caching and multiprocessing
# cdef class FastInitSingleSliceMdpPolicy(SingleSliceMdpPolicy):
#
#     cdef void _generate_transition_matrix(self):
#         cdef int a
#         cdef object processes = []
#
#         print("ECCOMIMIEMIEMIEIMEMI")
#         for a in range(self._config.server_max_cap + 1):
#             processes.append(multiprocessing.Process(target=_calculate_single_trans_matrix, args=(a, copy(self))))
#             processes[-1].start()
#
#         for process in processes:
#             process.join()







