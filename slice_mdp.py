import mdptoolbox
import numpy as np

from memory_profiler import profile
from state import State


class SliceMDP:
    def __init__(self, arrivals_histogram, departures_histogram, queue_size, max_server_num, algorithm='vi',
                 periods=1000, c_job=1, c_server=1, c_lost=1, alpha=1, beta=1, gamma=1, delayed_action=True,
                 verbose=False):

        self._verbose = verbose

        self._delayed_action = delayed_action

        self._arrivals_histogram = arrivals_histogram
        self._departures_histogram = departures_histogram
        self._queue_size = queue_size
        self._max_server_num = max_server_num
        self._algorithm = algorithm
        self._periods = periods

        # trans matrix stuff
        self._states = self._generate_states()
        self._transition_matrix = self._generate_transition_matrix()

        # reward stuff
        self._c_job = c_job
        self._c_server = c_server
        self._c_lost = c_lost
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._reward_matrix = self._generate_reward_matrix()

    @property
    def transition_matrix(self):
        return self._transition_matrix

    @property
    def reward_matrix(self):
        return self._reward_matrix

    @property
    def states(self):
        return self._states

    """
    es. b=2; s=1
    
    (k job, n server)
    S0: (0,0)
    S1: (1,0)
    S2: (2,0)
    S3: (0,1)
    S4: (1,1)
    S5: (2,1)
    """
    def _generate_states(self):
        states = []  # state at pos 0 -> S0, etc..
        for i in range(self._max_server_num + 1):
            for j in range(self._queue_size + 1):
                states.append(State(j, i))

        if self._verbose:
            for i in range(len(states)):
                print(f"S{i}: {states[i]}")

        return states

    def _calculate_h_d(self, state):
        # le probabilità di departure hanno senso solo per server_num > 0
        # in caso di server_num > 0, H_p per il sistema = H_p convuluto H_p nel caso di due server
        # perchè H_p_nserver = [P(s1:0)*P(s2:0), P(s1:0)*P(s2:1) + P(s1:1)*P(s2:0), P(s1:1)*P(s2:1)]
        # NB: la convoluzione è associativa, quindi farla a due a due per n volte è ok

        h_d = [1.]  # default H_d value for S = 0
        if state.n > 0:
            h_d = self._departures_histogram
            for i in range(1, state.n):
                h_d = np.convolve(h_d, self._departures_histogram)
        return h_d

    def _calculate_transition_probability(self, from_state, to_state, action_id):
        if self._delayed_action:
            h_d = self._calculate_h_d(from_state)
        else:
            h_d = self._calculate_h_d(to_state)

        diff = to_state - from_state

        tmp = 0
        tmp2 = 0

        for x in range(max(0, diff.k), self._queue_size - from_state.k + 1):
            p_proc = 0

            try:
                p_arr = self._arrivals_histogram[x]
            except IndexError:
                p_arr = 0

            if from_state.k + x - to_state.k >= from_state.k + x:
                p_proc = sum(h_d[from_state.k + x - to_state.k:])
            else:
                try:
                    p_proc = h_d[from_state.k + x - to_state.k]
                except IndexError:
                    pass

            tmp += p_arr * p_proc

        for x in range(self._queue_size - from_state.k + 1, len(self._arrivals_histogram) + 1):
            p_proc = 0

            try:
                p_arr = self._arrivals_histogram[x]
            except IndexError:
                p_arr = 0

            if self._queue_size - to_state.k >= self._queue_size:
                p_proc = sum(h_d[self._queue_size - to_state.k:])
            else:
                try:
                    p_proc = h_d[self._queue_size - to_state.k]
                except IndexError:
                    pass

            tmp2 += p_arr * p_proc

        transition_probability = tmp + tmp2

        # adesso valuto le eventuali transizioni "verticali"
        if action_id == 0 \
                or (action_id == 1 and diff.n == 0 and from_state.n == self._max_server_num)\
                or (action_id == 2 and diff.n == 0 and from_state.n == 0):  # do nothing
            if diff.n != 0:
                return 0.

        elif action_id == 1:  # allocate 1 server
            if diff.n != 1:  # and to_state.n != self._max_server_num:
                return 0.

        elif action_id == 2:  # deallocate 1 server
            if diff.n != -1:  # and to_state.n != 0:
                return 0.

        return transition_probability

    """
    The transition matrix is of dim action_num * states_num * states_num
    Q[0] -> transition matrix related action 0 (do nothing)
    Q[1] -> transition matrix related action 0 (allocate 1)
    Q[2] -> transition matrix related action 0 (deallocate 1)
    """
    def _generate_transition_matrix(self):
        transition_matrix = np.zeros((3, len(self._states), len(self._states)))

        # lets iterate the trans matrix and fill with correct probabilities
        for a in range(len(transition_matrix)):
            for i in range(len(self._states)):
                for j in range(len(self._states)):
                    transition_matrix[a][i][j] = self._calculate_transition_probability(self._states[i],
                                                                                        self._states[j], a)
        return transition_matrix

    def _calculate_transition_reward(self, to_state):
        # utilizzo approccio "costo di stare nello stato", mi serve solo lo stato di arrivo
        # costs are mapped into the reward matrix
        # C = alpha * C_k * num of jobs + beta * C_n * num of server + gamma * C_l * E(num of lost jobs)
        cost1 = self._c_job * to_state.k
        cost2 = self._c_server * to_state.n
        cost3 = 0

        # expected value of lost packets
        for i in range(len(self._arrivals_histogram)):
            if to_state.k + i > self._queue_size:
                cost3 += self._arrivals_histogram[i] * i * self._c_lost

        return - (self._alpha * cost1 + self._beta * cost2 + self._gamma * cost3)

    def _generate_reward_matrix(self):
        reward_matrix = np.zeros((3, len(self._states), len(self._states)))

        for a in range(len(reward_matrix)):
            for i in range(len(self._states)):
                for j in range(len(self._states)):
                    if self._transition_matrix[a][i][j] > 0:
                        #  reward_matrix[a][i][j] = self._calculate_transition_reward2(self._states[i], self._states[j])
                        reward_matrix[a][i][j] = self._calculate_transition_reward(self._states[j])
        return reward_matrix

    def _run_value_iteration(self, discount):
        # The standard family of algorithms to calculate optimal policies for finite state and action MDPs requires
        # storage for two arrays indexed by state: value V, which contains real values, and policy PI,
        # which contains actions. At the end of the algorithm, PI  will
        # contain the solution and V(s) will contain the discounted sum of the rewards to be earned
        # (on average) by following that solution from state s.
        if type(discount) == list:
            to_return = {}
            for i in discount:
                vi = mdptoolbox.mdp.ValueIteration(self._transition_matrix, self._reward_matrix, i)
                vi.run()
                to_return[f"mdp({str(round(i, 1)).replace('.', ',')})"] = vi.policy
            return to_return

        vi = mdptoolbox.mdp.ValueIteration(self._transition_matrix, self._reward_matrix, discount)
        vi.run()
        return vi.policy

    def _run_finite_horizon(self, discount):
        if type(discount) == list:
            to_return = {}
            for i in discount:
                vi = mdptoolbox.mdp.FiniteHorizon(self._transition_matrix, self._reward_matrix, i, self._periods)
                vi.run()
                to_return[f"mdp({str(round(i, 1)).replace('.', ',')})"] = vi.policy
            return to_return

        vi = mdptoolbox.mdp.FiniteHorizon(self._transition_matrix, self._reward_matrix, discount, self._periods)
        vi.run()
        return vi.policy

    # @profile
    def run(self, discount):
        if self._algorithm == 'vi':
            return self._run_value_iteration(discount)
        if self._algorithm == 'fh':
            return self._run_finite_horizon(discount)
