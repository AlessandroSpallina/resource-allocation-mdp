import mdptoolbox
import numpy as np
import utils


class State:
    def __init__(self, k, n):
        self._k = k  # k jobs
        self._n = n  # n servers

    @property
    def k(self):
        return self._k

    @k.setter
    def k(self, k):
        self._k = k

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, n):
        self._n = n

    def __str__(self):
        return "(" + str(self._k) + "," + str(self._n) + ")"

    def __sub__(self, other):
        k = self.k - other.k
        n = self.n - other.n
        return State(k, n)

    def __eq__(self, other):
        if isinstance(other, State):
            return self._k == other._k and self._n == other._n
        return False


class SliceMDP:
    def __init__(self, arrivals_histogram, departures_histogram, queue_size, max_server_num,
                 c_job=1, c_server=1, c_lost=1, alpha=0.5, verbose=True):

        self._verbose = verbose

        self._arrivals_histogram = arrivals_histogram
        self._departures_histogram = departures_histogram
        self._queue_size = queue_size
        self._max_server_num = max_server_num

        # trans matrix stuff
        self._states = self._generate_states()
        self._transition_matrix = self._generate_transition_matrix()

        # reward stuff
        self._c_job = c_job
        self._c_server = c_server
        self._c_lost = c_lost
        self._alpha = alpha
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

    def _calculate_h_p(self, state):
        # le probabilità di departure hanno senso solo per server_num > 0
        # in caso di server_num > 0, H_p per il sistema = H_p convuluto H_p nel caso di due server
        # perchè H_p_nserver = [P(s1:0)*P(s2:0), P(s1:0)*P(s2:1) + P(s1:1)*P(s2:0), P(s1:1)*P(s2:1)]
        # NB: la convoluzione è associativa, quindi farla a due a due per n volte è ok
        h_p = self._departures_histogram
        for i in range(1, state.n):
            h_p = np.convolve(h_p, self._departures_histogram)
        return h_p

    # old calculate_transition_prob https://0bin.net/paste/myC+f0f5d6CoKF79#4nJw-dt4IhQxddw6D4EjbiG0aWeUhzdYniWC3i0DWjm
    def _calculate_transition_probability(self, from_state, to_state, action_id):
        h_p = self._calculate_h_p(from_state)
        transition_probability = 0
        diff = to_state - from_state

        # prima valuto le transizioni "orizzontali"
        if from_state.n == 0:
            if diff.k == 0 and to_state.k == self._queue_size:
                transition_probability = 1.
            elif diff.k >= 0:
                try:
                    transition_probability = self._arrivals_histogram[diff.k]
                except IndexError:
                    pass
        else:
            if diff.k == 0 and to_state.k == 0:
                tmp = 0
                for i in range(1, len(h_p)):
                    try:
                        tmp += self._arrivals_histogram[i] * h_p[i]
                    except IndexError:
                        pass
                transition_probability = self._arrivals_histogram[0] + tmp

            elif diff.k >= 0:
                if to_state.k < self._queue_size:
                    tmp = 0
                    tmp2 = 0
                    for i in range(diff.k, self._queue_size):
                        try:
                            tmp += self._arrivals_histogram[i] * h_p[i - diff.k]
                        except IndexError:
                            pass
                    for i in range(len(self._arrivals_histogram) - self._queue_size):
                        try:
                            tmp2 += self._arrivals_histogram[self._queue_size + i] * h_p[self._queue_size - to_state.k]
                        except IndexError:
                            pass
                    transition_probability = tmp + tmp2
                elif to_state.k == self._queue_size:
                    for i in range(diff.k, len(self._arrivals_histogram)):
                        try:
                            transition_probability += self._arrivals_histogram[i] * h_p[0]
                        except IndexError:
                            pass

            elif diff.k < 0:
                for i in range(-diff.k, len(h_p)):
                    try:
                        transition_probability += self._arrivals_histogram[diff.k + i] * h_p[i]
                    except IndexError:
                        pass
                if self._queue_size <= len(h_p) - 1 or from_state.k == self._queue_size and -diff.k <= len(h_p) - 1:
                    for i in range(len(h_p) - 1, len(self._arrivals_histogram)):
                        try:
                            transition_probability += self._arrivals_histogram[i] * h_p[len(h_p) - 1]
                        except IndexError:
                            pass

        # adesso valuto le eventuali transizioni "verticali"
        if action_id == 0:  # do nothing
            if diff.n != 0:
                return 0.

        elif action_id == 1:  # allocate 1 server
            if diff.n != 1 and to_state.n != self._max_server_num:
                return 0.0

        elif action_id == 2:  # deallocate 1 server
            if diff.n != -1 and to_state.n != 0:
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
                    transition_matrix[a][i][j] = self._calculate_transition_probability(self._states[i], self._states[j], a)
        return transition_matrix

    def _calculate_transition_reward(self, to_state):
        # utilizzo approccio "costo di stare nello stato", mi serve solo lo stato di arrivo
        # costs are mapped into the reward matrix
        # C = alpha * C_k * num of jobs + (1 - alpha) * C_n * num of server
        costs = self._alpha * self._c_job * to_state.k + (1 - self._alpha) * self._c_server * to_state.n

        # expected value of lost packets
        for i in range(len(self._arrivals_histogram)):
            if to_state.k + i > self._queue_size:
                #print(to_state)
                costs += self._arrivals_histogram[i] * i * self._c_lost

        return -costs

    def _calculate_transition_reward2(self, from_state, to_state):
        # utilizzo approccio "costo del passaggio"
        costs_from = self._alpha * self._c_job * from_state.k + (1 - self._alpha) * self._c_server * from_state.n
        costs_to = self._alpha * self._c_job * to_state.k + (1 - self._alpha) * self._c_server * to_state.n

        # expected value of lost packets
        for i in range(len(self._arrivals_histogram)):
            if to_state.k + i > self._queue_size:
                #print(to_state)
                costs_to += self._arrivals_histogram[i] * i * self._c_lost

        return costs_from - costs_to

    def _generate_reward_matrix(self):
        reward_matrix = np.zeros((3, len(self._states), len(self._states)))

        for a in range(len(reward_matrix)):
            for i in range(len(self._states)):
                for j in range(len(self._states)):
                    if self._transition_matrix[a][i][j] > 0:
                        #reward_matrix[a][i][j] = self._calculate_transition_reward2(self._states[i], self._states[j])
                        reward_matrix[a][i][j] = self._calculate_transition_reward(self._states[j])
        return reward_matrix

    def run_value_iteration(self, discount):
        # The standard family of algorithms to calculate optimal policies for finite state and action MDPs requires
        # storage for two arrays indexed by state: value V, which contains real values, and policy PI,
        # which contains actions. At the end of the algorithm, PI  will
        # contain the solution and V(s) will contain the discounted sum of the rewards to be earned
        # (on average) by following that solution from state s.
        vi = mdptoolbox.mdp.ValueIteration(self._transition_matrix, self._reward_matrix, discount)
        vi.run()
        print(f"Expected values: {vi.V}")
        return vi.policy
