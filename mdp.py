import mdptoolbox
import numpy as np


class State:
    def __init__(self, k, n):
        self._k = k  # k jobs
        self._n = n  # n servers

    @property
    def k(self):
        return self._k

    @property
    def n(self):
        return self._n

    def __str__(self):
        return "(" + str(self._k) + "," + str(self._n) + ")"


class SliceMDP:
    def __init__(self, arrivals_histogram, departures_histogram, queue_size, max_server_num,
                 c_job=1, c_server=1, alpha=0.5, verbose=True):

        self._verbose = verbose

        self._arrivals_histogram = arrivals_histogram
        self._departures_histogram = departures_histogram
        self._queue_size = queue_size
        self._max_server_num = max_server_num

        # trans matrix stuff
        self._current_server_num = 0
        self._transition_matrix = self._calculate_transition_matrix()

        # reward stuff
        self._c_job = c_job
        self._c_server = c_server
        self._alpha = alpha
        self._current_reward = self._calculate_reward()

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

        return states

    """
    The transition matrix is of dim action_num * states_num * states_num
    Q[0] -> transition matrix related action 0 (do nothing)
    Q[1] -> transition matrix related action 0 (allocate 1)
    Q[2] -> transition matrix related action 0 (deallocate 1)
    """
    def _calculate_transition_matrix(self):
        states = self._generate_states()

        if self._verbose:
            for i in range(len(states)):
                print(f"S{i}: {states[i]}")

        self._transition_matrix = np.zeros((3, len(states), len(states)))

        # action 0 (do nothing)
        if self._verbose:
            print("Calculating transition matrix for action 0 (do nothing)")

        for i in range(len(self._transition_matrix[0])):
            for j in range(len(self._transition_matrix[0])):

                # _______________

                if i <= j < self._queue_size:  # m' >= m; m' < max_jobs
                    temp = 0
                    for p_t in range(len(self._departures_histogram)):
                        try:
                            temp += self._arrivals_histogram[j - i + p_t] * self._departures_histogram[p_t]
                        except IndexError:
                            pass

                    self._transition_matrix[0][i][j] = temp

                    if self._verbose:
                        print(f"1) Q({states[i]}->{states[j]}) = {self._transition_matrix[0][i][j]}")

                elif j >= i and j == self._queue_size:  # m' >= m; m' = max_jobs
                    temp = 0
                    temp2 = 0
                    for p_t in range(len(self._departures_histogram)):
                        try:
                            temp += self._arrivals_histogram[j - i + p_t] * self._departures_histogram[p_t]
                        except IndexError:
                            pass

                        for x in range(p_t + j - i + 1, self._queue_size + 1):  # ci vuole un +1 in queue_size(?)
                            try:
                                temp2 += self._arrivals_histogram[x] * self._departures_histogram[p_t]
                            except IndexError:
                                pass

                    self._transition_matrix[0][i][j] = temp + temp2

                    if self._verbose:
                        print(f"2) Q({states[i]}->{states[j]}) = {self._transition_matrix[0][i][j]}")

                elif j < i:
                    temp = 0
                    for p_t in range(1, len(self._departures_histogram)):
                        try:
                            temp += self._arrivals_histogram[j - i + p_t] * self._departures_histogram[p_t]
                        except IndexError:
                            pass

                    self._transition_matrix[0][i][j] = temp

                    if self._verbose:
                        print(f"3) Q({states[i]}->{states[j]}) = {self._transition_matrix[0][i][j]}")






                # _______________



        return self._transition_matrix

    def _calculate_reward(self):
        pass

    def allocate_server(self, count=1):
        pass

    def deallocate_server(self, count=1):
        pass

    def run_value_iteration(self):
        pass


if __name__ == '__main__':
    # generate histogram of arrivals
    # this means: Pr(0 job incoming in this timeslot) = 0.5; Pr(1 job incoming in this timeslot) = 0.5
    arrivals = [0.5, 0.5]

    # generate histogram of arrivals
    # this means: Pr(0 job processed in this timeslot) = 0.6; Pr(1 job processed in this timeslot) = 0.4
    departures = [0.6, 0.4]

    # @findme : generare grafico partenze e arrivi

    slice_mdp = SliceMDP(arrivals, departures, 2, 1)

    print(slice_mdp._transition_matrix)



