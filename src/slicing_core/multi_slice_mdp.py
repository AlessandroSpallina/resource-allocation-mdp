import numpy as np

from state import State


# multi-slice support for different kind of slices
# TODO: support for more than two slices (look at _generate_states, fix it!)
class MultiSliceMDP:
    def __init__(self, slices):
        self._slices = slices

        self._states = self._generate_states()
        self._transition_matrix = self._generate_transition_matrix()
        # self._reward_matrix = self._generate_reward_matrix()

    def _generate_states(self):
        # see https://www.kite.com/python/answers/how-to-get-all-element-combinations-of-two-numpy-arrays-in-python
        mesh = np.array(np.meshgrid(self._slices[0].states, self._slices[1].states))
        to_ret = mesh.T.reshape(-1, 2)
        return to_ret

    @property
    def transition_matrix(self):
        return self._transition_matrix

    @property
    def reward_matrix(self):
        return self._reward_matrix

    @property
    def states(self):
        return self._states

    # ------------------------------------------
    def _filter_transition_probability_by_action(self, transition_probability, from_state, to_state, action_id):
        pass

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

            if from_state.k + x - to_state.k == ((from_state.k + x) if self._arrival_processing_phase else from_state.k):
                p_proc = sum(h_d[from_state.k + x - to_state.k:])
            elif from_state.k + x - to_state.k < ((from_state.k + x) if self._arrival_processing_phase else from_state.k):
                try:
                    p_proc = h_d[from_state.k + x - to_state.k]
                except IndexError:
                    pass
            else:
                p_proc = 0

            tmp += p_arr * p_proc

        for x in range(self._queue_size - from_state.k + 1, len(self._arrivals_histogram) + 1):
            p_proc = 0

            try:
                p_arr = self._arrivals_histogram[x]
            except IndexError:
                p_arr = 0

            if self._queue_size - to_state.k == self._queue_size:
                p_proc = sum(h_d[self._queue_size - to_state.k:])
            elif self._queue_size - to_state.k < self._queue_size:
                try:
                    p_proc = h_d[self._queue_size - to_state.k]
                except IndexError:
                    pass
            else:
                p_proc = 0

            tmp2 += p_arr * p_proc

        transition_probability = tmp + tmp2

        return self._filter_transition_probability_by_action(transition_probability, from_state, to_state, action_id)

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
                    transition_matrix[a][i][j] = \
                        self._calculate_transition_probability(self._states[i], self._states[j], a)
        return transition_matrix
    # ------------------------------------------


    def _generate_reward_matrix(self):
        pass







