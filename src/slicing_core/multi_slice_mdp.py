import numpy as np
import mdptoolbox

from state import State


# multi-slice support for different kind of slices
# TODO: support for more than two slices (look at _generate_states and _generate_actions, fix it!)
class MultiSliceMDP:
    def __init__(self, slices):
        self._slices = slices

        self._states = self._generate_states()
        self._actions = self._generate_actions()
        self._transition_matrix = self._generate_transition_matrix()
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

    @property
    def actions(self):
        return self._actions

    def _generate_states(self):
        # see https://www.kite.com/python/answers/how-to-get-all-element-combinations-of-two-numpy-arrays-in-python
        mesh = np.array(np.meshgrid(self._slices[0].states, self._slices[1].states))
        to_ret = mesh.T.reshape(-1, 2)
        return to_ret.tolist() # qui fidnme

    def _generate_actions(self):
        tmp = []
        for single_slice_state in self._states[-1]:
            tmp.append(list(range(single_slice_state.n + 1)))

        mesh = np.array(np.meshgrid(tmp[0], tmp[1]))
        to_filter = mesh.T.reshape(-1, 2)

        to_ret = []
        for i in to_filter:
            if sum(i) <= tmp[0][-1]:
                to_ret.append(i)

        return to_ret

    # ---------------------------

    def _calculate_transition_reward(self, to_state):
        # utilizzo approccio "costo di stare nello stato", mi serve solo lo stato di arrivo
        # costs are mapped into the reward matrix
        # C = alpha * C_k * num of jobs + beta * C_n * num of server + gamma * C_l * E(num of lost jobs)
        cost1 = [self._slices[0].c_job * to_state[0].k, self._slices[1].c_job * to_state[1].k]
        cost2 = [self._slices[0].c_server * to_state[0].n, self._slices[1].c_server * to_state[1].n]
        cost3 = [0, 0]

        # expected value of lost packets
        for single_slice_index in range(len(self._slices)):
            for i in range(len(self._slices[single_slice_index].arrivals_histogram)):
                if to_state[single_slice_index].k + i > self._slices[single_slice_index].queue_size:
                    cost3[single_slice_index] += self._slices[single_slice_index].arrivals_histogram[i] * i * self._slices[single_slice_index].c_lost

        return - (
            (self._slices[0].alpha * cost1[0] + self._slices[0].beta * cost2[0] + self._slices[0].gamma * cost3[0]) +
            (self._slices[1].alpha * cost1[1] + self._slices[1].beta * cost2[1] + self._slices[1].gamma * cost3[1])
        )

    def _generate_reward_matrix(self):
        reward_matrix = np.zeros((len(self._actions), len(self._states), len(self._states)))

        for a in range(len(reward_matrix)):
            for i in range(len(self._states)):
                for j in range(len(self._states)):
                    if self._transition_matrix[a][i][j] > 0:
                        reward_matrix[a][i][j] = self._calculate_transition_reward(self._states[j])
        return reward_matrix

    # ----------------------------

    def _calculate_transition_probability(self, from_state, to_state, action):
        transition_probability = \
            self._slices[0].transition_matrix[action[0]][self._slices[0].states.index(from_state[0])][self._slices[0].states.index(to_state[0])] \
            * self._slices[1].transition_matrix[action[1]][self._slices[1].states.index(from_state[1])][self._slices[1].states.index(to_state[1])]

        return transition_probability

    def _generate_transition_matrix(self):
        transition_matrix = np.zeros((len(self._actions), len(self._states), len(self._states)))

        # lets iterate the trans matrix and fill with correct probabilities
        for a in range(len(transition_matrix)):
            for i in range(len(self._states)):
                for j in range(len(self._states)):
                    transition_matrix[a][i][j] = \
                        self._calculate_transition_probability(self._states[i], self._states[j], self._actions[a])
        return transition_matrix

    def _run_finite_horizon(self, discount):
        if type(discount) == list:
            to_return = {}
            for i in discount:
                vi = mdptoolbox.mdp.FiniteHorizon(self._transition_matrix, self._reward_matrix, i - 1e-10, self._slices[0].periods)
                vi.run()
                to_return[f"multi-mdp({str(round(i, 1)).replace('.', ',')})"] = vi.policy
            return to_return

        vi = mdptoolbox.mdp.FiniteHorizon(self._transition_matrix, self._reward_matrix, discount - 1e-10, self._slices[0].periods)
        vi.run()
        return vi.policy

    def run(self, discount):
        return self._run_finite_horizon(discount)







