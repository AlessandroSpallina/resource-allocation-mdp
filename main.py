from slice_mdp import SliceMDP
from slice_simulator import SliceSimulator
import utils

if __name__ == '__main__':
    arrivals = [0.5, 0.5]
    departures = [0.6, 0.4]

    # @findme : generare grafico partenze e arrivi

    slice_mdp = SliceMDP(arrivals, departures, 2, 1, alpha=0.5)
    states = slice_mdp.states

    utils.plot_markov_chain("toy", "a0-do-nothing", slice_mdp.states,
                            slice_mdp.transition_matrix[0], slice_mdp.reward_matrix[0], view=True)
    utils.plot_markov_chain("toy", "a1-alloc1", slice_mdp.states,
                            slice_mdp.transition_matrix[1], slice_mdp.reward_matrix[1], view=True)
    utils.plot_markov_chain("toy", "a2-dealloc1", slice_mdp.states,
                            slice_mdp.transition_matrix[2], slice_mdp.reward_matrix[1], view=True)

    #print(slice_mdp.reward_matrix)

    policy = slice_mdp.run_value_iteration(0.8)

    slice_simulator = SliceSimulator(arrivals, departures)

    action = 0
    for i in range(100):
        current_state = slice_simulator.simulate_timeslot(action, True)
        print(f"RETURNED {current_state}")
        action = policy[states.index(current_state)]


