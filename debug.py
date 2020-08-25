from slice_mdp import SliceMDP
import plotter

if __name__ == '__main__':
    slice_mdp = SliceMDP([0.5, 0.4, 0.1], [0.6, 0.4], 2, 1, c_lost=10)

    # plotter.plot_markov_chain(slice_mdp.states, slice_mdp.transition_matrix, slice_mdp.reward_matrix,
    #                           projectname="debug", view=True)


    # matrix = mdp.transition_matrix
    # r = mdp.run_value_iteration(0.8)
    # print(r.policy)
    # e = mdptoolbox.mdp.ValueIteration(matrix, mdp.reward_matrix, 0.8)
    # e.run()
    # print(e.policy)
