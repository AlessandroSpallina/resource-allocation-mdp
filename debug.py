from slice_mdp import SliceMDP
import plotter

if __name__ == '__main__':
    slice_mdp = SliceMDP([0.5, 0.5], [0., 1.], 1, 2, c_lost=10)

    plotter.plot_markov_chain(slice_mdp.states, slice_mdp.transition_matrix, slice_mdp.reward_matrix,
                              projectname="res/exported/debug", view=True)


    print(slice_mdp.transition_matrix)
    r = slice_mdp.run_value_iteration(0.8)
    print(r)
    # e = mdptoolbox.mdp.ValueIteration(matrix, mdp.reward_matrix, 0.8)
    # e.run()
    # print(e.policy)


    # queue size = 1
    # H_a = [0.1, 0.4, 0.4, 0.1]
    # H_p = [0, 1]
    #
    # Q(0,1 -> 0,1) = P(0) + P(1)P(1) + P(2)P(1) + P(3)P(1) = 1.
    # Q(0,1 -> 1,1) = 0

    # ------------

    # queue size = 1
    # H_a = [0.5, 0.5]
    # H_p = [0.6, 0.2, 0.2]
    #
    # Q(0,1 -> 0,1) = P(0) + P(1)P(1) = 0.6
    # Q(0,1 -> 1,1) = P(1)P(0) = 0.3
