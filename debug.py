from slice_mdp import SliceMDP
import plotter


if __name__ == '__main__':

    data1 = [0,1,3,5,3,2,1,4]
    data2 = [1,1,1,1,1,1,1,1]

    plotter.plot_two_scales(data1, data2, ylabel1="jobs in queue", ylabel2="active servers", xlabel="timeslot",
                            projectname="res/exported/debug", view=True, title="ciao")


    # slice_mdp = SliceMDP([0.5, 0.5], [0., 1.], 1, 2, c_lost=10)
    #
    # plotter.plot_markov_chain(slice_mdp.states, slice_mdp.transition_matrix, slice_mdp.reward_matrix,
    #                           projectname="res/exported/debug", view=True)
    #
    #
    # print(slice_mdp.transition_matrix)
    # r = slice_mdp.run_value_iteration(0.8)
    # print(r)
