from graphviz import Digraph
import matplotlib.pyplot as plt
import numpy as np

STORAGE_PATH = "./res/exported/"


# Export to file a graph representing the markov chain related to an action
def plot_markov_chain(projectname, states, transition_matrix, reward_matrix=None, view=False):

    for a in range(len(transition_matrix)):
        dot = Digraph(filename=f"action{a}" + ".gv", format="png")

        for i in range(len(states)):
            dot.node(str(i), "S" + str(i) + ": " + str(states[i]))

        for x in range(len(transition_matrix[a])):
            for y in range(len(transition_matrix[a])):
                if transition_matrix[a][x][y] > 0:
                    if reward_matrix is not None:
                        dot.edge(str(x), str(y), label=f"P: {transition_matrix[a][x][y]} [R: {reward_matrix[a][x][y]}]")
                    else:
                        dot.edge(str(x), str(y), label=f"P: {transition_matrix[a][x][y]}")

        dot.render(STORAGE_PATH + projectname + "/" + f"action{a}", view=view)

# def plot_cumulative_cost_processed_lost_jobs(costs_per_timeslot, processed_per_timeslot, lost_per_timeslot):
#     cumulative_costs = []
#     cumulative_processed_jobs = []
#     cumulative_lost_jobs = []
#     for i in range(len(costs_per_timeslot)):
#         if i > 0:
#             cumulative_costs.append(cumulative_costs[i-1] + costs_per_timeslot[i])
#             cumulative_processed_jobs.append(cumulative_processed_jobs[i-1] + processed_per_timeslot[i])
#             cumulative_lost_jobs.append(cumulative_lost_jobs[i-1] + lost_per_timeslot[i])
#         else:
#             cumulative_costs.append(costs_per_timeslot[0])
#             cumulative_processed_jobs.append(processed_per_timeslot[0])
#             cumulative_lost_jobs.append(lost_per_timeslot[0])
#
#     fig, ax = plt.subplots()
#
#     ax.set_title("Mean Cumulative Costs")
#     ax.set_xlabel('Timeslot')
#     ax.set_ylabel('Cost')
#     ax.plot(list(range(len(costs_per_timeslot))), cumulative_costs, label="costs")
#     ax.plot(list(range(len(processed_per_timeslot))), cumulative_processed_jobs, label="processed jobs")
#     ax.plot(list(range(len(lost_per_timeslot))), cumulative_lost_jobs, label="lost jobs")
#     ax.legend()
#     plt.show()


def plot_cumulative(stuff, title="", xlabel="", ylabel=""):
    fig, ax = plt.subplots()
    for k in stuff:
        cumulative_buf = []

        for i in range(len(stuff[k])):
            if i > 0:
                cumulative_buf.append(cumulative_buf[i-1] + stuff[k][i])
            else:
                cumulative_buf.append(stuff[k][0])

        ax.plot(list(range(len(stuff[k]))), cumulative_buf, label=k)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    plt.show()


def plot(stuff, title="", xlabel="", ylabel=""):
    fig, ax = plt.subplots()
    for k in stuff:
        ax.plot(list(range(len(stuff[k]))), stuff[k], label=k)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    plt.show()
