import os

# import random
import matplotlib.pyplot as plt
from graphviz import Digraph


# Export to file a graph representing the markov chain related to an action
def plot_markov_chain(states, transition_matrix, reward_matrix=None, projectname="", view=False):
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

        if not os.path.exists(projectname + "/"):
            os.makedirs(projectname + "/")
        dot.render(projectname + "/" + f"action{a}", view=view)


def plot_cumulative(stuff, title="", xlabel="", ylabel="", projectname="", view=False):
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.grid(True)
    ax.minorticks_on()
    for k in stuff:
        cumulative_buf = []

        for i in range(len(stuff[k])):
            if i > 0:
                cumulative_buf.append(cumulative_buf[i - 1] + stuff[k][i])
            else:
                cumulative_buf.append(stuff[k][0])

        ax.plot(list(range(len(stuff[k]))), cumulative_buf, label=k)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    if not os.path.exists(projectname + "/"):
        os.makedirs(projectname + "/")
    plt.savefig(projectname + "/" + title)
    if view:
        plt.show()


def plot(stuff, title="", xlabel="", ylabel="", projectname="", view=False):
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.grid(True)
    ax.minorticks_on()
    for k in stuff:
        ax.plot(list(range(len(stuff[k]))), stuff[k], label=k)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    if not os.path.exists(projectname + "/"):
        os.makedirs(projectname + "/")
    plt.savefig(projectname + "/" + title)
    if view:
        plt.show()


def bar(stuff, title="", xlabel="", ylabel="", projectname="", view=False):
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.minorticks_on()
    ax.grid(True)
    for k in stuff:
        ax.bar(list(range(len(stuff[k]))), stuff[k], label=k)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    if not os.path.exists(projectname + "/"):
        os.makedirs(projectname + "/")
    plt.savefig(projectname + "/" + title)
    if view:
        plt.show()


# this function supports multiple y axes
# def plot_cumulative(stuff, title="", xlabel="", projectname="", view=False):
#     fig, ax = plt.subplots(figsize=(15, 10))
#     ax.grid(True)
#
#     # see https://matplotlib.org/3.1.1/gallery/ticks_and_spines/multiple_yaxis_with_spines.html
#     is_host = True
#
#     ax.minorticks_on()
#     for k in stuff:
#         cumulative_buf = []
#
#         for i in range(len(stuff[k])):
#             if i > 0:
#                 cumulative_buf.append(cumulative_buf[i - 1] + stuff[k][i])
#             else:
#                 cumulative_buf.append(stuff[k][0])
#
#         if is_host:
#             is_host = False
#             tmp, = ax.plot(list(range(len(stuff[k]))), cumulative_buf, label=k)
#             ax.set_ylim(0, cumulative_buf[-1])
#             ax.set_ylabel(k)
#         else:
#             part = ax.twinx()
#             tmp, = part.plot(list(range(len(stuff[k]))), cumulative_buf, label=k)
#             part.set_ylim(0, cumulative_buf[-1])
#             part.set_ylabel(k)
#
#     ax.set_title(title)
#     ax.set_xlabel(xlabel)
#     ax.legend()
#     if not os.path.exists(projectname + "/"):
#         os.makedirs(projectname + "/")
#     plt.savefig(projectname + "/" + title)
#     if view:
#         plt.show()

# def _make_patch_spines_invisible(ax):
#     ax.set_frame_on(True)
#     ax.patch.set_visible(False)
#     for sp in ax.spines.values():
#         sp.set_visible(False)

# see https://matplotlib.org/3.1.1/gallery/ticks_and_spines/multiple_yaxis_with_spines.html

# def plot_cumulative_multiple_yaxes(stuff, title="", xlabel="", projectname="", view=False):
#     fig, host = plt.subplots()
#     fig.subplots_adjust(right=0.75)
#     parts = [host]
#
#     for i in range(len(stuff) - 1):
#         parts.append(host.twinx())
#
#     parts[-1].spines["right"].set_position(("axes", 1.2))
#     _make_patch_spines_invisible(parts[-1])
#     parts[-1].spines["right"].set_visible(True)
#
#     p_i = []
#     cumulatives = []
#
#     for k in stuff:
#         cumulative_buf = []
#
#         for i in range(len(stuff[k])):
#             if i > 0:
#                 cumulative_buf.append(cumulative_buf[i - 1] + stuff[k][i])
#             else:
#                 cumulative_buf.append(stuff[k][0])
#
#         cumulatives.append(cumulative_buf)
#
#     tkw = dict(size=(len(parts) + 1), width=1.5)
#     for i in range(len(parts)):
#         tmp, = parts[i].plot(list(range(len(stuff[k]))), cumulatives[i], "0.5", label=k)
#         p_i.append(tmp)
#         parts[i].yaxis.label.set_color(p_i[i].get_color())
#         parts[i].set_ylim(0, cumulatives[i][-1])
#         parts[i].set_ylabel(k)
#
#         parts[i].tick_params(axis='y', colors=p_i[i].get_color, **tkw)
#
#     host.tick_params(axis='x', **tkw)
#     host.set_xlabel(xlabel)
#     host.set_title(title)
#
#     host.legend(p_i, [l.get_label() for l in p_i])
#
#     if not os.path.exists(projectname + "/"):
#         os.makedirs(projectname + "/")
#     plt.savefig(projectname + "/" + title)
#     if view:
#         plt.show()
