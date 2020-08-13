from graphviz import Digraph
import matplotlib.pyplot as plt
import os

STORAGE_PATH = "./res/exported/"


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

        if not os.path.exists(STORAGE_PATH + projectname + "/"):
            os.makedirs(STORAGE_PATH + projectname + "/")
        dot.render(STORAGE_PATH + projectname + "/" + f"action{a}", view=view)


def plot_cumulative(stuff, title="", xlabel="", ylabel="", projectname="", view=False):
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.grid(True)
    ax.minorticks_on()
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
    if not os.path.exists(STORAGE_PATH + projectname + "/"):
        os.makedirs(STORAGE_PATH + projectname + "/")
    plt.savefig(STORAGE_PATH + projectname + "/" + title)
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
    if not os.path.exists(STORAGE_PATH + projectname + "/"):
        os.makedirs(STORAGE_PATH + projectname + "/")
    plt.savefig(STORAGE_PATH + projectname + "/" + title)
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
    if not os.path.exists(STORAGE_PATH + projectname + "/"):
        os.makedirs(STORAGE_PATH + projectname + "/")
    plt.savefig(STORAGE_PATH + projectname + "/" + title)
    if view:
        plt.show()

