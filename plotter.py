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


# TODO: bar una accanto all'altra, se vedi il numero di stat (rand, mdp -> n=2) puoi fare width/n
# https://python-graph-gallery.com/11-grouped-barplot/
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
    if len(projectname) > 0:
        plt.savefig(projectname + "/" + title)
    else:
        plt.savefig(title)
    if view:
        plt.show()


def scatter(stuff, title="", xlabel="", ylabel="", projectname="", view=False):
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.minorticks_on()
    ax.grid(True)
    for k in stuff:
        ax.scatter(list(range(len(stuff[k]))), stuff[k], label=k)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    if not os.path.exists(projectname + "/"):
        os.makedirs(projectname + "/")
    if len(projectname) > 0:
        plt.savefig(projectname + "/" + title)
    else:
        plt.savefig(title)
    if view:
        plt.show()

# see https://towardsdatascience.com/simple-little-tables-with-matplotlib-9780ef5d0bc4
# def table(row_titles, column_titles, data, title=""):
#     cell_text = []
#     for row in data:
#         cell_text.append([f''])
