import os

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


def plot_two_scales(data1, data2, ylabel1="", ylabel2="", xlabel="", title="", projectname="", view=False):
    fig, ax1 = plt.subplots(figsize=(15, 10))
    ax1.minorticks_on()
    ax1.grid(True)
    ax1.set_title(title)

    t = list(range(max(len(data1), len(data2))))

    color = 'tab:red'
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel1, color=color)
    p1 = ax1.plot(t, data1, color=color, label=ylabel1)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel(ylabel2, color=color)  # we already handled the x-label with ax1
    p2 = ax2.plot(t, data2, color=color, label=ylabel2)
    ax2.tick_params(axis='y', labelcolor=color)

    p = p1 + p2
    labs = [l.get_label() for l in p]
    ax1.legend(p, labs)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    if not os.path.exists(projectname + "/"):
        os.makedirs(projectname + "/")
    if len(projectname) > 0:
        plt.savefig(projectname + "/" + title)
    else:
        plt.savefig(title)
    if view:
        plt.show()


# see https://towardsdatascience.com/simple-little-tables-with-matplotlib-9780ef5d0bc4
def table(row_headers, column_headers, data, title="", projectname="", view=False):
    fig_background_color = 'skyblue'
    fig_border = 'steelblue'

    # Table data needs to be non-numeric text. Format the data
    # while I'm at it.
    cell_text = []
    for row in data:
        cell_text.append([str(x) for x in row])

    # Get some lists of color specs for row and column headers
    rcolors = plt.cm.BuPu([0.1] * len(row_headers))
    ccolors = plt.cm.BuPu([0.1] * len(column_headers))

    plt.figure(linewidth=2,
               edgecolor=fig_border,
               facecolor=fig_background_color,
               tight_layout={'pad': 1},
               figsize=(len(column_headers), len(row_headers)/2)
               )

    # Add a table at the bottom of the axes
    the_table = plt.table(cellText=cell_text,
                          rowLabels=row_headers,
                          rowColours=rcolors,
                          rowLoc='right',
                          colColours=ccolors,
                          colLabels=column_headers,
                          loc='center')

    # Scaling is the only influence we have over top and bottom cell padding.
    # Make the rows taller (i.e., make cell y scale larger).
    the_table.scale(1, 1.5)

    # Hide axes
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Hide axes border
    plt.box(on=None)

    # Add title
    plt.suptitle(title)

    # Force the figure to update, so backends center objects correctly within the figure.
    # Without plt.draw() here, the title will center on the axes and not the figure.
    plt.draw()

    if not os.path.exists(projectname + "/"):
        os.makedirs(projectname + "/")
    if len(projectname) > 0:
        plt.savefig(projectname + "/" + title)
    else:
        plt.savefig(f"{title}.pdf")
    if not view:
        plt.close()

