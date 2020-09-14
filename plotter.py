import os

import matplotlib.pyplot as plt
# import mpld3
from graphviz import Digraph


# def _save_fig_html(fig, filename):
#     text = mpld3.fig_to_html(fig)
#
#     with open(filename, 'w') as f:
#         f.write(text)


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


def plot_cumulative(ydata, xdata=[], title="", xlabel="", ylabel="", projectname="", multiple_plots=False, view=False):
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.grid(True)
    ax.minorticks_on()
    final_values = []
    for k in ydata:
        cumulative_buf = []
        for i in range(len(ydata[k])):
            if i > 0:
                cumulative_buf.append(cumulative_buf[i - 1] + ydata[k][i])
            else:
                cumulative_buf.append(ydata[k][0])
        final_values.append((cumulative_buf[-1], k))
        ax.plot(xdata if len(xdata) else list(range(len(cumulative_buf))), cumulative_buf, label=k)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    if not os.path.exists(projectname + "/"):
        os.makedirs(projectname + "/")
    plt.savefig(projectname + "/" + title)
    # _save_fig_html(fig, f"{projectname}/0_{title}.html")
    if view:
        plt.show()
    plt.close(fig)

    # find subgroup of observations by the final value and avoid overplotting (a plot unreadable due to different scales)
    if multiple_plots:
        groups = []
        final_values.sort()
        for value in final_values:
            for i in range(len(groups)):
                if (value[0] / groups[i][0][0]) <= 1.3:  # magic number: if the ratio is more than 30% then is a different group!
                    groups[i].append(value)
                    break
            else:
                groups.append([value])
        if len(groups) > 1:
            for i in range(len(groups)):
                to_plot = {}
                for e in groups[i]:
                    to_plot[e[1]] = ydata[e[1]]
                plot_cumulative(to_plot, xdata=xdata, title=f"{title} ({i+1}of{len(groups)})", xlabel=xlabel,
                                ylabel=ylabel, projectname=projectname, view=view)


def plot(ydata, xdata=[], title="", xlabel="", ylabel="", projectname="", view=False):
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.grid(True)
    ax.minorticks_on()
    for k in ydata:
        ax.plot(xdata if len(xdata) else list(range(len(ydata[k]))), ydata[k], label=k)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    if not os.path.exists(projectname + "/"):
        os.makedirs(projectname + "/")
    plt.savefig(projectname + "/" + title)
    # _save_fig_html(fig, f"{projectname}/0_{title}.html")
    if view:
        plt.show()
    plt.close(fig)


# TODO: bar una accanto all'altra, se vedi il numero di stat (rand, mdp -> n=2) puoi fare width/n
# https://python-graph-gallery.com/11-grouped-barplot/
def bar(ydata, xdata=[], title="", xlabel="", ylabel="", projectname="", view=False):
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.minorticks_on()
    ax.grid(True)
    for k in ydata:
        ax.bar(xdata if len(xdata) else list(range(len(ydata[k]))), ydata[k], label=k)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    if not os.path.exists(projectname + "/"):
        os.makedirs(projectname + "/")
    if len(projectname) > 0:
        plt.savefig(projectname + "/" + title)
        # _save_fig_html(fig, f"{projectname}/0_{title}.html")
    else:
        plt.savefig(title)
        # _save_fig_html(fig, f"0_{title}.html")
    if view:
        plt.show()
    plt.close(fig)


def scatter(ydata, xdata=[], title="", xlabel="", ylabel="", projectname="", view=False):
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.minorticks_on()
    ax.grid(True)
    for k in ydata:
        ax.scatter(xdata if len(xdata) else list(range(len(ydata[k]))), ydata[k], label=k)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    if not os.path.exists(projectname + "/"):
        os.makedirs(projectname + "/")
    if len(projectname) > 0:
        plt.savefig(projectname + "/" + title)
        # _save_fig_html(fig, f"{projectname}/0_{title}.html")
    else:
        plt.savefig(title)
        # _save_fig_html(fig, f"0_{title}.html")
    if view:
        plt.show()
    plt.close(fig)


def plot_two_scales(data1, data2, xdata=[], ylabel1="", ylabel2="", xlabel="", title="", projectname="", view=False):
    fig, ax1 = plt.subplots(figsize=(15, 10))
    ax1.minorticks_on()
    ax1.grid(True)
    ax1.set_title(title)

    t = list(range(max(len(data1), len(data2))))

    color = 'tab:red'
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel1, color=color)
    p1 = ax1.plot(xdata if len(xdata) else t, data1, color=color, label=ylabel1)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel(ylabel2, color=color)  # we already handled the x-label with ax1
    p2 = ax2.plot(xdata if len(xdata) else t, data2, color=color, label=ylabel2)
    ax2.tick_params(axis='y', labelcolor=color)

    p = p1 + p2
    labs = [l.get_label() for l in p]
    ax1.legend(p, labs)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    if not os.path.exists(projectname + "/"):
        os.makedirs(projectname + "/")
    if len(projectname) > 0:
        plt.savefig(projectname + "/" + title)
        # _save_fig_html(fig, f"{projectname}/0_{title}.html")
    else:
        plt.savefig(title)
        # _save_fig_html(fig, f"0_{title}.html")
    if view:
        plt.show()
    plt.close(fig)


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
               figsize=(len(column_headers), len(row_headers)/2.5)
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
        plt.savefig(f"{title}")
    if not view:
        plt.close()

