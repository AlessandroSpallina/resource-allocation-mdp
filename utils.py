import colorama as color
from graphviz import Digraph
import matplotlib.pyplot as plt

STORAGE_PATH = "./res/exported/"


def print_blue(message):
    print(f"{color.Fore.BLUE}{message}{color.Style.RESET_ALL}")


# Export to file a graph representing the markov chain related to an action
def plot_markov_chain(projectname, filename, states, transition_matrix, reward_matrix=[], view=False):
    dot = Digraph(filename=filename + ".gv", format="png")

    for i in range(len(states)):
        dot.node(str(i), "S" + str(i) + ": " + str(states[i]))

    for x in range(len(transition_matrix)):
        for y in range(len(transition_matrix)):
            if transition_matrix[x][y] > 0:
                if len(reward_matrix) > 0:
                    dot.edge(str(x), str(y), label=f"P: {transition_matrix[x][y]} [R: {reward_matrix[x][y]}]")
                else:
                    dot.edge(str(x), str(y), label=f"P: {transition_matrix[x][y]}")

    dot.render(STORAGE_PATH + projectname + "/" + filename, view=view)


# def simple_bar_plot(x, y):
#     ax = plt.subplot(111)
#     ax.bar(x, y, width=10)
#
#     plt.show()
#
#
#     plt.bar(x, y)
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.title(title)
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#
#
# def plot_histograms(arrivals_histogram, departures_histogram):
#     x = list(range(len(arrivals_histogram)))
#
#     simple_bar_plot(x, arrivals_histogram, "Arrivals Histogram", "Timeslot", "Arrivals")
#     simple_bar_plot(x, departures_histogram, "Departures Histogram", "Timeslot", "Departures")