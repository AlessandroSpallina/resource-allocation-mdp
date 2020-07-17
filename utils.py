import colorama as color
from graphviz import Digraph

STORAGE_PATH = "./res/exported/"


def print_blue(message):
    print(f"{color.Fore.BLUE}{message}{color.Style.RESET_ALL}")


# Export to file a graph representing the markov chain related to an action
def export_markov_chain(projectname, filename, states, transition_matrix):
    dot = Digraph(filename=filename + ".gv", format="png")

    for i in range(len(states)):
        dot.node(str(i), "S" + str(i) + ": " + str(states[i]))

    for x in range(len(transition_matrix)):
        for y in range(len(transition_matrix)):
            if transition_matrix[x][y] > 0:
                dot.edge(str(x), str(y), label=str(transition_matrix[x][y]))

    dot.render(STORAGE_PATH + projectname + "/" + filename, view=False)
