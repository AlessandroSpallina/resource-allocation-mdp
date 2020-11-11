# PLOTTER MAIN

# questo software è chiamato da slicing_core passandogli il path di results.data
import getopt
import sys

import src.plotter.utils as utils


def cli_handler(argv):
    USAGE = "main.py -d <dataPath>"
    to_return = {}
    try:
        opts, args = getopt.getopt(argv, "hd:", ["data="])
    except getopt.GetoptError:
        print(USAGE)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(USAGE)
            sys.exit()
        elif opt in ('-d', '--data'):
            to_return['data'] = arg
    return to_return


def main(argv):
    DATA_PATH = cli_handler(argv)['data']

    imported_data = utils.import_data(DATA_PATH)

    print(imported_data)


if __name__ == '__main__':
    main(sys.argv[1:])
