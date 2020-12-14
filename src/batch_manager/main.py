# BATCH MANAGER MAIN

import os
import subprocess
import sys
import time
import getopt


def cli_handler(argv):
    USAGE = "main.py -w <workingDirectory> -c <configPath> -n <simulationName>"
    to_return = {}
    try:
        # help, config (path), name (directory name of the results)
        opts, args = getopt.getopt(argv, "hw", ["wdir="])
    except getopt.GetoptError:
        print(USAGE)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(USAGE)
            sys.exit()
        elif opt in ('-w', '--wdir'):
            to_return['wdir'] = arg

    return to_return


def config_paths(path="./configs"):
    to_return = []

    for file in os.listdir(path):
        file_path = os.path.abspath(os.path.join(path, file))
        if os.path.isfile(file_path) and '.yaml' in file_path:
            to_return.append(file_path.replace('\\', '/'))
    return to_return


def main(argv):
    # ---- CLI ARGS HANDLING -----------------------
    cli_args = cli_handler(argv)
    if 'wdir' in cli_args:
        os.chdir(cli_args['wdir'])
        print(f"changed working dir to {os.getcwd()}")
    # ---------------------------------------------

    paths = config_paths()
    start_time = int(time.time())

    while len(paths) > 0:
        path = paths.pop()
        os.system(f"cd ../../ && "
                  f"python -m src.slicing_core.main "
                  f"-w ./src/slicing_core/ -c {path} -n 'b-{start_time}/{path.split('/')[-1].split('.')[0]}'")

        # process = subprocess.Popen(['../../venv/Scripts/python', '../slicing_core/main.py',
        #                             '-w', '../slicing_core/',
        #                             '-c', path,
        #                             '-n', f"b-{start_time}/{path.split('/')[-1].split('.')[0]}"])
        # process.communicate()  # this will wait for the process termination


if __name__ == '__main__':
    main(sys.argv[1:])
