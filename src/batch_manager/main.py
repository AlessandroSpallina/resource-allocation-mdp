# BATCH MANAGER MAIN

import os
import sys
import time
import getopt


def cli_handler(argv):
    USAGE = "main.py -w <workingDirectory> -c <configDirectory>"
    to_return = {}
    try:
        # help, config (path), name (directory name of the results)
        opts, args = getopt.getopt(argv, "hw:c:", ["wdir=", "config="])
    except getopt.GetoptError:
        print(USAGE)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(USAGE)
            sys.exit()
        elif opt in ('-w', '--wdir'):
            to_return['wdir'] = arg
        elif opt in ('-c', '--config'):
            to_return['config'] = arg

    return to_return


def config_paths(path="./configs"):
    to_return = []

    for file in os.listdir(path):
        file_path = os.path.abspath(os.path.join(path, file))
        if os.path.isfile(file_path) and '.yaml' in file_path:
            to_return.append(file_path.replace('\\', '/'))
    to_return.sort()
    to_return.reverse()
    return to_return


def main(argv):
    # ---- CLI ARGS HANDLING -----------------------
    cli_args = cli_handler(argv)
    print(cli_args)
    if 'wdir' in cli_args:
        os.chdir(cli_args['wdir'])
        print(f"changed working dir to {os.getcwd()}", flush=True)
    if 'config' in cli_args:
        paths = config_paths(path=cli_args['config'])
    else:
        paths = config_paths()
    # ---------------------------------------------

    start_time = int(time.time())

    while len(paths) > 0:
        path = paths.pop()
        os.system(f"cd ../../ && "
                  f"python -m src.slicing_core.main "
                  f"-w ./src/slicing_core/ -c {path} -n b-{start_time}/{path.split('/')[-1].split('.')[0]}")


if __name__ == '__main__':
    main(sys.argv[1:])
