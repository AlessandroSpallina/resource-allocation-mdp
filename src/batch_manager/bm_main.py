import os
import subprocess
import time


def config_paths(path="./configs"):
    to_return = []

    for file in os.listdir(path):
        file_path = os.path.abspath(os.path.join(path, file))
        if os.path.isfile(file_path) and '.yaml' in file_path:
            to_return.append(file_path.replace('\\', '/'))
    return to_return


def main():
    paths = config_paths()
    start_time = int(time.time())

    while len(paths) > 0:
        path = paths.pop()
        process = subprocess.Popen(['../../venv/Scripts/python.exe', '../slicing_core/sc_main.py',
                                    '-c', path, '-n', f"b-{start_time}/{path.split('/')[-1].split('.')[0]}"])
        process.communicate()  # this will wait for the process termination


if __name__ == '__main__':
    main()
