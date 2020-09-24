import os
import subprocess


def config_paths(path="./configs"):
    to_return = []

    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isfile(file_path) and '.yaml' in file_path:
            to_return.append(file_path)
    return to_return


def main():
    paths = config_paths()

    while len(paths) > 0:
        process = subprocess.Popen([''])

if __name__ == '__main__':
    main()
