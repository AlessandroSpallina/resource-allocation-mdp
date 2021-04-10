import subprocess
import json
import pickle
import time

from src.slicing_core.config import POLICY_CACHE_FILES_PATH


def get_last_commit_link():
    commit_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
    return f"https://github.com/AlessandroSpallina/Slicing-5G-MDP/commit/{commit_hash[:-1].decode('utf-8')}"


def export_data(data, path):
    to_export = json.dumps(data)
    with open(path, 'w') as f:
        f.write(to_export)


def serialize_data(data, path):
    pickle.dump(data, open(path, 'wb'))


class _Cache:
    def __init__(self, config, file_extension):
        self._path = f"{POLICY_CACHE_FILES_PATH}{config.hash}.{file_extension}"

    def load(self, blocking=False):
        loaded = None

        if not blocking:
            try:
                loaded = pickle.load(open(self._path, "rb"))
            except FileNotFoundError:
                pass
        else:
            waiting = True

            while waiting:
                try:
                    loaded = pickle.load(open(self._path, "rb"))
                    waiting = False
                except FileNotFoundError:
                    time.sleep(1)
                    # print(f"Blocking cache.load is waiting {self._path}")

        return loaded

    def store(self, policy):
        pickle.dump(policy, open(self._path, "wb"))
        return self._path
