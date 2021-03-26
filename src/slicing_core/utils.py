import subprocess
import json
import pickle

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

    def load(self):
        try:
            loaded = pickle.load(open(self._path, "rb"))
        except FileNotFoundError:
            loaded = None
        return loaded

    def store(self, policy):
        pickle.dump(policy, open(self._path, "wb"))
        return self._path
