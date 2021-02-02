import subprocess
import json
import orjson


def get_last_commit_link():
    commit_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
    return f"https://github.com/AlessandroSpallina/Slicing-5G-MDP/commit/{commit_hash[:-1].decode('utf-8')}"


def export_data(data, path):
    to_export = orjson.dumps(data)
    with open(path, 'w') as f:
        f.write(to_export)

