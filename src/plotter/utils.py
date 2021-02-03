import json
import pickle


def import_json_data(path):
    with open(path, 'r') as f:
        to_return = f.read()
    to_return = json.loads(to_return)
    return to_return


def import_serialized_data(path):
    return pickle.load(open(path, 'rb'))
