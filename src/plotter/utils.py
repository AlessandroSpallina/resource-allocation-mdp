import jsons


def import_data(path):
    with open(path, 'r') as f:
        to_return = f.read(path)
    to_return = jsons.load(to_return)
    return to_return
