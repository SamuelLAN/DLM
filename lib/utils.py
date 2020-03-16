import os
from six.moves import cPickle as pickle

__cur_path = os.path.split(os.path.abspath(__file__))[0]
root_dir = os.path.split(__cur_path)[0]


def load_pkl(_path):
    with open(_path, 'rb') as f:
        return pickle.load(f)


def write_pkl(_path, data):
    with open(_path, 'wb') as f:
        pickle.dump(data, f)


def cache(file_name, data):
    runtime_dir = os.path.join(root_dir, 'runtime')
    if not os.path.exists(runtime_dir):
        os.mkdir(runtime_dir)

    file_path = os.path.join(runtime_dir, file_name)
    write_pkl(file_path, data)


def read_cache(file_name):
    runtime_dir = os.path.join(root_dir, 'runtime')
    if not os.path.exists(runtime_dir):
        return

    file_path = os.path.join(runtime_dir, file_name)
    if not os.path.exists(file_path):
        return
    return load_pkl(file_path)
