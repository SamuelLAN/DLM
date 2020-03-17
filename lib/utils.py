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


def create_dir_in_root(*args):
    """ create directory under the root dir """
    dir_path = root_dir
    for arg in args:
        dir_path = os.path.join(dir_path, arg)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
    return dir_path


def cache(file_name, data):
    """ cache data in the root_dir/runtime/cache """
    file_path = os.path.join(create_dir_in_root('runtime', 'cache'), file_name)
    write_pkl(file_path, data)


def read_cache(file_name):
    """ read data from cache in the root_dir/runtime/cache """
    file_path = os.path.join(create_dir_in_root('runtime', 'cache'), file_name)
    if not os.path.exists(file_path):
        return
    return load_pkl(file_path)


def mkdir_time(upper_path, _time):
    """ create directory with time (for save model) """
    dir_path = os.path.join(upper_path, _time)
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    return dir_path
