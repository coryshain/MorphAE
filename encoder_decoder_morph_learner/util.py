import re
import numpy as np
import pickle


def sn(string):
    """
    Compute a Tensorboard-compatible version of a string.

    :param string: ``str``; input string
    :return: ``str``; transformed string
    """

    return re.sub('[^A-Za-z0-9_.\\-/]', '.', string)


def get_random_permutation(n):
    p = np.random.permutation(np.arange(n))
    p_inv = np.zeros_like(p)
    p_inv[p] = np.arange(n)
    return p, p_inv


def load_edml(dir_path):
    """
    Convenience method for reconstructing a saved EDML object. First loads in metadata from ``m.obj``, then uses
    that metadata to construct the computation graph. Then, if saved weights are found, these are loaded into the
    graph.

    :param dir_path: Path to directory containing the EDML checkpoint files.
    :return: The loaded EDML instance.
    """

    with open(dir_path + '/m.obj', 'rb') as f:
        m = pickle.load(f)
    m.build(outdir=dir_path)
    m.load(outdir=dir_path)
    return m