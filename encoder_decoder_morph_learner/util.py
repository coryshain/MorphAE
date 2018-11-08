import re

def sn(string):
    """
    Compute a Tensorboard-compatible version of a string.

    :param string: ``str``; input string
    :return: ``str``; transformed string
    """

    return re.sub('[^A-Za-z0-9_.\\-/]', '.', string)