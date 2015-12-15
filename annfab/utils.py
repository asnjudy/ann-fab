import numpy as np


def is_1d(x):
    if np.isscalar(x):
        return False

    nd = len(x.shape)

    if nd == 1:
        return True
    elif np.all(np.array(x.shape[1:]) == 1):
        return True

    return False


