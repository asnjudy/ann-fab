import numpy as np

import nearpy.distances


class MatrixDistance(nearpy.distances.Distance):
    """
    A class for calculating the discance measure between matrices.

    This class wraps a nearpy distance class for the calculation of the
    individual distance elements.
    """

    def __init__(self, scalar_distance):
        self._scalar_distance = scalar_distance.distance

    def distance(self, x, y):
        assert x.shape[0] == y.shape[0]

        x_nd = len(x.shape)
        y_nd = len(y.shape)

        if x_nd == 1 and y_nd == 1:
            d = self._scalar_distance(x, y)
        elif x_nd == 1:
            m = y.shape[1]
            d = np.empty(m, dtype=np.dtype(x[0]))
            for i in range(m):
                d[i] = self._scalar_distance(x, y[:, i])
        elif y_nd == 1:
            k = x.shape[1]
            d = np.empty(k, dtype=np.dtype(x[0, 0]))
            for i in range(k):
                d[i] = self._scalar_distance(x[:, i], y)
        else:
            k = x.shape[1]
            m = y.shape[1]
            d = np.empty((k, m), dtype=np.dtype(x[0, 0]))
            for j in xrange(m):
                for i in xrange(k):
                    d[i, j] = self._scalar_distance(x[:, i], y[:, j])

        return d
