import numpy as np

import nearpy.distances


class MatrixDistance(nearpy.distances.Distance):
    def __init__(self, scalar_distance):
        self._scalar_distance = scalar_distance

    def distance(self, x, y):
        if len(x.shape) <= 1:
            return self._scalar_distance.distance(x, y)

        assert x.shape[0] == y.shape[0]

        k = x.shape[1]
        if len(y.shape) > 1:
            m = y.shape[1]
        else:
            m = 1

        d = np.empty((k, m), dtype=np.dtype(x[0, 0]))
        for i in xrange(k):
            d[i, :] = self._scalar_distance.distance(x[:, i], y)

        return d
