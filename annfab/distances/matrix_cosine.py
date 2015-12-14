import numpy as np

import nearpy.distances


class MatrixCosineDistance(nearpy.distances.CosineDistance):
    """
    A distance measure for calculating the cosine distance between matrices.
    """

    def distance(self, x, y):
        if len(x.shape) <= 1:
            return super(MatrixCosineDistance, self).distance(x, y)

        k = x.shape[1]
        if len(y.shape) > 1:
            m = y.shape[1]
        else:
            m = 1

        d = np.empty((k, m), dtype=np.dtype(x[0, 0]))
        for i in xrange(k):
            d[i, :] = super(MatrixCosineDistance, self).distance(x[:, i], y)

        return d
