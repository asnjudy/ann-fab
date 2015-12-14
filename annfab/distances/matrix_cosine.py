import nearpy.distances


class MatrixCosineDistance(nearpy.distances.CosineDistance):
    """
    A distance measure for calculating the cosine distance between matrices.
    """

    def distance(self, x, y):
        return super(MatrixCosineDistance, self).distance(x, y)
