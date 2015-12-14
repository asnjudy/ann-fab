from nearpy.distances.cosine import CosineDistance
from annfab.distances.matrix import MatrixDistance


class MatrixCosineDistance(MatrixDistance):
    """
    A distance measure for calculating the cosine distance between matrices.
    """

    def __init__(self):
        MatrixDistance.__init__(self, CosineDistance)
