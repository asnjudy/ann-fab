import numpy as np

import annfab.distances
import nearpy.distances


def test_init_matrix_distance():
    d = annfab.distances.MatrixCosineDistance()

    assert d is not None


def test_matrix_distance_for_vector_is_distance():
    n = 100
    x = np.random.randn(n)
    y = np.random.randn(n)

    vD = nearpy.distances.CosineDistance()
    vd = vD.distance(x, y)

    mD = annfab.distances.MatrixCosineDistance()
    md = mD.distance(x, y)

    assert vd == md


def test_matrix_distance_y_is_matrix():
    n = 100
    m = 10
    x = np.random.randn(n)
    y0 = np.random.randn(n)

    y = np.empty((n,m))

    for i in xrange(m):
        y[:, i] = y0

    vD = nearpy.distances.CosineDistance()
    vd = vD.distance(x, y0)

    mD = annfab.distances.MatrixCosineDistance()
    md = mD.distance(x, y)

    assert len(md) == m

    for i in xrange(m):
        assert md[i] == vd
