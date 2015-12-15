import numpy as np
import pytest

import annfab.distances
import nearpy.distances


@pytest.fixture(scope="module",
                params=[(0, 0), (0, 1), (1, 0), (1, 1), (0, 5), (1, 5), (6, 0),
                        (6, 1), (6, 5)])
def xy_data(request):
    N = 100
    k = request.param[0]
    m = request.param[1]

    x0 = np.random.randn(N)
    y0 = np.random.randn(N)

    if k == 0:
        x = x0
    else:
        x = np.random.randn(N, k)
        for i in xrange(k):
            x[:, i] = x0

    if m == 0:
        y = y0
    else:
        y = np.random.randn(N, m)
        for i in xrange(m):
            y[:, i] = y0

    data = {'N': N, 'k': k, 'm': m, 'x0': x0, 'y0': y0, 'x': x, 'y': y}

    return data


def test_cosine_matrix_distance_adapter(xy_data):
    vD = nearpy.distances.CosineDistance()
    vd = vD.distance(xy_data['x0'], xy_data['y0'])

    mD = annfab.distances.MatrixDistance(vD)
    md = mD.distance(xy_data['x'], xy_data['y'])

    np.testing.assert_almost_equal(vd, md)


def test_euclidean_matrix_distance_adapter(xy_data):
    vD = nearpy.distances.EuclideanDistance()
    vd = vD.distance(xy_data['x0'], xy_data['y0'])

    mD = annfab.distances.MatrixDistance(vD)
    md = mD.distance(xy_data['x'], xy_data['y'])

    np.testing.assert_almost_equal(vd, md)


def test_manhattan_matrix_distance_adapter(xy_data):
    vD = nearpy.distances.ManhattanDistance()
    vd = vD.distance(xy_data['x0'], xy_data['y0'])

    mD = annfab.distances.MatrixDistance(vD)
    md = mD.distance(xy_data['x'], xy_data['y'])

    np.testing.assert_almost_equal(vd, md)


def test_cosine_matrix_distance(xy_data):
    vD = nearpy.distances.CosineDistance()
    vd = vD.distance(xy_data['x0'], xy_data['y0'])

    mD = annfab.distances.MatrixCosineDistance()
    md = mD.distance(xy_data['x'], xy_data['y'])

    np.testing.assert_almost_equal(vd, md)
