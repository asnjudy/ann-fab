import numpy as np
import pytest

import annfab.filters
import nearpy.filters


@pytest.fixture(scope="module",
                params=[(0, False),
                        (0, True),
                        (1, False),
                        (1, True),
                        (5, False),
                        (5, True), ])
def test_data(request):
    N = 10
    k = request.param[0]
    has_dist = request.param[1]
    m = 0
    il0 = []
    il = []

    mat = np.empty((k, N))
    d = np.empty(k)
    dist = np.empty(k)
    for i in xrange(k):
        if has_dist:
            item = (np.ones(N) * (i + 1), i + 1, i + 1)
            dist[i] = item[2]
        else:
            item = (np.ones(N) * (i + 1), i + 1)

        il0.append(item)
        mat[i, :] = item[0]
        d[i] = item[1]

    if k > 0:
        if has_dist:
            il.append((mat, d, dist))
        else:
            il.append((mat, d))

    data = {'dist': has_dist, 'N': N, 'k': k, 'm': m, 'il0': il0, 'il': il}

    return data


def test_expand_item_scalars():
    items = (1, 1)
    mF = annfab.filters.MatrixFilter(nearpy.filters.VectorFilter)
    expanded = mF.expand_item(items)

    assert len(expanded) == 1
    assert expanded[0] == items


def test_expand_item_vector():
    items = (np.zeros(4), 1)
    mF = annfab.filters.MatrixFilter(nearpy.filters.VectorFilter)
    expanded = mF.expand_item(items)

    assert len(expanded) == 1
    assert expanded[0] == items


def test_matrix_unique_filter_adapter(test_data):
    vF = nearpy.filters.UniqueFilter()
    vf = vF.filter_vectors(test_data['il0'])

    mF = annfab.filters.MatrixFilter(vF)
    mf = mF.filter_vectors(test_data['il'])

    assert vf is not None
    for i in xrange(test_data['k']):
        assert all(vf[i][0] == mf[i][0])
        assert vf[i][1] == mf[i][1]


def test_matrix_filter_expand_items(test_data):
    mF = annfab.filters.MatrixFilter(nearpy.filters.VectorFilter)

    k = test_data['k']
    if k > 0:
        item = test_data['il'][0]
        expanded = mF.expand_item(item)
        assert k == len(expanded)


def test_matrix_filter_expand(test_data):

    mF = annfab.filters.MatrixFilter(nearpy.filters.VectorFilter)
    expanded = mF.expand(test_data['il'])

    assert len(expanded) == len(test_data['il0'])

    for i in xrange(test_data['k']):
        assert all(expanded[i][0] == test_data['il0'][i][0])
        assert expanded[i][1] == test_data['il0'][i][1]
