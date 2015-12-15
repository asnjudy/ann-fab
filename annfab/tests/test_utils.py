import numpy as np

import annfab.utils


def test_scalar_is_not_1d():
    x = 10

    assert not annfab.utils.is_1d(x)


def test_vector_is_1d():
    x = np.random.randn(10)

    assert annfab.utils.is_1d(x)


def test_second_dimension_is_unity_is_1d():
    x = np.random.randn(10, 1)

    assert annfab.utils.is_1d(x)


def test_matrix_is_not_1d():
    x = np.random.randn(10, 2)

    assert not annfab.utils.is_1d(x)


def test_3d_with_unity_is_1d():
    x = np.random.randn(10, 1, 1)

    assert annfab.utils.is_1d(x)


def test_3d_with_unity_and_n_is_not_1d():
    x = np.random.randn(10, 1, 2)

    assert not annfab.utils.is_1d(x)
