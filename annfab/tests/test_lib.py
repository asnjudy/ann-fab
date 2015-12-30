# flake8: noqa
import numpy as np
import pytest
_annfab = pytest.importorskip("annfab._annfab")


def test_import():
    assert True


def test_simple_cpu():
    batch_size = 2
    input_dim = 5
    projection_count = 4
    seed = 12
    rbp = _annfab.RandomBinaryProjection(
        projection_count, batch_size, input_dim, seed, False)
    n = batch_size * input_dim
    a = np.arange(n, dtype=np.float32).reshape(batch_size, input_dim)
    assert rbp.hash_matrix(a).shape == (
        batch_size, projection_count)


@pytest.mark.xfail
def test_simple_gpu():
    batch_size = 2
    input_dim = 5
    projection_count = 4
    seed = 12
    rbp = _annfab.RandomBinaryProjection(
        projection_count, batch_size, input_dim, seed, True)
    n = batch_size * input_dim
    a = np.arange(n, dtype=np.float32).reshape(batch_size, input_dim)
    assert rbp.hash_matrix(a).shape == (
        batch_size, projection_count)
