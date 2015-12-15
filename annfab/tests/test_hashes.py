import numpy as np
import pytest

import annfab.hashes
import nearpy.hashes


@pytest.fixture(scope="module", params=range(6))
def hash_function(request):

    hash_function_map = [
        nearpy.hashes.UniBucket('0123456789'),
        nearpy.hashes.RandomBinaryProjections('rand binary proj', 10, 1234),
        nearpy.hashes.PCABinaryProjections('pca binary proj', 10,
                                           np.random.random((20, 10))),
        nearpy.hashes.PCADiscretizedProjections('pca discr proj', 10,
                                                np.random.random((20, 10)), 3),
        nearpy.hashes.RandomBinaryProjectionTree('rand binary proj tree', 10,
                                                 5, 1234),
        nearpy.hashes.RandomDiscretizedProjections('rand discr proj', 10, 5,
                                                   1234),
    ]



    return hash_function_map[request.param]


def helptest_init_matrix_hash(test_hash):
    test_hash.reset(20)
    mh = annfab.hashes.MatrixHash(test_hash)

    assert mh is not None


def helptest_hash_for_vector(test_hash):
    test_hash.reset(20)
    mh = annfab.hashes.MatrixHash(test_hash)
    a = np.random.random(20)
    rbh_answer = test_hash.hash_vector(a)
    mh_answer = mh.hash_vector(a)

    assert len(rbh_answer) == 1
    assert len(mh_answer) == 1
    assert rbh_answer[0] == mh_answer[0]


def helptest_hash_for_matrix(test_hash):
    test_hash.reset(20)
    mh = annfab.hashes.MatrixHash(test_hash)
    a = np.random.random((5, 20))
    mh_answer = mh.hash_vector(a)
    assert len(mh_answer) == 5
    for i in range(5):
        rbh_answer = test_hash.hash_vector(a[i, :])
        assert len(rbh_answer) == 1
        assert rbh_answer[0] == mh_answer[i]


def test_init_matrix_hash(hash_function):
    helptest_init_matrix_hash(hash_function)


def test_hash_for_vector(hash_function):
    helptest_hash_for_vector(hash_function)


def test_hash_for_matrix(hash_function):
    helptest_hash_for_matrix(hash_function)
