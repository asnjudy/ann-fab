import numpy as np

import annfab.hashes
import nearpy.hashes


def check_length(string, length):
    if string.find('_') >= 0:
        assert len(string.split('_')) == length
    else:
        assert len(string) == length


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
    check_length(rbh_answer[0], 10)
    check_length(mh_answer[0], 10)
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
        check_length(mh_answer[i], 10)
        check_length(rbh_answer[0], 10)
        assert rbh_answer[0] == mh_answer[i]


def get_hash_functions():
    hash_functions = []
    hash_functions.append(nearpy.hashes.RandomBinaryProjections(
        'rand binary proj', 10, 1234))
    hash_functions.append(nearpy.hashes.PCABinaryProjections(
        'pca binary proj', 10, np.random.random((20, 10))))
    hash_functions.append(nearpy.hashes.PCADiscretizedProjections(
        'pca discr proj', 10, np.random.random((20, 10)), 3))
    hash_functions.append(nearpy.hashes.RandomBinaryProjectionTree(
        'rand binary proj tree', 10, 5, 1234))
    hash_functions.append(nearpy.hashes.RandomDiscretizedProjections(
        'rand discr proj', 10, 5, 1234))
    return hash_functions


def test_init_matrix_hash():
    for h in get_hash_functions():
        helptest_init_matrix_hash(h)


def test_hash_for_vector():
    for h in get_hash_functions():
        helptest_hash_for_vector(h)


def test_hash_for_matrix():
    for h in get_hash_functions():
        helptest_hash_for_matrix(h)
