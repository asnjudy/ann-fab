import numpy as np

import annfab.hashes
import nearpy.hashes

def test_init_matrix_hash():
    RBH = nearpy.hashes.RandomBinaryProjections('rand binary proj', 10, 1234)
    RBH.reset(20)
    mh  = annfab.hashes.MatrixHash(RBH)

    assert mh is not None
    
def test_hash_for_vector():
    RBH = nearpy.hashes.RandomBinaryProjections('rand binary proj', 10, 1234)
    RBH.reset(20)
    mh  = annfab.hashes.MatrixHash(RBH)
    a = np.random.random(20)
    rbh_answer = RBH.hash_vector(a)
    mh_answer  = mh.hash_vector(a)
    
    assert len(rbh_answer) == 1
    assert len(mh_answer) == 1
    assert rbh_answer[0] == mh_answer[0]
    