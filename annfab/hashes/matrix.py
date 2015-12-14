import numpy as np

import nearpy.hashes

class MatrixHash(nearpy.hashes.LSHash):
    def __init__(self, scalar_hash):
        self._scalar_hash = scalar_hash

    def hash_vector(self, v, querying=False):
        if len(v.shape) <= 1:
            return self._scalar_hash.hash_vector(v, querying)

        d = []
        for i in xrange(v.shape[0]):
            d.append(self._scalar_hash.hash_vector(v[i,:], querying))

        return d
    
    def reset(self, dim):
        self._scalar_hash.reset(dim)

