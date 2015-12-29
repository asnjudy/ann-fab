import hasher
import _annfab
import numpy as np
import utils


class MatHash(_annfab.RandomBinaryProjection):

    def __init__(self, m, L, batch_size, dim, rand_seed, is_gpu):
        super(MatHash, self).__init__(m*L, batch_size, dim,
                                      rand_seed, is_gpu)
        self._m = m
        self._L = L
        self.hash_name = "hash_%d" % rand_seed

    def reset(self, dim):
        super(MatHash, self).set_dim(dim)

    def get_dim(self):
        return self.dim

    def hash_vector(self, v, querying=False):
        self.set_batch_size(1)

        matrix = v.reshape((1, len(v)))

        bk = self.hash_matrix(matrix)
        bucket_keys = []
        for j in range(0, self._m*self._L, self._m):
            bucket_keys.append(''.join(c for c in bk[0, j:j+self._m]))

        return bucket_keys


class BatchHasher(hasher.Hasher):

    def __init__(self, storage, name, L, m, is_gpu):
        self.is_gpu = is_gpu
        # The batch hasher uses only a single seed.
        hash_seeds = np.random.randint(2**31, size=1)
        dim = 1

        self.__create__(storage, name, L, m, dim, hash_seeds)

    def _create_hasher(self, i, seed):
        # projection_count, batch_size, dim, rand_seed
        return MatHash(self._m, self._L, 1, self._dim, seed, self.is_gpu)

    def allocate_matrices(self):
        self._matrix = np.empty(
            (self._batch_size, self._dim),
            dtype=np.float32)
        self._norm_matrix = np.empty(
            (self._batch_size, self._dim),
            dtype=np.float32)

    def set_batch_size(self, batch_size):
        self._batch_size = batch_size
        self._data = []
        self._next_vector = 0

        self.allocate_matrices()

        for h in self.lshashes:
            h.set_batch_size(self._batch_size)

    def set_dim(self, new_dim):
        for h in self.lshashes:
            h.set_dim(new_dim)

        self._dim = new_dim

        self.allocate_matrices()

    def flush(self):
        # Hash the input matrix for each ot the hashes.
        for lshash in self.lshashes:
            bucket_keys = lshash.hash_matrix(self._matrix)
            for i in range(self._next_vector):

                for j in range(0, self._m*self._L, self._m):
                    bucket_key = ''.join(
                        c for c in bucket_keys[i, j:j+self._m])
                    self.storage.store_vector(lshash.hash_name, bucket_key,
                                              self._norm_matrix[i, :],
                                              self._data[i])

        self._data = []
        self._next_vector = 0

    def store_vector(self, v, data=None):
        if self._next_vector == self._batch_size:
            self.flush()

        # Store the vector and data locally
        self._data.append(data)
        i = self._next_vector
        self._matrix[i, :] = v
        self._norm_matrix[i, :] = utils.unitvec(v)
        self._next_vector += 1
