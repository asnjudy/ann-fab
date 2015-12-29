# An example to use LSF hashing to find similar images in the database.
import numpy as np

import nearpy
import nearpy.utils.utils

import pickle


def min_m(eps, delta, r1, r2):
    return 8

    m = np.log(np.log(eps) / np.log(1 - delta)) / np.log((1 - r1) / (1 - r2))
    return int(np.ceil(m))


def calc_L_m(eps, delta, r1, r2):
    m = 1

    while m < 100:
        lower_L = -np.log(eps) / (1 - r1)**m
        upper_L = np.log(1.0 - delta) / np.log(1.0 - (1.0 - r2)**m)

        L = int(np.ceil(lower_L))
        print m, lower_L, L, upper_L
        if L <= upper_L:
            break
        m += 1

    assert L <= upper_L

    return L, m


class Hasher(nearpy.Engine):

    def __init__(self, storage, name=None, L=None, m=None):
        # Generate random seeds for the Random Binary projectsions used as
        # hashes.
        hash_seeds = np.random.randint(2**31, size=L)
        dim = 1

        self.__create__(storage, name, L, m, dim, hash_seeds)

    def _generate_hash_seeds(self, L):
        return np.random.randint(2**31, size=L)

    def __create__(self, storage, name, L, m, dim, hash_seeds):
        self._name = name
        self._dim = dim
        self._L = L
        self._m = m
        self._hash_seeds = hash_seeds

        hashes = []
        for i in range(len(hash_seeds)):
            hashes.append(self._create_hasher(i, hash_seeds[i]))
        # Initialise the super.
        super(Hasher, self).__init__(dim, lshashes=hashes, storage=storage)

    def _create_hasher(self, i, seed):
        hash_name = 'hash_%d_%s' % (i, str(seed))
        return nearpy.hashes.RandomBinaryProjections(hash_name,
                                                     self._m,
                                                     rand_seed=seed)

    def set_dim(self, new_dim):
        for h in self.lshashes:
            h.reset(new_dim)
        self._dim = new_dim

    def add_image(self, key, image):
        print "key=%s" % key
        if len(image) != self._dim:
            self.set_dim(len(image))

        self.store_vector(image, data=key)

    def setup_distance_and_filter(self, k_nearest):
        self.distance = nearpy.distances.CosineDistance()
        self.vector_filters = [nearpy.filters.NearestFilter(k_nearest), ]

    def get_neighbours(self, key, value):
        results = self.neighbours(value)
        return results

    def get_config(self):
        return {
            'name': self._name,
            'L': self._L,
            'm': self._m,
            'dim': self._dim,
            'seeds': self._hash_seeds,
            'storage': self.storage.get_config(),
        }

    def default_filename(self):
        return '%s_%d_%d.out' % (self._name, self._L, self._m)

    def save_to_file(self, filename):
        if filename is None:
            use_filename = self.default_filename()
        else:
            use_filename = filename

        config = self.get_config()
        with open(use_filename, 'w') as f:
            pickle.dump(config, f)

    def load_from_file(self, filename, storage):
        with open(filename, 'r') as f:
            config = pickle.load(f)

        self.clean_all_buckets()

        name = config['name']
        L = config['L']
        m = config['m']
        dim = config['dim']
        seeds = config['seeds']

        print config['name']
        print config['L']
        print config['m']
        print config['dim']
        print config['seeds']

        self.__create__(storage, name, L, m, dim, seeds)

        self.storage.apply_config(config['storage'])
