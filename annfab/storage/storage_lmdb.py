from nearpy.storage.storage_memory import MemoryStorage


def noop(input):
    return input


class LmdbStorage(MemoryStorage):
    """
    Only keys are stored in the storage, with the actual vector data kept
    in an LMDB database.
    """

    def __init__(self, lmdb_env, data_conversion=None):
        super(LmdbStorage, self).__init__()
        self.lmdb_env = lmdb_env
        if data_conversion:
            self.data_conversion = data_conversion
        else:
            self.data_conversion = noop

    def _data_exists(self, data):
        with self.lmdb_env.begin() as txn:
            return txn.get(data) is not None

    def _get_bucket_item(self, data):
        assert self._data_exists(data)

        with self.lmdb_env.begin() as txn:
            value = txn.get(data)
            assert value is not None

            return self.data_conversion(value), data

    def store_vector(self, hash_name, bucket_key, v, data):
        """
        Stores vector and JSON-serializable data in bucket with specified key.
        """

        # TODO: Ensure that the vector exists in the database
        assert self._data_exists(data)

        # Store only the data (not the vector) in memory.
        super(LmdbStorage, self).store_vector(hash_name, bucket_key, None,
                                              data)

    def get_bucket(self, hash_name, bucket_key):
        """
        Returns bucket content as list of tuples (vector, data).
        """
        buckets = super(LmdbStorage, self).get_bucket(hash_name, bucket_key)

        if not buckets:
            return buckets

        for i in range(len(buckets)):
            # For each entry in the bucket, replace the None vector with the
            # correct one from the lmdb database.
            buckets[i] = self._get_bucket_item(buckets[i][1])

        return buckets
