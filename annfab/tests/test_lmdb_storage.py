import pytest

from annfab.storage import LmdbStorage


class MockLMDB(object):
    def __init__(self, value):
        self.value = value

    def begin(self):
        return self

    def get(self, key):
        return self.value

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def test_store_invalid_vector_asserts():
    db = MockLMDB(None)

    sut = LmdbStorage(db)

    with pytest.raises(Exception):
        sut.store_vector('hash_name', 'bucket_key', 'v', 'key')


def test_store_valid_vector_succeeds():
    db = MockLMDB('vector')

    sut = LmdbStorage(db)
    sut.store_vector('hash_name', 'bucket_key', 'v', 'key')

    assert 'hash_name' in sut.buckets
    assert 'bucket_key' in sut.buckets['hash_name']
    assert (None, 'key') == sut.buckets['hash_name']['bucket_key'][-1]


def test_bucket_is_empty():
    db = MockLMDB(None)

    sut = LmdbStorage(db)

    assert len(sut.get_bucket('hash_name', 'bucket_key')) == 0


def test_get_bucket_sets_value():
    db = MockLMDB('vector')

    sut = LmdbStorage(db)
    sut.store_vector('hash_name', 'bucket_key', 'v', 'key')

    buckets = sut.get_bucket('hash_name', 'bucket_key')

    assert len(buckets) == 1
    assert buckets[0] == ('vector', 'key')


def test_store_vector_maps_value():
    def duplicate(x):
        return x + x

    db = MockLMDB('vector')

    sut = LmdbStorage(db, duplicate)
    sut.store_vector('hash_name', 'bucket_key', 'v', 'key')

    buckets = sut.get_bucket('hash_name', 'bucket_key')

    assert len(buckets) == 1
    assert buckets[0] == ('vectorvector', 'key')
