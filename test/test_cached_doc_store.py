import os
import pickle
import tempfile
from typing import Generator

import pytest

from langchain.docstore.document import Document
from langchain_core.stores import InvalidKeyException
from redis import Redis

from cached_docstore import CachedDocStore


doc1 = Document(
    page_content="This is the first document.",
    metadata={"a": 1, "b": "two", "c": True},
)

doc2 = Document(
    page_content="This is the second document.",
    metadata={"a": 2, "b": "three", "c": False},
)


@pytest.fixture
def doc_store() -> Generator[CachedDocStore, None, None]:
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Instantiate the CachedDocStore with the temporary directory as the root path
        store = CachedDocStore(temp_dir, cached=False)
        yield store


@pytest.fixture
def cached_doc_store() -> Generator[CachedDocStore, None, None]:
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Instantiate the CachedDocStore with the temporary directory as the root path
        store = CachedDocStore(temp_dir, cached=True)
        yield store


@pytest.fixture
def redis_client() -> Redis:
    """Yield redis client."""
    import redis

    # Using standard port, but protecting against accidental data loss
    # by requiring a password.
    # This fixture flushes the database!
    # The only role of the password is to prevent users from accidentally
    # deleting their data.
    # The password should establish the identity of the server being.
    port = 6379
    # password = os.environ.get("REDIS_PASSWORD") or str(uuid.uuid4())
    # client = redis.Redis(host="localhost", port=port, password=password, db=0)
    client = redis.Redis(host="localhost", port=port, db=7)
    try:
        client.ping()
    except redis.exceptions.ConnectionError:
        pytest.skip(
            "Redis server is not running or is not accessible. "
            "Verify that credentials are correct. "
        )
    # ATTENTION: This will delete all keys in the database!
    client.flushdb()
    return client


def test_mset_and_mget(doc_store: CachedDocStore) -> None:
    # Set values for keys
    key_value_pairs = [("key1", doc1), ("key2", doc2)]
    doc_store.mset(key_value_pairs)

    # Get values for keys
    values = doc_store.mget(["key1", "key2"])

    # Assert that the retrieved values match the original values
    assert values == [doc1, doc2]


def test_mset_and_mget_cached(cached_doc_store: CachedDocStore) -> None:
    # Set values for keys
    key_value_pairs = [("key1", doc1), ("key2", doc2)]
    cached_doc_store.mset(key_value_pairs)

    # Get values for keys
    values = cached_doc_store.mget(["key1", "key2"])

    # Assert that the retrieved values match the original values
    assert values == [doc1, doc2]


@pytest.mark.parametrize(
    "chmod_dir_s, chmod_file_s", [("777", "666"), ("770", "660"), ("700", "600")]
)
def test_mset_chmod(chmod_dir_s: str, chmod_file_s: str) -> None:
    chmod_dir = int(chmod_dir_s, base=8)
    chmod_file = int(chmod_file_s, base=8)

    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Instantiate the CachedDocStore with a directory inside the temporary directory
        # as the root path
        temp_dir = os.path.join(temp_dir, "store_dir")
        doc_store = CachedDocStore(
            temp_dir, cached=False, chmod_dir=chmod_dir, chmod_file=chmod_file
        )

        # Set values for keys
        key_value_pairs = [("key1", doc1), ("key2", doc2)]
        doc_store.mset(key_value_pairs)

        # verify the permissions are set correctly
        # (test only the standard user/group/other bits)
        dir_path = str(doc_store.root_path)
        file_path = os.path.join(dir_path, "key1")
        assert (os.stat(dir_path).st_mode & 0o777) == chmod_dir
        assert (os.stat(file_path).st_mode & 0o777) == chmod_file


@pytest.mark.parametrize(
    "chmod_dir_s, chmod_file_s", [("777", "666"), ("770", "660"), ("700", "600")]
)
def test_mset_chmod_cached(chmod_dir_s: str, chmod_file_s: str) -> None:
    chmod_dir = int(chmod_dir_s, base=8)
    chmod_file = int(chmod_file_s, base=8)

    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Instantiate the CachedDocStore with a directory inside the temporary directory
        # as the root path
        temp_dir = os.path.join(temp_dir, "store_dir")
        cached_doc_store = CachedDocStore(
            temp_dir, cached=True, chmod_dir=chmod_dir, chmod_file=chmod_file
        )

        # Set values for keys
        key_value_pairs = [("key1", doc1), ("key2", doc2)]
        cached_doc_store.mset(key_value_pairs)

        # verify the permissions are set correctly
        # (test only the standard user/group/other bits)
        dir_path = str(cached_doc_store.root_path)
        file_path = os.path.join(dir_path, "key1")
        assert (os.stat(dir_path).st_mode & 0o777) == chmod_dir
        assert (os.stat(file_path).st_mode & 0o777) == chmod_file


def test_mget_update_atime() -> None:
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Instantiate the CachedDocStore with a directory inside the temporary directory
        # as the root path
        temp_dir = os.path.join(temp_dir, "store_dir")
        doc_store = CachedDocStore(temp_dir, cached=False, update_atime=True)

        # Set values for keys
        key_value_pairs = [("key1", doc1), ("key2", doc2)]
        doc_store.mset(key_value_pairs)

        # Get original access time
        dir_path = str(doc_store.root_path)
        file_path = os.path.join(dir_path, "key1")
        atime1 = os.stat(file_path).st_atime

        # Get values for keys
        _ = doc_store.mget(["key1", "key2"])

        # Make sure the filesystem access time has been updated
        atime2 = os.stat(file_path).st_atime
        assert atime2 != atime1


def test_mget_update_atime_cached() -> None:
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Instantiate the CachedDocStore with a directory inside the temporary directory
        # as the root path
        temp_dir = os.path.join(temp_dir, "store_dir")
        cached_doc_store = CachedDocStore(temp_dir, cached=True, update_atime=True)

        # Set values for keys
        key_value_pairs = [("key1", doc1), ("key2", doc2)]
        cached_doc_store.mset(key_value_pairs)

        # Get original access time
        dir_path = str(cached_doc_store.root_path)
        file_path = os.path.join(dir_path, "key1")
        atime1 = os.stat(file_path).st_atime

        # Get values for keys
        _ = cached_doc_store.mget(["key1", "key2"])

        # Make sure the filesystem access time has been updated
        atime2 = os.stat(file_path).st_atime
        assert atime2 != atime1


def test_mdelete(doc_store: CachedDocStore) -> None:
    # Set values for keys
    key_value_pairs = [("key1", doc1), ("key2", doc2)]
    doc_store.mset(key_value_pairs)

    # Delete keys
    doc_store.mdelete(["key1"])

    # Check if the deleted key is present
    values = doc_store.mget(["key1"])

    # Assert that the value is None after deletion
    assert values == [None]


def test_mdelete_cached(cached_doc_store: CachedDocStore) -> None:
    # Set values for keys
    key_value_pairs = [("key1", doc1), ("key2", doc2)]
    cached_doc_store.mset(key_value_pairs)

    # Delete keys
    cached_doc_store.mdelete(["key1"])

    # Check if the deleted key is present
    values = cached_doc_store.mget(["key1"])

    # Assert that the value is None after deletion
    assert values == [None]


def test_set_invalid_key(doc_store: CachedDocStore) -> None:
    """Test that an exception is raised when an invalid key is set."""
    # Set a key-value pair
    key = "crying-cat/😿"
    value = Document("This is a test value")
    with pytest.raises(InvalidKeyException):
        doc_store.mset([(key, value)])


def test_set_invalid_key_cached(cached_doc_store: CachedDocStore) -> None:
    """Test that an exception is raised when an invalid key is set."""
    # Set a key-value pair
    key = "crying-cat/😿"
    value = Document("This is a test value")
    with pytest.raises(InvalidKeyException):
        cached_doc_store.mset([(key, value)])


def test_set_key_and_verify_content(doc_store: CachedDocStore) -> None:
    """Test that the content of the file is the same as the value set."""
    # Set a key-value pair
    key = "test_key"
    value = Document("This is a test value")
    doc_store.mset([(key, value)])

    # Verify the content of the actual file
    full_path = doc_store._get_full_path(key)
    assert full_path.exists()
    assert full_path.read_bytes() == pickle.dumps(Document("This is a test value"))


def test_set_key_and_verify_content_cached(cached_doc_store: CachedDocStore) -> None:
    """Test that the content of the file is the same as the value set."""
    # Set a key-value pair
    key = "test_key"
    value = Document("This is a test value")
    cached_doc_store.mset([(key, value)])

    # Verify the content of the actual file
    full_path = cached_doc_store._get_full_path(key)
    assert full_path.exists()
    assert full_path.read_bytes() == pickle.dumps(Document("This is a test value"))


def test_yield_keys(doc_store: CachedDocStore) -> None:
    # Set values for keys
    key_value_pairs = [("key1", doc1), ("subdir/key2", doc2)]
    doc_store.mset(key_value_pairs)

    # Iterate over keys
    keys = list(doc_store.yield_keys())

    # Assert that the yielded keys match the expected keys
    expected_keys = ["key1", os.path.join("subdir", "key2")]
    assert keys == expected_keys


def test_yield_keys_cached(cached_doc_store: CachedDocStore) -> None:
    # Set values for keys
    key_value_pairs = [("key1", doc1), ("subdir/key2", doc2)]
    cached_doc_store.mset(key_value_pairs)

    # Iterate over keys
    keys = list(cached_doc_store.yield_keys())

    # Assert that the yielded keys match the expected keys
    expected_keys = ["key1", os.path.join("subdir", "key2")]
    assert keys == expected_keys


def test_catches_forbidden_keys(doc_store: CachedDocStore) -> None:
    """Make sure we raise exception on keys that are not allowed; e.g., absolute path"""
    with pytest.raises(InvalidKeyException):
        doc_store.mset([("/etc", doc1)])
    with pytest.raises(InvalidKeyException):
        list(doc_store.yield_keys("/etc/passwd"))
    with pytest.raises(InvalidKeyException):
        doc_store.mget(["/etc/passwd"])

    # check relative paths
    with pytest.raises(InvalidKeyException):
        list(doc_store.yield_keys(".."))

    with pytest.raises(InvalidKeyException):
        doc_store.mget(["../etc/passwd"])

    with pytest.raises(InvalidKeyException):
        doc_store.mset([("../etc", doc1)])

    with pytest.raises(InvalidKeyException):
        list(doc_store.yield_keys("../etc/passwd"))


def test_catches_forbidden_keys_cached(cached_doc_store: CachedDocStore) -> None:
    """Make sure we raise exception on keys that are not allowed; e.g., absolute path"""
    with pytest.raises(InvalidKeyException):
        cached_doc_store.mset([("/etc", doc1)])
    with pytest.raises(InvalidKeyException):
        list(cached_doc_store.yield_keys("/etc/passwd"))
    with pytest.raises(InvalidKeyException):
        cached_doc_store.mget(["/etc/passwd"])

    # check relative paths
    with pytest.raises(InvalidKeyException):
        list(cached_doc_store.yield_keys(".."))

    with pytest.raises(InvalidKeyException):
        cached_doc_store.mget(["../etc/passwd"])

    with pytest.raises(InvalidKeyException):
        cached_doc_store.mset([("../etc", doc1)])

    with pytest.raises(InvalidKeyException):
        list(cached_doc_store.yield_keys("../etc/passwd"))


def test_redis_mget(redis_client: Redis) -> None:
    """Test mget method."""
    store = CachedDocStore(cached=True, redis_client=redis_client, ttl=None)
    keys = ["key1", "key2"]
    redis_client.mset({"key1": pickle.dumps(doc1), "key2": pickle.dumps(doc2)})
    result = store.mget(keys)
    assert result == [doc1, doc2]


def test_redis_mset(redis_client: Redis) -> None:
    """Test that multiple keys can be set."""
    store = CachedDocStore(cached=True, redis_client=redis_client, ttl=None)
    key_value_pairs = [("key1", doc1), ("key2", doc2)]
    store.mset(key_value_pairs)
    result = [pickle.loads(raw) for raw in redis_client.mget(["key1", "key2"])]
    assert result == [doc1, doc2]


def test_redis_mdelete(redis_client: Redis) -> None:
    """Test that deletion works as expected."""
    store = CachedDocStore(cached=True, redis_client=redis_client, ttl=None)
    keys = ["key1", "key2"]
    redis_client.mset({"key1": pickle.dumps(doc1), "key2": pickle.dumps(doc2)})
    store.mdelete(keys)
    result = redis_client.mget(keys)
    assert result == [None, None]
