import os
import tempfile

import pytest

from ids_db_sql import IDsDB


@pytest.fixture
def ids_db():
    # Create a temporary directory for the test database
    temp_dir = tempfile.mkdtemp()
    db_name = "test_ids_db.sqlite"
    db = IDsDB(db_path=temp_dir, db_name=db_name)
    yield db
    # Cleanup after tests
    os.remove(os.path.join(temp_dir, db_name))
    os.rmdir(temp_dir)


def test_add_ids(ids_db):
    ids_db.add_ids("test_key", ["id1", "id2", "id3"], "test_namespace")
    ids = ids_db.get_ids("test_key", "test_namespace")
    assert ids == ["id1", "id2", "id3"]


def test_add_ids_existing(ids_db):
    ids_db.add_ids("test_key", ["id1", "id2"], "test_namespace")
    ids_db.add_ids("test_key", ["id3", "id4"], "test_namespace")
    ids = ids_db.get_ids("test_key", "test_namespace")
    assert ids == ["id1", "id2", "id3", "id4"]


def test_get_ids_nonexistent(ids_db):
    ids = ids_db.get_ids("nonexistent_key", "nonexistent_namespace")
    assert ids == []


def test_delete_ids(ids_db):
    ids_db.add_ids("test_key", ["id1", "id2"], "test_namespace")
    ids_db.delete_ids("test_key", "test_namespace")
    ids = ids_db.get_ids("test_key", "test_namespace")
    assert ids == []


def test_add_ids_empty_list(ids_db):
    ids_db.add_ids("test_key", [], "test_namespace")
    ids = ids_db.get_ids("test_key", "test_namespace")
    assert ids == []


def test_add_ids_invalid_type(ids_db):
    ids_db.add_ids("test_key", "not_a_list", "test_namespace")
    ids = ids_db.get_ids("test_key", "test_namespace")
    assert ids == []


def test_multiple_namespaces(ids_db):
    ids_db.add_ids("test_key", ["id1", "id2"], "namespace1")
    ids_db.add_ids("test_key", ["id3", "id4"], "namespace2")
    ids1 = ids_db.get_ids("test_key", "namespace1")
    ids2 = ids_db.get_ids("test_key", "namespace2")
    assert ids1 == ["id1", "id2"]
    assert ids2 == ["id3", "id4"]
