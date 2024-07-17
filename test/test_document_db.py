import os
import tempfile
from datetime import datetime
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    Dict,
    Generator,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Type,
)
from unittest.mock import patch

import pytest
import pytest_asyncio

from document_db import DocumentDB
from langchain.indexes import (
    SQLRecordManager,
    aindex,
    index,
)
from langchain.indexes._api import (
    _abatch,
    _HashedDocument,
)
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import (
    VST,
    VectorStore,
)


class ToyLoader(BaseLoader):
    """Toy loader that always returns the same documents."""

    def __init__(self, documents: Sequence[Document]) -> None:
        """Initialize with the documents to return."""
        self.documents = documents

    def lazy_load(
        self,
    ) -> Iterator[Document]:
        yield from self.documents

    async def alazy_load(
        self,
    ) -> AsyncIterator[Document]:
        for document in self.documents:
            yield document


class InMemoryVectorStore(VectorStore):
    """In-memory implementation of VectorStore using a dictionary."""

    def __init__(self, permit_upserts: bool = False) -> None:
        """Vector store interface for testing things in memory."""
        self.store: Dict[str, Document] = {}
        self.permit_upserts = permit_upserts

    def delete(self, ids: Optional[Sequence[str]] = None, **kwargs: Any) -> None:
        """Delete the given documents from the store using their IDs."""
        if ids:
            for _id in ids:
                self.store.pop(_id, None)

    async def adelete(self, ids: Optional[Sequence[str]] = None, **kwargs: Any) -> None:
        """Delete the given documents from the store using their IDs."""
        if ids:
            for _id in ids:
                self.store.pop(_id, None)

    def add_documents(  # type: ignore
        self,
        documents: Sequence[Document],
        *,
        ids: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add the given documents to the store (insert behavior)."""
        if ids and len(ids) != len(documents):
            raise ValueError(
                f"Expected {len(ids)} ids, got {len(documents)} documents."
            )

        if not ids:
            raise NotImplementedError("This is not implemented yet.")

        for _id, document in zip(ids, documents):
            if _id in self.store and not self.permit_upserts:
                raise ValueError(
                    f"Document with uid {_id} already exists in the store."
                )
            self.store[_id] = document

        return list(ids)

    async def aadd_documents(
        self,
        documents: Sequence[Document],
        *,
        ids: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        if ids and len(ids) != len(documents):
            raise ValueError(
                f"Expected {len(ids)} ids, got {len(documents)} documents."
            )

        if not ids:
            raise NotImplementedError("This is not implemented yet.")

        for _id, document in zip(ids, documents):
            if _id in self.store and not self.permit_upserts:
                raise ValueError(
                    f"Document with uid {_id} already exists in the store."
                )
            self.store[_id] = document
        return list(ids)

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict[Any, Any]]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add the given texts to the store (insert behavior)."""
        raise NotImplementedError()

    @classmethod
    def from_texts(
        cls: Type[VST],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[Dict[Any, Any]]] = None,
        **kwargs: Any,
    ) -> VST:
        """Create a vector store from a list of texts."""
        raise NotImplementedError()

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Find the most similar documents to the given query."""
        raise NotImplementedError()


@pytest.fixture
def document_db() -> Generator[DocumentDB, None, None]:
    """Document DB fixture."""
    # Create a temporary directory for the test database
    temp_dir = tempfile.mkdtemp()
    document_db = DocumentDB(
        location=temp_dir,
        vectorstore=InMemoryVectorStore(),
        db_url="sqlite:///:memory:",
    )
    yield document_db
    # Cleanup after tests
    document_db.delete_index()


@pytest_asyncio.fixture
async def adocument_db() -> AsyncGenerator[DocumentDB, None]:
    """Document DB fixture."""
    # Create a temporary directory for the test database
    temp_dir = tempfile.mkdtemp()
    document_db = await DocumentDB.ainit(
        location=temp_dir,
        vectorstore=InMemoryVectorStore(),
        db_url="sqlite+aiosqlite:///:memory:",
        async_mode=True,
    )
    yield document_db
    # Cleanup after tests
    document_db.delete_index()


def test_upserting_same_content(document_db: DocumentDB) -> None:
    """Upserting some content to confirm it gets added only once."""
    loader = ToyLoader(
        documents=[
            Document(
                page_content="This is a test document.",
                metadata={"source": "a source"},
            ),
            Document(
                page_content="This is another document.",
                metadata={"source": "a source"},
            ),
        ]
    )

    assert document_db.upsert_documents(loader) == {
        "num_added": 2,
        "num_deleted": 0,
        "num_skipped": 0,
        "num_updated": 0,
    }

    assert len(list(document_db.vectorstore.store)) == 2

    for _ in range(2):
        # Run the indexing again
        assert document_db.upsert_documents(loader) == {
            "num_added": 0,
            "num_deleted": 0,
            "num_skipped": 2,
            "num_updated": 0,
        }


@pytest.mark.requires("aiosqlite")
@pytest.mark.asyncio
async def test_aupserting_same_content(adocument_db: DocumentDB) -> None:
    """Upserting some content to confirm it gets added only once."""
    loader = ToyLoader(
        documents=[
            Document(
                page_content="This is a test document.",
                metadata={"source": "a source"},
            ),
            Document(
                page_content="This is another document.",
                metadata={"source": "a source"},
            ),
        ]
    )

    assert await adocument_db.aupsert_documents(loader) == {
        "num_added": 2,
        "num_deleted": 0,
        "num_skipped": 0,
        "num_updated": 0,
    }

    assert len(list(adocument_db.vectorstore.store)) == 2

    for _ in range(2):
        # Run the indexing again
        assert await adocument_db.aupsert_documents(loader) == {
            "num_added": 0,
            "num_deleted": 0,
            "num_skipped": 2,
            "num_updated": 0,
        }


def test_upsert_fails_with_bad_source_ids(document_db: DocumentDB) -> None:
    """Test upserting fails with bad source ids."""
    loader = ToyLoader(
        documents=[
            Document(
                page_content="This is a test document.",
                metadata={"source": "1"},
            ),
            Document(
                page_content="This is another document.",
                metadata={"source": "2"},
            ),
            Document(
                page_content="This is yet another document.",
                metadata={"source": None},
            ),
        ]
    )

    with pytest.raises(ValueError):
        document_db.upsert_documents(loader)


@pytest.mark.requires("aiosqlite")
@pytest.mark.asyncio
async def test_aupsert_fails_with_bad_source_ids(adocument_db: DocumentDB) -> None:
    """Test upserting fails with bad source ids."""
    loader = ToyLoader(
        documents=[
            Document(
                page_content="This is a test document.",
                metadata={"source": "1"},
            ),
            Document(
                page_content="This is another document.",
                metadata={"source": "2"},
            ),
            Document(
                page_content="This is yet another document.",
                metadata={"source": None},
            ),
        ]
    )

    with pytest.raises(ValueError):
        await adocument_db.aupsert_documents(loader)


def test_upserting_deletes(document_db: DocumentDB) -> None:
    """Test upserting updated documents results in deletion."""
    loader = ToyLoader(
        documents=[
            Document(
                page_content="This is a test document.",
                metadata={"source": "1"},
            ),
            Document(
                page_content="This is another document.",
                metadata={"source": "2"},
            ),
        ]
    )

    with patch.object(
        document_db.record_manager,
        "get_time",
        return_value=datetime(2021, 1, 2).timestamp(),
    ):
        assert document_db.upsert_documents(loader) == {
            "num_added": 2,
            "num_deleted": 0,
            "num_skipped": 0,
            "num_updated": 0,
        }

    doc_texts = set(
        # Ignoring type since doc should be in the store and not a None
        document_db.vectorstore.store.get(uid).page_content  # type: ignore
        for uid in document_db.vectorstore.store
    )
    assert doc_texts == {"This is another document.", "This is a test document."}

    # Attempt to index again verify that nothing changes
    with patch.object(
        document_db.record_manager,
        "get_time",
        return_value=datetime(2021, 1, 2).timestamp(),
    ):
        assert document_db.upsert_documents(loader) == {
            "num_added": 0,
            "num_deleted": 0,
            "num_skipped": 2,
            "num_updated": 0,
        }

    # Create 2 documents from the same source all with mutated content
    loader = ToyLoader(
        documents=[
            Document(
                page_content="mutated document 1",
                metadata={"source": "1"},
            ),
            Document(
                page_content="mutated document 2",
                metadata={"source": "1"},
            ),
            Document(
                page_content="This is another document.",  # <-- Same as original
                metadata={"source": "2"},
            ),
        ]
    )

    # Attempt to index again verify that nothing changes
    with patch.object(
        document_db.record_manager,
        "get_time",
        return_value=datetime(2021, 1, 3).timestamp(),
    ):
        assert document_db.upsert_documents(loader) == {
            "num_added": 2,
            "num_deleted": 1,
            "num_skipped": 1,
            "num_updated": 0,
        }

    doc_texts = set(
        # Ignoring type since doc should be in the store and not a None
        document_db.vectorstore.store.get(uid).page_content  # type: ignore
        for uid in document_db.vectorstore.store
    )
    assert doc_texts == {
        "mutated document 1",
        "mutated document 2",
        "This is another document.",
    }


@pytest.mark.requires("aiosqlite")
@pytest.mark.asyncio
async def test_aupserting_deletes(adocument_db: DocumentDB) -> None:
    """Test upserting updated documents results in deletion."""
    loader = ToyLoader(
        documents=[
            Document(
                page_content="This is a test document.",
                metadata={"source": "1"},
            ),
            Document(
                page_content="This is another document.",
                metadata={"source": "2"},
            ),
        ]
    )

    with patch.object(
        adocument_db.record_manager,
        "get_time",
        return_value=datetime(2021, 1, 2).timestamp(),
    ):
        assert await adocument_db.aupsert_documents(loader) == {
            "num_added": 2,
            "num_deleted": 0,
            "num_skipped": 0,
            "num_updated": 0,
        }

    doc_texts = set(
        # Ignoring type since doc should be in the store and not a None
        adocument_db.vectorstore.store.get(uid).page_content  # type: ignore
        for uid in adocument_db.vectorstore.store
    )
    assert doc_texts == {"This is another document.", "This is a test document."}

    # Attempt to index again verify that nothing changes
    with patch.object(
        adocument_db.record_manager,
        "get_time",
        return_value=datetime(2021, 1, 2).timestamp(),
    ):
        assert await adocument_db.aupsert_documents(loader) == {
            "num_added": 0,
            "num_deleted": 0,
            "num_skipped": 2,
            "num_updated": 0,
        }

    # Create 2 documents from the same source all with mutated content
    loader = ToyLoader(
        documents=[
            Document(
                page_content="mutated document 1",
                metadata={"source": "1"},
            ),
            Document(
                page_content="mutated document 2",
                metadata={"source": "1"},
            ),
            Document(
                page_content="This is another document.",  # <-- Same as original
                metadata={"source": "2"},
            ),
        ]
    )

    # Attempt to index again verify that nothing changes
    with patch.object(
        adocument_db.record_manager,
        "get_time",
        return_value=datetime(2021, 1, 3).timestamp(),
    ):
        assert await adocument_db.aupsert_documents(loader) == {
            "num_added": 2,
            "num_deleted": 1,
            "num_skipped": 1,
            "num_updated": 0,
        }

    doc_texts = set(
        # Ignoring type since doc should be in the store and not a None
        adocument_db.vectorstore.store.get(uid).page_content  # type: ignore
        for uid in adocument_db.vectorstore.store
    )
    assert doc_texts == {
        "mutated document 1",
        "mutated document 2",
        "This is another document.",
    }


def test_upserting_no_docs(document_db: DocumentDB) -> None:
    """Check edge case when loader returns no new docs."""
    loader = ToyLoader(documents=[])

    assert document_db.upsert_documents(loader) == {
        "num_added": 0,
        "num_deleted": 0,
        "num_skipped": 0,
        "num_updated": 0,
    }


@pytest.mark.requires("aiosqlite")
@pytest.mark.asyncio
async def test_aupserting_no_docs(adocument_db: DocumentDB) -> None:
    """Check edge case when loader returns no new docs."""
    loader = ToyLoader(documents=[])

    assert await adocument_db.aupsert_documents(loader) == {
        "num_added": 0,
        "num_deleted": 0,
        "num_skipped": 0,
        "num_updated": 0,
    }


def test_deduplication(document_db: DocumentDB) -> None:
    """Check edge case when loader returns no new docs."""
    docs = [
        Document(
            page_content="This is a test document.",
            metadata={"source": "1"},
        ),
        Document(
            page_content="This is a test document.",
            metadata={"source": "1"},
        ),
    ]

    # Should result in only a single document being added
    assert document_db.upsert_documents(docs) == {
        "num_added": 1,
        "num_deleted": 0,
        "num_skipped": 0,
        "num_updated": 0,
    }


@pytest.mark.requires("aiosqlite")
@pytest.mark.asyncio
async def test_adeduplication(adocument_db: DocumentDB) -> None:
    """Check edge case when loader returns no new docs."""
    docs = [
        Document(
            page_content="This is a test document.",
            metadata={"source": "1"},
        ),
        Document(
            page_content="This is a test document.",
            metadata={"source": "1"},
        ),
    ]

    # Should result in only a single document being added
    assert await adocument_db.aupsert_documents(docs) == {
        "num_added": 1,
        "num_deleted": 0,
        "num_skipped": 0,
        "num_updated": 0,
    }


def test_deduplication_v2(document_db: DocumentDB) -> None:
    """Check edge case when loader returns no new docs."""
    docs = [
        Document(
            page_content="1",
            metadata={"source": "1"},
        ),
        Document(
            page_content="1",
            metadata={"source": "1"},
        ),
        Document(
            page_content="2",
            metadata={"source": "2"},
        ),
        Document(
            page_content="3",
            metadata={"source": "3"},
        ),
    ]

    assert document_db.upsert_documents(docs) == {
        "num_added": 3,
        "num_deleted": 0,
        "num_skipped": 0,
        "num_updated": 0,
    }

    # using in memory implementation here
    assert isinstance(document_db.vectorstore, InMemoryVectorStore)
    contents = sorted(
        [document.page_content for document in document_db.vectorstore.store.values()]
    )
    assert contents == ["1", "2", "3"]

@pytest.mark.requires("aiosqlite")
@pytest.mark.asyncio
async def test_adeduplication_v2(adocument_db: DocumentDB) -> None:
    """Check edge case when loader returns no new docs."""
    docs = [
        Document(
            page_content="1",
            metadata={"source": "1"},
        ),
        Document(
            page_content="1",
            metadata={"source": "1"},
        ),
        Document(
            page_content="2",
            metadata={"source": "2"},
        ),
        Document(
            page_content="3",
            metadata={"source": "3"},
        ),
    ]

    assert await adocument_db.aupsert_documents(docs) == {
        "num_added": 3,
        "num_deleted": 0,
        "num_skipped": 0,
        "num_updated": 0,
    }

    # using in memory implementation here
    assert isinstance(adocument_db.vectorstore, InMemoryVectorStore)
    contents = sorted(
        [document.page_content for document in adocument_db.vectorstore.store.values()]
    )
    assert contents == ["1", "2", "3"]

def test_clean(document_db: DocumentDB) -> None:
    """Test that the clean method functions as expected."""
    docs = [
        Document(
            page_content="1",
            metadata={"source": "1"},
        ),
        Document(
            page_content="1",
            metadata={"source": "1"},
        ),
        Document(
            page_content="2",
            metadata={"source": "2"},
        ),
        Document(
            page_content="3",
            metadata={"source": "3"},
        ),
    ]

    document_db.upsert_documents(docs)

    assert document_db.clean() == {
        "num_added": 0,
        "num_deleted": 3,
        "num_skipped": 0,
        "num_updated": 0,
    }

    # using in memory implementation here
    assert isinstance(document_db.vectorstore, InMemoryVectorStore)

    contents = sorted(
        [document.page_content for document in document_db.vectorstore.store.values()]
    )
    assert contents == []

@pytest.mark.requires("aiosqlite")
@pytest.mark.asyncio
async def test_aclean(adocument_db: DocumentDB) -> None:
    """Test that the clean method functions as expected."""
    docs = [
        Document(
            page_content="1",
            metadata={"source": "1"},
        ),
        Document(
            page_content="1",
            metadata={"source": "1"},
        ),
        Document(
            page_content="2",
            metadata={"source": "2"},
        ),
        Document(
            page_content="3",
            metadata={"source": "3"},
        ),
    ]

    await adocument_db.aupsert_documents(docs)

    assert await adocument_db.aclean() == {
        "num_added": 0,
        "num_deleted": 3,
        "num_skipped": 0,
        "num_updated": 0,
    }

    # using in memory implementation here
    assert isinstance(adocument_db.vectorstore, InMemoryVectorStore)

    contents = sorted(
        [document.page_content for document in adocument_db.vectorstore.store.values()]
    )
    assert contents == []
