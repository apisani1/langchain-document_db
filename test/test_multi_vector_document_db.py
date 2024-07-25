from dotenv import (
    find_dotenv,
    load_dotenv,
)


load_dotenv(find_dotenv(), override=True)

import tempfile
from typing import (
    AsyncGenerator,
    Dict,
    Generator,
    List,
)

import pytest
import pytest_asyncio

from langchain_core.documents import Document

from load_document import load_document
from multi_vector_document_db import MultiVectorDocumentDB
from multi_vectorstore import _chunk
from test_document_db import InMemoryVectorStore


DOCS = [
    "../examples/files/state_of_the_union.txt",
    "../examples/files/us_constitution.pdf",
]
LARGE_CHUNK_SIZE = 4000
SMALL_CHUNK_SIZE = 1000


def chunk(doc: Document, **kwargs) -> List[Document]:
    return _chunk(doc, metadata={}, **kwargs)


def total_len(docs: List[Document]) -> int:
    return sum([len(doc.page_content) for doc in docs])


@pytest.fixture(scope="function")
def docs() -> List[Document]:
    chunks = []
    for doc_path in DOCS:
        chunks.extend(
            load_document(
                doc_path,
                chunk_it=True,
                chunk_size=LARGE_CHUNK_SIZE,
                chunk_overlap=LARGE_CHUNK_SIZE * 0.1,
            )
        )
    return chunks


@pytest.fixture(scope="function")
def docs_per_source(docs: List[Document]) -> Dict:
    docs_per_source = {}
    for doc in docs:
        if doc.metadata["source"] not in docs_per_source:
            docs_per_source[doc.metadata["source"]] = []
        docs_per_source[doc.metadata["source"]].append(doc)
    return docs_per_source


@pytest.fixture(scope="function")
def sub_docs_chunk(docs: List[Document]) -> List[Document]:
    sub_docs = []
    for doc in docs:
        sub_docs.extend(
            chunk(
                doc, chunk_size=SMALL_CHUNK_SIZE, chunk_overlap=SMALL_CHUNK_SIZE * 0.1
            )
        )
    return sub_docs


@pytest.fixture(scope="function")
def document_db() -> Generator[MultiVectorDocumentDB, None, None]:
    """Document DB fixture."""
    # Create a temporary directory for the test database
    temp_dir = tempfile.mkdtemp()
    document_db = MultiVectorDocumentDB(
        location=temp_dir,
        vectorstore=InMemoryVectorStore(),
        db_url="sqlite:///:memory:",
        functor="chunk",
        func_kwargs={
            "chunk_size": SMALL_CHUNK_SIZE,
            "chunk_overlap": SMALL_CHUNK_SIZE * 0.1,
        }
    )
    yield document_db
    # Cleanup after tests
    document_db.delete_index()


@pytest_asyncio.fixture
async def adocument_db() -> AsyncGenerator[MultiVectorDocumentDB, None]:
    """Document DB fixture."""
    # Create a temporary directory for the test database
    temp_dir = tempfile.mkdtemp()
    adocument_db = await MultiVectorDocumentDB.ainit(
        location=temp_dir,
        vectorstore=InMemoryVectorStore(),
        db_url="sqlite+aiosqlite:///:memory:",
        functor="chunk",
        func_kwargs={
            "chunk_size": SMALL_CHUNK_SIZE,
            "chunk_overlap": SMALL_CHUNK_SIZE * 0.1,
        }
    )
    yield adocument_db
    # Cleanup after tests
    adocument_db.delete_index()


def test_upserting_same_content(
    document_db: MultiVectorDocumentDB,
    docs: List[Document],
    sub_docs_chunk: List[Document],
) -> None:
    """Upserting some content to confirm it gets added only once."""
    assert document_db.upsert_documents(docs) == {
        "num_added": len(docs),
        "num_deleted": 0,
        "num_skipped": 0,
        "num_updated": 0,
    }

    assert len(
        document_db.vectorstore.docstore.mget(
            document_db.vectorstore.docstore.yield_keys()
        )
    ) == len(docs)
    assert total_len(
        document_db.vectorstore.docstore.mget(
            document_db.vectorstore.docstore.yield_keys()
        )
    ) == total_len(docs)
    assert len(document_db.vectorstore.vectorstore.store) == len(sub_docs_chunk)
    assert total_len(document_db.vectorstore.vectorstore.store.values()) == total_len(
        sub_docs_chunk
    )

    for doc in docs:
        doc.metadata.pop("id")

    # Insert the same content again, verify it doesn't get added again
    assert document_db.upsert_documents(docs) == {
        "num_added": 0,
        "num_deleted": 0,
        "num_skipped": len(docs),
        "num_updated": 0,
    }


@pytest.mark.requires("aiosqlite")
@pytest.mark.asyncio
async def test_aupserting_same_content(
    adocument_db: MultiVectorDocumentDB,
    docs: List[Document],
    sub_docs_chunk: List[Document],
) -> None:
    """Upserting some content to confirm it gets added only once."""
    assert await adocument_db.aupsert_documents(docs) == {
        "num_added": len(docs),
        "num_deleted": 0,
        "num_skipped": 0,
        "num_updated": 0,
    }

    assert len(
        adocument_db.vectorstore.docstore.mget(
            adocument_db.vectorstore.docstore.yield_keys()
        )
    ) == len(docs)
    assert total_len(
        adocument_db.vectorstore.docstore.mget(
            adocument_db.vectorstore.docstore.yield_keys()
        )
    ) == total_len(docs)
    assert len(list(adocument_db.vectorstore.vectorstore.store)) == len(sub_docs_chunk)
    assert total_len(
        list(adocument_db.vectorstore.vectorstore.store.values())
    ) == total_len(sub_docs_chunk)

    for doc in docs:
        doc.metadata.pop("id")

    # Insert the same content again, verify it doesn't get added again
    assert await adocument_db.aupsert_documents(docs) == {
        "num_added": 0,
        "num_deleted": 0,
        "num_skipped": len(docs),
        "num_updated": 0,
    }


def test_upserting_deletes(
    document_db: MultiVectorDocumentDB, docs: List[Document], docs_per_source: Dict
) -> None:
    """Test upserting updated documents results in deletion."""
    assert document_db.upsert_documents(docs) == {
        "num_added": len(docs),
        "num_deleted": 0,
        "num_skipped": 0,
        "num_updated": 0,
    }

    for doc in docs:
        print(doc.metadata)
        doc.metadata.pop("id")

    # Keep 1 document and create 2 documents from the same source all with mutated content
    docs2 = [
        docs[0],
        Document(
            page_content="mutated document 1",
            metadata={"source": docs[0].metadata["source"]},
        ),
        Document(
            page_content="mutated document 2",
            metadata={"source": docs[0].metadata["source"]},
        ),
    ]

    # Upsert the new documents and verify that the old documents are deleted
    assert document_db.upsert_documents(docs2) == {
        "num_added": 2,
        "num_deleted": len(docs_per_source[docs[0].metadata["source"]]) - 1,
        "num_skipped": 1,
        "num_updated": 0,
    }


@pytest.mark.requires("aiosqlite")
@pytest.mark.asyncio
async def test_aupserting_deletes(
    adocument_db: MultiVectorDocumentDB, docs: List[Document], docs_per_source: Dict
) -> None:
    """Test upserting updated documents results in deletion."""
    assert await adocument_db.aupsert_documents(docs) == {
        "num_added": len(docs),
        "num_deleted": 0,
        "num_skipped": 0,
        "num_updated": 0,
    }

    for doc in docs:
        doc.metadata.pop("id")

    # Keep 1document and create 2 documents from the same source all with mutated content
    docs2 = [
        docs[0],
        Document(
            page_content="mutated document 2",
            metadata={"source": docs[0].metadata["source"]},
        ),
        Document(
            page_content="This is another document.",  # <-- Same as original
            metadata={"source": docs[0].metadata["source"]},
        ),
    ]

    # Upsert the new documents and verify that the old documents are deleted
    assert await adocument_db.aupsert_documents(docs2) == {
        "num_added": 2,
        "num_deleted": len(docs_per_source[docs[0].metadata["source"]]) - 1,
        "num_skipped": 1,
        "num_updated": 0,
    }


def test_deduplication(
    document_db: MultiVectorDocumentDB, docs: List[Document]
) -> None:
    """Check that duplicates are not added."""
    more_docs = docs + [docs[0]]

    # Should result in only a single copy of each document being added
    assert document_db.upsert_documents(more_docs) == {
        "num_added": len(docs),
        "num_deleted": 0,
        "num_skipped": 0,
        "num_updated": 0,
    }


@pytest.mark.requires("aiosqlite")
@pytest.mark.asyncio
async def test_adeduplication(
    adocument_db: MultiVectorDocumentDB, docs: List[Document]
) -> None:
    """Check that duplicates are not added."""
    more_docs = docs + [docs[0]]

    # Should result in only a single copy of each document being added
    assert await adocument_db.aupsert_documents(more_docs) == {
        "num_added": len(docs),
        "num_deleted": 0,
        "num_skipped": 0,
        "num_updated": 0,
    }


def test_delete(
    document_db: MultiVectorDocumentDB, docs: List[Document], docs_per_source: Dict
) -> None:
    """Test that the delete method functions as expected."""
    assert document_db.upsert_documents(docs) == {
        "num_added": len(docs),
        "num_deleted": 0,
        "num_skipped": 0,
        "num_updated": 0,
    }

    to_delete = docs[0].metadata["source"]
    assert document_db.delete_documents([to_delete]) == {
        "num_added": 1,
        "num_deleted": len(docs_per_source[to_delete]),
        "num_skipped": 0,
        "num_updated": 0,
    }

    contents = sorted(
        [
            document.page_content
            for document in document_db.vectorstore.vectorstore.store.values()
            if document.metadata["source"] == to_delete
        ]
    )
    assert contents == ["Deleted DO NOT USE"]


@pytest.mark.requires("aiosqlite")
@pytest.mark.asyncio
async def test_adelete(
    adocument_db: MultiVectorDocumentDB, docs: List[Document], docs_per_source: Dict
) -> None:
    """Test that the delete method functions as expected."""
    assert await adocument_db.aupsert_documents(docs) == {
        "num_added": len(docs),
        "num_deleted": 0,
        "num_skipped": 0,
        "num_updated": 0,
    }

    to_delete = docs[0].metadata["source"]
    assert await adocument_db.adelete_documents([to_delete]) == {
        "num_added": 1,
        "num_deleted": len(docs_per_source[to_delete]),
        "num_skipped": 0,
        "num_updated": 0,
    }

    contents = sorted(
        [
            document.page_content
            for document in adocument_db.vectorstore.vectorstore.store.values()
            if document.metadata["source"] == to_delete
        ]
    )
    assert contents == ["Deleted DO NOT USE"]
