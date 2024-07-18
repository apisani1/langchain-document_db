from dotenv import (
    find_dotenv,
    load_dotenv,
)


load_dotenv(find_dotenv(), override=True)

import os
import tempfile
from typing import (
    Any,
    AsyncGenerator,
    Generator,
    List,
)

import pytest
import pytest_asyncio

from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache
from langchain.storage import InMemoryStore
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from load_document import load_document
from multi_vectorstore import (
    MultiVectorStore,
    _chunk,
    _generate_questions,
    _sumarize,
    load_question_chain,
    load_summarize_chain,
)
from test_document_db import InMemoryVectorStore

set_llm_cache(InMemoryCache())
llm = ChatOpenAI()

large_chuk_size = 4000
docs = load_document(
    "../examples/files/state_of_the_union.txt",
    chunk_it=True,
    chunk_size=large_chuk_size,
)


def chunk(doc: Document, **kwargs) -> List[Document]:
    return _chunk(doc, metadata={}, **kwargs)


small_chunk_size = 1000
sub_docs_chuck = []
for doc in docs:
    sub_docs_chuck.extend(chunk(doc, chunk_size=small_chunk_size))

summarize_chain = load_summarize_chain(llm)


def summarize(doc: Document) -> List[Document]:
    return _sumarize(doc, metadata={}, chain=summarize_chain)


sub_docs_summarize = []
for doc in docs:
    sub_docs_summarize.extend(summarize(doc))

question_chain = load_question_chain(llm)


def generate_questions(doc: Document, q: int = 5) -> List[Document]:
    return _generate_questions(doc, metadata={}, chain=question_chain, q=q)


sub_docs_question = []
for doc in docs:
    sub_docs_question.extend(generate_questions(doc, q=2))


def total_len(docs: List[Document]) -> int:
    return sum([len(doc.page_content) for doc in docs])


class InMemoryVectorstoreWithSearch(InMemoryVectorStore):
    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        res = self.store.get(query)
        if res is None:
            return []
        return [res]


@pytest.fixture
def multi_vectorstore() -> Generator[MultiVectorStore, None, None]:
    """MultiVectorStore fixture."""
    # Create a temporary directory for the test database
    temp_dir = tempfile.mkdtemp()

    multi_vectorstore = MultiVectorStore(
        vectorstore=InMemoryVectorstoreWithSearch(),
        docstore=InMemoryStore(),
        ids_db_path=temp_dir,
    )
    yield multi_vectorstore
    # Cleanup after tests
    os.remove(os.path.join(temp_dir, "ids_db.sqlite"))
    os.rmdir(temp_dir)


@pytest_asyncio.fixture
async def amulti_vectorstore() -> AsyncGenerator[MultiVectorStore, None]:
    """MultiVectorStore fixture."""
    # Create a temporary directory for the test database
    temp_dir = tempfile.mkdtemp()

    multi_vectorstore = MultiVectorStore(
        vectorstore=InMemoryVectorstoreWithSearch(),
        docstore=InMemoryStore(),
        ids_db_path=temp_dir,
    )
    yield multi_vectorstore
    # Cleanup after tests
    os.remove(os.path.join(temp_dir, "ids_db.sqlite"))
    os.rmdir(temp_dir)


def test_add_documents_chunk(multi_vectorstore: MultiVectorStore) -> None:
    """Test adding documents to the vectorstore."""
    ids = multi_vectorstore.add_documents(
        docs, functor="chunk", func_kwargs={"chunk_size": small_chunk_size}
    )

    assert len(ids) == len(docs)
    assert len(multi_vectorstore.docstore.store) == len(docs)
    for i, doc in enumerate(docs):
        assert multi_vectorstore.docstore.store[ids[i]] == doc

    assert len(multi_vectorstore.vectorstore.store) == len(sub_docs_chuck)
    assert total_len(multi_vectorstore.vectorstore.store.values()) == total_len(
        sub_docs_chuck
    )


@pytest.mark.requires("aiosqlite")
@pytest.mark.asyncio
async def test_aadd_documents_chunk(multi_vectorstore: MultiVectorStore) -> None:
    """Test adding documents to the vectorstore."""
    ids = await multi_vectorstore.aadd_documents(
        docs, functor="chunk", func_kwargs={"chunk_size": small_chunk_size}
    )

    assert len(ids) == len(docs)
    assert len(multi_vectorstore.docstore.store) == len(docs)
    for i, doc in enumerate(docs):
        assert multi_vectorstore.docstore.store[ids[i]] == doc

    assert len(multi_vectorstore.vectorstore.store) == len(sub_docs_chuck)
    assert total_len(multi_vectorstore.vectorstore.store.values()) == total_len(
        sub_docs_chuck
    )


def test_add_documents_summarize(multi_vectorstore: MultiVectorStore) -> None:
    """Test adding documents to the vectorstore."""
    ids = multi_vectorstore.add_documents(docs, functor="summary", llm=llm)

    assert len(ids) == len(docs)
    assert len(multi_vectorstore.docstore.store) == len(docs)
    for i, doc in enumerate(docs):
        assert multi_vectorstore.docstore.store[ids[i]] == doc

    assert len(multi_vectorstore.vectorstore.store) == len(sub_docs_summarize)
    assert total_len(multi_vectorstore.vectorstore.store.values()) == total_len(
        sub_docs_summarize
    )


@pytest.mark.requires("aiosqlite")
@pytest.mark.asyncio
async def test_aadd_documents_summarize(multi_vectorstore: MultiVectorStore) -> None:
    """Test adding documents to the vectorstore."""
    ids = await multi_vectorstore.aadd_documents(docs, functor="summary", llm=llm)

    assert len(ids) == len(docs)
    assert len(multi_vectorstore.docstore.store) == len(docs)
    for i, doc in enumerate(docs):
        assert multi_vectorstore.docstore.store[ids[i]] == doc

    assert len(multi_vectorstore.vectorstore.store) == len(sub_docs_summarize)
    assert total_len(multi_vectorstore.vectorstore.store.values()) == total_len(
        sub_docs_summarize
    )


def test_add_documents_question(multi_vectorstore: MultiVectorStore) -> None:
    """Test adding documents to the vectorstore."""
    ids = multi_vectorstore.add_documents(
        docs, functor="question", func_kwargs={"q": 2}, llm=llm
    )

    assert len(ids) == len(docs)
    assert len(multi_vectorstore.docstore.store) == len(docs)
    for i, doc in enumerate(docs):
        assert multi_vectorstore.docstore.store[ids[i]] == doc

    assert len(multi_vectorstore.vectorstore.store) == len(sub_docs_question)
    assert total_len(multi_vectorstore.vectorstore.store.values()) == total_len(
        sub_docs_question
    )


@pytest.mark.requires("aiosqlite")
@pytest.mark.asyncio
async def test_aadd_documents_question(multi_vectorstore: MultiVectorStore) -> None:
    """Test adding documents to the vectorstore."""
    ids = await multi_vectorstore.aadd_documents(docs, functor="question", func_kwargs={"q": 2}, llm=llm)

    assert len(ids) == len(docs)
    assert len(multi_vectorstore.docstore.store) == len(docs)
    for i, doc in enumerate(docs):
        assert multi_vectorstore.docstore.store[ids[i]] == doc

    assert len(multi_vectorstore.vectorstore.store) == len(sub_docs_question)
    assert total_len(multi_vectorstore.vectorstore.store.values()) == total_len(
        sub_docs_question
    )
