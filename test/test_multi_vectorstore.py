from dotenv import (
    find_dotenv,
    load_dotenv,
)


load_dotenv(find_dotenv(), override=True)

import os
import tempfile
from typing import (
    Generator,
    List,
)

import pytest
from scipy.spatial import distance

from langchain.globals import set_llm_cache
from langchain.storage import InMemoryStore
from langchain_community.cache import InMemoryCache
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import (
    ChatOpenAI,
    OpenAIEmbeddings,
)
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
DOC = "../examples/files/state_of_the_union.txt"
LARGE_CHUNK_SIZE = 4000
SMALL_CHUNK_SIZE = 1000
LLM = ChatOpenAI()
SUMMARIZE_CHAIN = load_summarize_chain(LLM)
QUESTION_CHAIN = load_question_chain(LLM)
EMBEDDINGS = OpenAIEmbeddings()
QUERY = "what is the state of the union?"
EMBEDDED_QUERY = EMBEDDINGS.embed_query(QUERY)


def chunk(doc: Document, **kwargs) -> List[Document]:
    return _chunk(doc, metadata={}, **kwargs)


def summarize(doc: Document) -> List[Document]:
    return _sumarize(doc, metadata={}, chain=SUMMARIZE_CHAIN)


def generate_questions(doc: Document, q: int = 5) -> List[Document]:
    return _generate_questions(doc, metadata={}, chain=QUESTION_CHAIN, q=q)


def total_len(docs: List[Document]) -> int:
    return sum([len(doc.page_content) for doc in docs])


@pytest.fixture(scope="function")
def docs() -> List[Document]:
    return load_document(DOC, chunk_it=True, chunk_size=LARGE_CHUNK_SIZE)


@pytest.fixture(scope="function")
def sub_docs_chunk(docs: List[Document]) -> List[Document]:
    sub_docs = []
    for doc in docs:
        sub_docs.extend(chunk(doc, chunk_size=SMALL_CHUNK_SIZE))
    return sub_docs


@pytest.fixture(scope="function")
def sub_docs_summarize(docs: List[Document]) -> List[Document]:
    sub_docs = []
    for doc in docs:
        sub_docs.extend(summarize(doc))
    return sub_docs


@pytest.fixture(scope="function")
def sub_docs_question(docs: List[Document]) -> List[Document]:
    sub_docs = []
    for doc in docs:
        sub_docs.extend(generate_questions(doc, q=2))
    return sub_docs


@pytest.fixture(scope="function")
def sub_docs_distances(sub_docs_chunk: List[Document]) -> List[float]:
    embedded_sub_docs = [
        EMBEDDINGS.embed_query(sub_doc.page_content) for sub_doc in sub_docs_chunk
    ]
    distances = [
        distance.euclidean(EMBEDDED_QUERY, embedded_sub_doc)
        for embedded_sub_doc in embedded_sub_docs
    ]
    distances.sort()
    return distances


@pytest.fixture(scope="function")
def multi_vectorstore() -> Generator[MultiVectorStore, None, None]:
    """MultiVectorStore fixture(scope="function")."""
    # Create a temporary directory for the test database
    temp_dir = tempfile.mkdtemp()

    multi_vectorstore = MultiVectorStore(
        vectorstore=InMemoryVectorStore(),
        docstore=InMemoryStore(),
        ids_db_path=temp_dir,
    )
    yield multi_vectorstore

    # Cleanup after tests
    os.remove(os.path.join(temp_dir, "ids_db.sqlite"))
    os.rmdir(temp_dir)


@pytest.fixture(scope="function")
def multi_vectorstore_with_search() -> Generator[MultiVectorStore, None, None]:
    """MultiVectorStore fixture(scope="function")."""
    # Create a temporary directory for the test database
    temp_dir = tempfile.mkdtemp()

    multi_vectorstore = MultiVectorStore(
        vectorstore=Chroma(embedding_function=EMBEDDINGS),
        docstore=InMemoryStore(),
        ids_db_path=temp_dir,
    )
    yield multi_vectorstore

    # Cleanup after tests
    os.remove(os.path.join(temp_dir, "ids_db.sqlite"))
    os.rmdir(temp_dir)


def test_retriever(
    multi_vectorstore_with_search: MultiVectorStore,
    docs: List[Document],
    sub_docs_distances: List[float],
) -> None:
    multi_vectorstore_with_search.add_documents(
        docs, functor="chunk", func_kwargs={"chunk_size": SMALL_CHUNK_SIZE}
    )
    retriever = multi_vectorstore_with_search.as_retriever(search_kwargs={"k": 1})
    results = retriever.invoke(QUERY)
    assert len(results) == 1
    sub_docs = chunk(results[0], chunk_size=SMALL_CHUNK_SIZE)
    for sub_doc in sub_docs:
        dist = distance.euclidean(
            EMBEDDINGS.embed_query(sub_doc.page_content), EMBEDDED_QUERY
        )
        if dist == pytest.approx(sub_docs_distances[0], abs=1e-3):
            return
    assert False


def test_add_documents_chunk(
    multi_vectorstore: MultiVectorStore,
    docs: List[Document],
    sub_docs_chunk: List[Document],
) -> None:
    """Test adding documents to the vectorstore."""
    ids = multi_vectorstore.add_documents(
        docs, functor="chunk", func_kwargs={"chunk_size": SMALL_CHUNK_SIZE}
    )

    assert len(ids) == len(docs)
    assert len(multi_vectorstore.docstore.store) == len(docs)
    for i, doc in enumerate(docs):
        assert multi_vectorstore.docstore.store[ids[i]] == doc

    assert len(multi_vectorstore.vectorstore.store) == len(sub_docs_chunk)
    assert total_len(multi_vectorstore.vectorstore.store.values()) == total_len(
        sub_docs_chunk
    )


@pytest.mark.requires("aiosqlite")
@pytest.mark.asyncio
async def test_aadd_documents_chunk(
    multi_vectorstore: MultiVectorStore,
    docs: List[Document],
    sub_docs_chunk: List[Document],
) -> None:
    """Test adding documents to the vectorstore."""
    ids = await multi_vectorstore.aadd_documents(
        docs, functor="chunk", func_kwargs={"chunk_size": SMALL_CHUNK_SIZE}
    )

    assert len(ids) == len(docs)
    assert len(multi_vectorstore.docstore.store) == len(docs)
    for i, doc in enumerate(docs):
        assert multi_vectorstore.docstore.store[ids[i]] == doc

    assert len(multi_vectorstore.vectorstore.store) == len(sub_docs_chunk)
    assert total_len(multi_vectorstore.vectorstore.store.values()) == total_len(
        sub_docs_chunk
    )


def test_add_documents_summarize(
    multi_vectorstore: MultiVectorStore,
    docs: List[Document],
    sub_docs_summarize: List[Document],
) -> None:
    """Test adding documents to the vectorstore."""
    ids = multi_vectorstore.add_documents(docs, functor="summary", llm=LLM)

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
async def test_aadd_documents_summarize(
    multi_vectorstore: MultiVectorStore,
    docs: List[Document],
    sub_docs_summarize: List[Document],
) -> None:
    """Test adding documents to the vectorstore."""
    ids = await multi_vectorstore.aadd_documents(docs, functor="summary", llm=LLM)

    assert len(ids) == len(docs)
    assert len(multi_vectorstore.docstore.store) == len(docs)
    for i, doc in enumerate(docs):
        assert multi_vectorstore.docstore.store[ids[i]] == doc

    assert len(multi_vectorstore.vectorstore.store) == len(sub_docs_summarize)
    assert total_len(multi_vectorstore.vectorstore.store.values()) == total_len(
        sub_docs_summarize
    )


def test_add_documents_question(
    multi_vectorstore: MultiVectorStore,
    docs: List[Document],
    sub_docs_question: List[Document],
) -> None:
    """Test adding documents to the vectorstore."""
    ids = multi_vectorstore.add_documents(
        docs, functor="question", func_kwargs={"q": 2}, llm=LLM
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
async def test_aadd_documents_question(
    multi_vectorstore: MultiVectorStore,
    docs: List[Document],
    sub_docs_question: List[Document],
) -> None:
    """Test adding documents to the vectorstore."""
    ids = await multi_vectorstore.aadd_documents(
        docs, functor="question", func_kwargs={"q": 2}, llm=LLM
    )

    assert len(ids) == len(docs)
    assert len(multi_vectorstore.docstore.store) == len(docs)
    for i, doc in enumerate(docs):
        assert multi_vectorstore.docstore.store[ids[i]] == doc

    assert len(multi_vectorstore.vectorstore.store) == len(sub_docs_question)
    assert total_len(multi_vectorstore.vectorstore.store.values()) == total_len(
        sub_docs_question
    )


# Test for add_documents_multiple
def test_add_documents_multiple(
    multi_vectorstore: MultiVectorStore,
    docs: List[Document],
    sub_docs_chunk: List[Document],
    sub_docs_summarize: List[Document],
    sub_docs_question: List[Document],
) -> None:
    func_list = [
        ("chunk", {"chunk_size": SMALL_CHUNK_SIZE}),
        "summary",
        ("question", {"q": 2}),
    ]
    ids = multi_vectorstore.add_documents(docs, functor=func_list, llm=LLM)

    assert len(ids) == len(docs)
    assert len(multi_vectorstore.docstore.store) == len(docs)
    total_subdocs = (
        len(sub_docs_chunk) + len(sub_docs_summarize) + len(sub_docs_question)
    )
    assert len(multi_vectorstore.vectorstore.store) == total_subdocs


# Test for aadd_documents_multiple
@pytest.mark.requires("aiosqlite")
@pytest.mark.asyncio
async def test_aadd_documents_multiple(
    multi_vectorstore: MultiVectorStore,
    docs: List[Document],
    sub_docs_chunk: List[Document],
    sub_docs_summarize: List[Document],
    sub_docs_question: List[Document],
) -> None:
    func_list = [
        ("chunk", {"chunk_size": SMALL_CHUNK_SIZE}),
        "summary",
        ("question", {"q": 2}),
    ]
    ids = await multi_vectorstore.aadd_documents(
        docs, functor=func_list, llm=LLM
    )

    assert len(ids) == len(docs)
    assert len(multi_vectorstore.docstore.store) == len(docs)
    total_subdocs = (
        len(sub_docs_chunk) + len(sub_docs_summarize) + len(sub_docs_question)
    )
    assert len(multi_vectorstore.vectorstore.store) == total_subdocs


# Test for delete method
def test_delete(multi_vectorstore: MultiVectorStore, docs: List[Document]) -> None:
    ids = multi_vectorstore.add_documents(docs)
    assert len(multi_vectorstore.docstore.store) == len(docs)

    # Delete half of the documents
    to_delete = ids[: len(ids) // 2]
    multi_vectorstore.delete(to_delete)

    assert len(multi_vectorstore.docstore.store) == len(docs) - len(to_delete)


# Test for adelete method
@pytest.mark.requires("aiosqlite")
@pytest.mark.asyncio
async def test_adelete(
    multi_vectorstore: MultiVectorStore, docs: List[Document]
) -> None:
    ids = await multi_vectorstore.aadd_documents(docs)
    assert len(multi_vectorstore.docstore.store) == len(docs)

    # Delete half of the documents
    to_delete = ids[: len(ids) // 2]
    await multi_vectorstore.adelete(to_delete)

    assert len(multi_vectorstore.docstore.store) == len(docs) - len(to_delete)


# Test for similarity_search method
def test_similarity_search(
    multi_vectorstore_with_search: MultiVectorStore,
    docs: List[Document],
    sub_docs_distances: List[float],
) -> None:
    multi_vectorstore_with_search.add_documents(
        docs, functor="chunk", func_kwargs={"chunk_size": SMALL_CHUNK_SIZE}
    )
    results = multi_vectorstore_with_search.similarity_search(QUERY, k=2)

    assert len(results) <= 2
    assert len(results) > 0
    for doc in results:
        dist = distance.euclidean(
            EMBEDDINGS.embed_query(doc.page_content), EMBEDDED_QUERY
        )
        assert dist == pytest.approx(
            sub_docs_distances[0], abs=1e-3
        ) or dist == pytest.approx(sub_docs_distances[1], abs=1e-3)


# Test for aasimilarity_search method
@pytest.mark.requires("aiosqlite")
@pytest.mark.asyncio
async def test_asimilarity_search(
    multi_vectorstore_with_search: MultiVectorStore,
    docs: List[Document],
    sub_docs_distances: List[float],
) -> None:
    await multi_vectorstore_with_search.aadd_documents(
        docs, functor="chunk", func_kwargs={"chunk_size": SMALL_CHUNK_SIZE}
    )
    results = await multi_vectorstore_with_search.asimilarity_search(QUERY, k=2)

    assert len(results) <= 2
    assert len(results) > 0
    for doc in results:
        dist = distance.euclidean(
            EMBEDDINGS.embed_query(doc.page_content), EMBEDDED_QUERY
        )
        assert dist == pytest.approx(
            sub_docs_distances[0], abs=1e-3
        ) or dist == pytest.approx(sub_docs_distances[1], abs=1e-3)
