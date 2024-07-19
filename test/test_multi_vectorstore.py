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
llm = ChatOpenAI()
embeddings = OpenAIEmbeddings()

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


query = "what is the state of the union?"
embedded_query = embeddings.embed_query(query)
embedded_sub_docs = [embeddings.embed_query(doc.page_content) for doc in sub_docs_chuck]
sub_docs_distances = [
    distance.euclidean(embedded_query, embedded_sub_doc)
    for embedded_sub_doc in embedded_sub_docs
]
sub_docs_distances.sort()
print(sub_docs_distances)


@pytest.fixture
def multi_vectorstore() -> Generator[MultiVectorStore, None, None]:
    """MultiVectorStore fixture."""
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


@pytest.fixture
def multi_vectorstore_with_search() -> Generator[MultiVectorStore, None, None]:
    """MultiVectorStore fixture."""
    # Create a temporary directory for the test database
    temp_dir = tempfile.mkdtemp()

    multi_vectorstore = MultiVectorStore(
        vectorstore=Chroma(embedding_function=embeddings),
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
    ids = await multi_vectorstore.aadd_documents(
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


# Test for add_documents_multiple
def test_add_documents_multiple(multi_vectorstore: MultiVectorStore) -> None:
    func_list = [
        ("chunk", {"chunk_size": small_chunk_size}),
        "summary",
        ("question", {"q": 2}),
    ]
    ids = multi_vectorstore.add_documents_multiple(docs, func_list=func_list, llm=llm)

    assert len(ids) == len(docs)
    assert len(multi_vectorstore.docstore.store) == len(docs)
    total_subdocs = (
        len(sub_docs_chuck) + len(sub_docs_summarize) + len(sub_docs_question)
    )
    assert len(multi_vectorstore.vectorstore.store) == total_subdocs


# Test for aadd_documents_multiple
@pytest.mark.requires("aiosqlite")
@pytest.mark.asyncio
async def test_aadd_documents_multiple(multi_vectorstore: MultiVectorStore) -> None:
    func_list = [
        ("chunk", {"chunk_size": small_chunk_size}),
        "summary",
        ("question", {"q": 2}),
    ]
    ids = await multi_vectorstore.aadd_documents_multiple(
        docs, func_list=func_list, llm=llm
    )

    assert len(ids) == len(docs)
    assert len(multi_vectorstore.docstore.store) == len(docs)
    total_subdocs = (
        len(sub_docs_chuck) + len(sub_docs_summarize) + len(sub_docs_question)
    )
    assert len(multi_vectorstore.vectorstore.store) == total_subdocs


# Test for delete method
def test_delete(multi_vectorstore: MultiVectorStore) -> None:
    ids = multi_vectorstore.add_documents(docs)
    assert len(multi_vectorstore.docstore.store) == len(docs)

    # Delete half of the documents
    to_delete = ids[: len(ids) // 2]
    multi_vectorstore.delete(to_delete)

    assert len(multi_vectorstore.docstore.store) == len(docs) - len(to_delete)


# Test for adelete method
@pytest.mark.requires("aiosqlite")
@pytest.mark.asyncio
async def test_adelete(multi_vectorstore: MultiVectorStore) -> None:
    ids = await multi_vectorstore.aadd_documents(docs)
    assert len(multi_vectorstore.docstore.store) == len(docs)

    # Delete half of the documents
    to_delete = ids[: len(ids) // 2]
    await multi_vectorstore.adelete(to_delete)

    assert len(multi_vectorstore.docstore.store) == len(docs) - len(to_delete)


# Test for similarity_search method
def test_similarity_search(multi_vectorstore_with_search: MultiVectorStore) -> None:
    multi_vectorstore_with_search.add_documents(
        docs, functor="chunk", func_kwargs={"chunk_size": small_chunk_size}
    )
    results = multi_vectorstore_with_search.similarity_search(query, k=2)

    assert len(results) <= 2
    assert len(results) > 0
    for doc in results:
        dist = distance.euclidean(
            embeddings.embed_query(doc.page_content), embedded_query
        )
        assert dist == pytest.approx(
            sub_docs_distances[0], abs=1e-5
        ) or dist == pytest.approx(sub_docs_distances[1], abs=1e-5)


# Test for aasimilarity_search method
@pytest.mark.requires("aiosqlite")
@pytest.mark.asyncio
async def test_asimilarity_search(
    multi_vectorstore_with_search: MultiVectorStore,
) -> None:
    await multi_vectorstore_with_search.aadd_documents(
        docs, functor="chunk", func_kwargs={"chunk_size": small_chunk_size}
    )
    results = await multi_vectorstore_with_search.asimilarity_search(query, k=2)

    assert len(results) <= 2
    assert len(results) > 0
    for doc in results:
        dist = distance.euclidean(
            embeddings.embed_query(doc.page_content), embedded_query
        )
        print(doc)
        assert dist == pytest.approx(
            sub_docs_distances[0], abs=1e-5
        ) or dist == pytest.approx(sub_docs_distances[1], abs=1e-5)
