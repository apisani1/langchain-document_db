from typing import (
    Any,
    List,
)

import pytest
import pytest_asyncio

from langchain.storage import InMemoryStore
from langchain_core.documents import Document
from multi_vectorstore import MultiVectorStore
from test_document_db import InMemoryVectorStore


class InMemoryVectorstoreWithSearch(InMemoryVectorStore):
    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        res = self.store.get(query)
        if res is None:
            return []
        return [res]


def test_multi_vector_retriever_initialization() -> None:
    vectorstore = InMemoryVectorstoreWithSearch()
    retriever = MultiVectorRetriever(  # type: ignore[call-arg]
        vectorstore=vectorstore, docstore=InMemoryStore(), doc_id="doc_id"
    )
    documents = [Document(page_content="test document", metadata={"doc_id": "1"})]
    retriever.vectorstore.add_documents(documents, ids=["1"])
    retriever.docstore.mset(list(zip(["1"], documents)))
    results = retriever.invoke("1")
    assert len(results) > 0
    assert results[0].page_content == "test document"


@pytest.mark.asyncio
async def test_multi_vector_retriever_initialization_async() -> None:
    vectorstore = InMemoryVectorstoreWithSearch()
    retriever = MultiVectorRetriever(  # type: ignore[call-arg]
        vectorstore=vectorstore, docstore=InMemoryStore(), doc_id="doc_id"
    )
    documents = [Document(page_content="test document", metadata={"doc_id": "1"})]
    await retriever.vectorstore.aadd_documents(documents, ids=["1"])
    await retriever.docstore.amset(list(zip(["1"], documents)))
    results = await retriever.ainvoke("1")
    assert len(results) > 0
    assert results[0].page_content == "test document"
