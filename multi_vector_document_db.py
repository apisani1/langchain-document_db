import os
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

from sqlalchemy import (
    URL,
    Engine,
)
from sqlalchemy.ext.asyncio import AsyncEngine

from cached_docstore import CachedDocStore
from document_db import DocumentDB
from langchain.docstore.document import Document
from langchain.llms.base import BaseLanguageModel
from langchain.retrievers.multi_vector import SearchType
from langchain.schema.vectorstore import VectorStore
from langchain.vectorstores.chroma import Chroma
from langchain_core.stores import BaseStore
from langchain_openai import (
    ChatOpenAI,
    OpenAIEmbeddings,
)
from multi_vectorstore import MultiVectorStore


class MultiVectorDocumentDB(DocumentDB):
    """
    This class lets you load and keep in sync documents from any source into a vector store using an index.
    Specifically, it helps:
    - Avoid writing duplicated content into the vector store
    - Avoid re-writing unchanged content
    - Avoid re-computing embeddings over unchanged content
    The index will work even with documents that have gone through several transformation steps
    (e.g., via text chunking) with respect to the original source documents.

    """

    def __init__(
        self,
        location: Union[Path, str],
        *,
        docstore: Optional[BaseStore[str, Document]] = None,
        vectorstore: Optional[VectorStore] = None,
        sql_engine: Optional[Union[Engine, AsyncEngine]] = None,
        db_url: Optional[Union[str, URL]] = None,
        engine_kwargs: Optional[Dict[str, Any]] = None,
        async_mode: bool = False,
        functor: Optional[
            Union[
                str,
                Callable,
                List[Union[str, Callable, Tuple[Union[str, Callable], Dict]]],
            ]
        ] = None,
        func_kwargs: Optional[dict] = None,
        llm: Optional[BaseLanguageModel] = None,
        max_retries: int = 0,
        search_kwargs: Optional[dict] = None,
        search_type: SearchType = SearchType.similarity,
        cached: bool = False,
        **kwargs: Any,
    ) -> None:

        docstore = docstore or CachedDocStore(location + "/parent_docs", cached=cached)

        if not vectorstore:
            embedding = OpenAIEmbeddings()

            vectorstore = Chroma(
                persist_directory=location + "/childs_docs",
                embedding_function=embedding,
            )

        llm = llm or ChatOpenAI()

        multi_vectorstore = MultiVectorStore(
            vectorstore=vectorstore,
            docstore=docstore,
            ids_db_path=location,
            id_key="id",
            functor=functor,
            func_kwargs=func_kwargs,
            llm=llm,
            max_retries=max_retries,
            search_kwargs=search_kwargs,
            search_type=search_type,
            **kwargs
        )
        super().__init__(
            location=location,
            vectorstore=multi_vectorstore,
            sql_engine=sql_engine,
            db_url=db_url,
            engine_kwargs=engine_kwargs,
            async_mode=async_mode,
        )

    @classmethod
    async def ainit(
        cls,
        *args: Any,
        **kwargs: Any,
    ) -> "MultiVectorDocumentDB":
        db = MultiVectorDocumentDB(
            *args,
            async_mode=True,
            **kwargs,
        )
        await db.record_manager.acreate_schema()
        return db
