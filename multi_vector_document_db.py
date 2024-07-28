from __future__ import annotations

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
from langchain_chroma import Chroma
from langchain_core.stores import BaseStore
from langchain_openai import (
    ChatOpenAI,
    OpenAIEmbeddings,
)

from multi_vectorstore import MultiVectorStore


class MultiVectorDocumentDB(DocumentDB):
    """
    This class extends DocumentDB to support multiple vectors per document.

    It provides additional functionality for:
    - Creating multiple vectors per document (e.g., smaller chunks, summaries, hypothetical questions)
    - Using LangChain's MultiVectorRetriever for efficient querying

    Args:
        location (Path | str): The directory where the database index will be saved.
        sql_engine (Engine | AsyncEngine, optional): An already existing SQL Alchemy engine.
        db_url (str | URL, optional): A database connection string used to create an SQL Alchemy engine.
        engine_kwargs (dict, optional): Additional keyword arguments for creating the engine.
        vectorstore (VectorStore, optional): VectorStore to use for storing generated child documents and embeddings.
        docstore (BaseStore[str, Document], optional): Docstore to store the parent documents.
        id_key (str, optional): Key to identify the parent documents. Defaults to "id".
        child_id_key (str, optional): Key to identify the child documents. Defaults to "child_ids".
        functor (str | Callable | List, optional): Function(s) to transform parent documents into child documents.
        func_kwargs (dict, optional): Keyword arguments for the transformation function.
        llm (BaseLanguageModel, optional): Language model for document transformation.
        max_retries (int, optional): Maximum number of retries for document transformation. Defaults to 0.
        add_originals (bool): Whether to add parent documents to the vectorstore. Defaults to False.
        search_kwargs (dict, optional): Keyword arguments for the MultiVectorRetriever.
        search_type (SearchType): Type of search for the retriever. Defaults to similarity.
        **kwargs: Additional arguments for MultiVectorRetriever.
    """

    def __init__(
        self,
        location: Union[Path, str],
        *,
        docstore: Optional[BaseStore[str, Document]] = None,
        cached: bool = False,
        id_key: str = "id",
        vectorstore: Optional[VectorStore] = None,
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
        sql_engine: Optional[Union[Engine, AsyncEngine]] = None,
        db_url: Optional[Union[str, URL]] = None,
        engine_kwargs: Optional[Dict[str, Any]] = None,
        async_mode: bool = False,
        **kwargs: Any,
    ) -> None:
        # Prepare the multi vector store.
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
            id_key=id_key,
            functor=functor,
            func_kwargs=func_kwargs,
            llm=llm,
            max_retries=max_retries,
            search_kwargs=search_kwargs,
            search_type=search_type,
            **kwargs,
        )

        # Initialize the parent document db.
        super().__init__(
            location=location,
            vectorstore=multi_vectorstore,
            sql_engine=sql_engine,
            db_url=db_url,
            engine_kwargs=engine_kwargs,
            async_mode=async_mode,
        )

    @classmethod
    def create(
        cls,
        location: Union[Path, str],
        **kwargs: Any,
    ) -> MultiVectorDocumentDB:
        """
        Create a synchronous MultiVectorDocumentDB instance.

        Args:
            location (Path | str): The directory where the database index will be saved.
            sql_engine (Engine | AsyncEngine, optional): An already existing SQL Alchemy engine.
            db_url (str | URL, optional): A database connection string used to create an SQL Alchemy engine.
            engine_kwargs (dict, optional): Additional keyword arguments for creating the engine.
            vectorstore (VectorStore, optional): VectorStore to use for storing generated child documents and embeddings.
            docstore (BaseStore[str, Document], optional): Docstore to store the parent documents.
            id_key (str, optional): Key to identify the parent documents. Defaults to "id".
            child_id_key (str, optional): Key to identify the child documents. Defaults to "child_ids".
            functor (str | Callable | List, optional): Function(s) to transform parent documents into child documents.
            func_kwargs (dict, optional): Keyword arguments for the transformation function.
            llm (BaseLanguageModel, optional): Language model for document transformation.
            max_retries (int, optional): Maximum number of retries for document transformation. Defaults to 0.
            add_originals (bool): Whether to add parent documents to the vectorstore. Defaults to False.
            search_kwargs (dict, optional): Keyword arguments for the MultiVectorRetriever.
            search_type (SearchType): Type of search for the retriever. Defaults to similarity.
            **kwargs: Additional arguments for MultiVectorRetriever.
        Returns:
            MultiVectorDocumentDB: A new MultiVectorDocumentDB instance.
        """

        instance = cls(location, **kwargs)
        instance.record_manager.create_schema()
        return instance

    @classmethod
    async def acreate(
        cls,
        location: Union[Path, str],
        **kwargs: Any,
    ) -> MultiVectorDocumentDB:
        """
        Create an asynchronous MultiVectorDocumentDB instance.

        Args:
            location (Path | str): The directory where the database index will be saved.
            sql_engine (Engine | AsyncEngine, optional): An already existing SQL Alchemy engine.
            db_url (str | URL, optional): A database connection string used to create an SQL Alchemy engine.
            engine_kwargs (dict, optional): Additional keyword arguments for creating the engine.
            vectorstore (VectorStore, optional): VectorStore to use for storing generated child documents and embeddings.
            docstore (BaseStore[str, Document], optional): Docstore to store the parent documents.
            id_key (str, optional): Key to identify the parent documents. Defaults to "id".
            child_id_key (str, optional): Key to identify the child documents. Defaults to "child_ids".
            functor (str | Callable | List, optional): Function(s) to transform parent documents into child documents.
            func_kwargs (dict, optional): Keyword arguments for the transformation function.
            llm (BaseLanguageModel, optional): Language model for document transformation.
            max_retries (int, optional): Maximum number of retries for document transformation. Defaults to 0.
            add_originals (bool): Whether to add parent documents to the vectorstore. Defaults to False.
            search_kwargs (dict, optional): Keyword arguments for the MultiVectorRetriever.
            search_type (SearchType): Type of search for the retriever. Defaults to similarity.
            **kwargs: Additional arguments for MultiVectorRetriever.
        Returns:
            MultiVectorDocumentDB: A new asynchronous MultiVectorDocumentDB instance.
        """
        instance = cls(location, async_mode=True, **kwargs)
        await instance.record_manager.acreate_schema()
        return instance
