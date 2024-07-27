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
from langchain_community.vectorstores import Chroma
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

    Additionally, this class lets you create multiple vectors per document. There are multiple use cases where this is
    beneficial. LangChain has a base MultiVectorRetriever which makes querying this type of setup easy.
    The methods to create multiple vectors per document include:
        -Smaller chunks: split a document into smaller chunks, and embed those.
        -Summary: create a summary for each document, embed that along with (or instead of) the document.
        -Hypothetical questions: create hypothetical questions that each document would be appropriate to answer,
                                 embed those along with (or instead of) the document.
        -Custom: use a custom function to transform the document into multiple documents, embed those.

    Args:
        location (Path | str): the directory where the database index will be saved.
        engine (Engine | AsyncEngine, optional): An already existing SQL Alchemy engine. Default is None.
        db_url (str | URL, optional): A database connection string used to create an SQL Alchemy engine.
                                      Default is None.
        engine_kwargs (dic, optional): Additional keyword arguments to be passed when creating the engine.
                                       Default is an empty dictionary.

        vectorstore (VectorStore): VectorStore to use to store generated child documents and their embeddings.
        byte_store (ByteStore, optional): ByteStore to store the parent documents. Defaults to None.
        docstore (BaseStore[str, Document], optional): Docstore to store the parent documents. Defaults to None.
            If both `byte_store` and `docstore` are provided, `byte_store` will be used.
            If neither `byte_store` nor `docstore` is provided, an `InMemoryStore` will be used.
        id_key (str, optional): Key to use to identify the parent documents. Defaults to "id".
        child_id_key (str, optional): Key to use to identify the child document. Defaults to "child_ids".
        functor (str | Callable, optional): Function to transform the parent document into the child documents.
            Defaults to chunking the parent documents into smaller chunks.
        func_kwargs (dict, optional): Keyword arguments to pass to the transformation function.
            Defaults to {"chunk_size": 500, "chunk_overlap": 50}.
        llm (BaseLanguageModel, optional): Language model to use for the transformation function of the parent documents.
            Defaults to None. If no language model is provided and the transformation function rquires a LLM, an
            exception will be raised.
        max_retries (int, optional): Maximum number of retries to use when failing to transform douments.
            Defaults to 0.
        add_originals (bool): Whether to also add the parent documents to the vectorstore. Defaults to False.
        search_kwargs (dict, optional): Keyword arguments to pass to the MultiVectorRetriever.
        search_type (SearchType): Type of search to perform when using the retriever. Defaults to similarity.
        kwargs: Additional kwargs to pass to the MultiVectorRetriever.
    """
    def __init__(
        self,
        location: Union[Path, str],
        *,

        sql_engine: Optional[Union[Engine, AsyncEngine]] = None,
        db_url: Optional[Union[str, URL]] = None,
        engine_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:

        multi_vectorstore = self._prepare_multi_vector_store(
            location,
            **kwargs,
        )
        super().__init__(
            location=location,
            vectorstore=multi_vectorstore,
            sql_engine=sql_engine,
            db_url=db_url,
            engine_kwargs=engine_kwargs,
            async_mode=False,
        )

    @classmethod
    async def ainit(
        cls,
        location: Union[Path, str],
        *,
        db_url: Optional[Union[str, URL]] = None,
        engine_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> "MultiVectorDocumentDB":
        """
        This class lets you load and keep in sync documents from any source into a vector store using an index.
        Specifically, it helps:
        - Avoid writing duplicated content into the vector store
        - Avoid re-writing unchanged content
        - Avoid re-computing embeddings over unchanged content
        The index will work even with documents that have gone through several transformation steps
        (e.g., via text chunking) with respect to the original source documents.

        Additionally, this class lets you create multiple vectors per document. There are multiple use cases where this is
        beneficial. LangChain has a base MultiVectorRetriever which makes querying this type of setup easy.
        The methods to create multiple vectors per document include:
            -Smaller chunks: split a document into smaller chunks, and embed those.
            -Summary: create a summary for each document, embed that along with (or instead of) the document.
            -Hypothetical questions: create hypothetical questions that each document would be appropriate to answer,
                                    embed those along with (or instead of) the document.
            -Custom: use a custom function to transform the document into multiple documents, embed those.

        Args:
            location (Path | str): the directory where the database index will be saved.
            engine (Engine | AsyncEngine, optional): An already existing SQL Alchemy engine. Default is None.
            db_url (str | URL, optional): A database connection string used to create an SQL Alchemy engine.
                                        Default is None.
            engine_kwargs (dic, optional): Additional keyword arguments to be passed when creating the engine.
                                        Default is an empty dictionary.

            vectorstore (VectorStore): VectorStore to use to store generated child documents and their embeddings.
            byte_store (ByteStore, optional): ByteStore to store the parent documents. Defaults to None.
            docstore (BaseStore[str, Document], optional): Docstore to store the parent documents. Defaults to None.
                If both `byte_store` and `docstore` are provided, `byte_store` will be used.
                If neither `byte_store` nor `docstore` is provided, an `InMemoryStore` will be used.
            id_key (str, optional): Key to use to identify the parent documents. Defaults to "id".
            child_id_key (str, optional): Key to use to identify the child document. Defaults to "child_ids".
            functor (str | Callable, optional): Function to transform the parent document into the child documents.
                Defaults to chunking the parent documents into smaller chunks.
            func_kwargs (dict, optional): Keyword arguments to pass to the transformation function.
                Defaults to {"chunk_size": 500, "chunk_overlap": 50}.
            llm (BaseLanguageModel, optional): Language model to use for the transformation function of the parent documents.
                Defaults to None. If no language model is provided and the transformation function rquires a LLM, an
                exception will be raised.
            max_retries (int, optional): Maximum number of retries to use when failing to transform douments.
                Defaults to 0.
            add_originals (bool): Whether to also add the parent documents to the vectorstore. Defaults to False.
            search_kwargs (dict, optional): Keyword arguments to pass to the MultiVectorRetriever.
            search_type (SearchType): Type of search to perform when using the retriever. Defaults to similarity.
            kwargs: Additional kwargs to pass to the MultiVectorRetriever.
        """

        multi_vectorstore = cls._prepare_multi_vector_store(
            location,
            **kwargs,
        )
        return await super().ainit(
            location=location,
            vectorstore=multi_vectorstore,
            db_url=db_url,
            engine_kwargs=engine_kwargs,
        )

    @staticmethod
    def _prepare_multi_vector_store(
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
        **kwargs: Any,
    ) -> MultiVectorStore:
        """
        Prepare the multi vector store.
        """
        docstore = docstore or CachedDocStore(location + "/parent_docs", cached=cached)

        if not vectorstore:
            embedding = OpenAIEmbeddings()

            vectorstore = Chroma(
                persist_directory=location + "/childs_docs",
                embedding_function=embedding,
            )

        llm = llm or ChatOpenAI()

        return MultiVectorStore(
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
