import shutil
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Union,
)

from sqlalchemy import (
    URL,
    Engine,
)
from sqlalchemy.ext.asyncio import AsyncEngine

from langchain.docstore.document import Document
from langchain.indexes import (
    SQLRecordManager,
    index,
    aindex
)
from langchain.schema.vectorstore import VectorStore
from langchain_core.document_loaders.base import BaseLoader


class DocumentDB:
    """
    This class lets you load and keep in sync documents from any source into a vector store using an index.
    Specifically, it helps:
    - Avoid writing duplicated content into the vector store
    - Avoid re-writing unchanged content
    - Avoid re-computing embeddings over unchanged content
    The index will work even with documents that have gone through several transformation steps
    (e.g., via text chunking) with respect to the original source documents.

    Args:
            location (Path  str): the directory where the index will be saved
            vectorstore (VectorStore): The vectore store to store the embeddings.
            engine (Engine | AsyncEngine, optional): An already existing SQL Alchemy engine. Default is None.
            db_url (str | URL, optional): A database connection string used to create an SQL Alchemy engine.
                                          Default is None.
            engine_kwargs (dic, optional): Additional keyword arguments to be passed when creating the engine.
                                           Default is an empty dictionary.
            async_mode (bool, optional): Whether to create an async engine. Driver should support async operations.
                                         It only applies if db_url is provided. Default is False.
    """

    def __init__(
        self,
        location: Union[Path, str],
        vectorstore: VectorStore,
        sql_engine: Optional[Union[Engine, AsyncEngine]] = None,
        db_url: Optional[Union[str, URL]] = None,
        engine_kwargs: Optional[Dict[str, Any]] = None,
        async_mode: bool = False,
        **kwargs,
    ) -> None:
        self.location = Path(location).resolve()
        self.namespace = self.location.name
        self.vectorstore = vectorstore
        self.location.mkdir(parents=True, exist_ok=True)
        if not sql_engine and not db_url:
            db_url = f"sqlite:///{self.location.as_posix()}/record_manager_cache.sql"
        engine_kwargs = engine_kwargs or {}
        self.record_manager = SQLRecordManager(
            self.namespace,
            engine=sql_engine,
            db_url=db_url,
            engine_kwargs=engine_kwargs,
            async_mode=async_mode,
        )
        self.record_manager.create_schema()

    def as_retriever(self, *args, **kwargs) -> VectorStore:
        return self.vectorstore.as_retriever(*args, **kwargs)

    def delete_index(self) -> None:
        """
        Deletes the index and enclosing directory.
        """
        shutil.rmtree(f"{self.location.as_posix()}")

    def clean(
        self,
        source_id_key: str = "souce",
    ) -> Dict:
        """
        Delete all the entries in the index and the associates entries in the vector store.

        Args:
            id_key (str, optional): The key of the metadata entry use to index the documents. Defaults to "source".

        Returns:
            dict: A dictionary with the number of entries deleted.
        """
        return index(  # type: ignore
            [],
            self.record_manager,
            self.vectorstore,
            cleanup="full",
            source_id_key=source_id_key,
        )

    def upsert_documents(
        self,
        docs: Union[BaseLoader, Iterable[Document]],
        cleanup: str = "incremental",
        source_id_key: str = "source",
    ) -> Dict:
        """
        Add or updated documents into the vector store and index

        Args:
            docs (BaseLoader | Iterable[Document]): Documents to add or update.
            cleanup (str, optional): Cleanup method. Defaults to "incremental".
                                    See: https://python.langchain.com/docs/modules/data_connection/indexing
            source_id_key (str, optional): The key of the metadata used to identify the original document to create
                                           the chunks. Defaults to "source".

        Returns:
            dict: A dictionary with the number of entries added, deleted or skipped.
        """
        return index(  # type: ignore
            docs,
            self.record_manager,
            self.vectorstore,
            cleanup=cleanup,  # type: ignore
            source_id_key=source_id_key,
        )

    def delete_documents(
        self,
        docs_ids: List[str],
        source_id_key: str = "source",
    ) -> Dict:
        """
        Delete documents from the vector store and index

        Args:
            docs_ids (List[str]): List of document ids to delete.
            source_id_key (str, optional): The key of the metadata used to identify the original document to create
                                           the chunks. Defaults to "source".
        """
        docs_to_delete = [
            Document(
                page_content="Deleted DO NOT USE", metadata={source_id_key: doc_id}
            )
            for doc_id in docs_ids
        ]
        return self.upsert_documents(docs_to_delete, source_id_key=source_id_key)

    async def aclean(
        self,
        source_id_key: str = "souce",
    ) -> Dict:
        """
        Delete all the entries in the index and the associates entries in the vector store.

        Args:
            id_key (str, optional): The key of the metadata entry use to index the documents. Defaults to "source".

        Returns:
            dict: A dictionary with the number of entries deleted.
        """
        return await aindex(  # type: ignore
            [],
            self.record_manager,
            self.vectorstore,
            cleanup="full",
            source_id_key=source_id_key,
        )

    async def aupsert_documents(
        self,
        docs: Union[BaseLoader, Iterable[Document]],
        cleanup: str = "incremental",
        source_id_key: str = "source",
    ) -> Dict:
        """
        Add or updated documents into the vector store and index

        Args:
            docs (BaseLoader | Iterable[Document]): Documents to add or update.
            cleanup (str, optional): Cleanup method. Defaults to "incremental".
                                    See: https://python.langchain.com/docs/modules/data_connection/indexing
            source_id_key (str, optional): The key of the metadata used to identify the original document to create
                                           the chunks. Defaults to "source".

        Returns:
            dict: A dictionary with the number of entries added, deleted or skipped.
        """
        return await aindex(  # type: ignore
            docs,
            self.record_manager,
            self.vectorstore,
            cleanup=cleanup,  # type: ignore
            source_id_key=source_id_key,
        )

    async def adelete_documents(
        self,
        docs_ids: List[str],
        source_id_key: str = "source",
    ) -> Dict:
        """
        Delete documents from the vector store and index

        Args:
            docs_ids (List[str]): List of document ids to delete.
            source_id_key (str, optional): The key of the metadata used to identify the original document to create
                                           the chunks. Defaults to "source".
        """
        docs_to_delete = [
            Document(
                page_content="Deleted DO NOT USE", metadata={source_id_key: doc_id}
            )
            for doc_id in docs_ids
        ]
        return await self.aupsert_documents(docs_to_delete, source_id_key=source_id_key)
