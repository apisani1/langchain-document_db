# pylint: disable=pedantic

import shutil
from pathlib import Path
from typing import (
    Any,
    Optional,
    Union,
)

from langchain.docstore.document import Document
from langchain.indexes import SQLRecordManager
from langchain.schema.embeddings import Embeddings
from langchain.schema.vectorstore import VectorStore

from ..exception import exception_handler
from ._api import index
from .chroma_store import ChromaStore
from .embedding_retriever import EmbeddingRetriever
from .embedding_store import EmbeddingStore


DB_DIRECTORY = Path("/Users/antonio/Desktop/DataScience/MyData")


class DocumentDB(EmbeddingRetriever):
    """
    This class lets you load and keep in sync documents from any source into a vector store using an index.
    Specifically, it helps:
    - Avoid writing duplicated content into the vector store
    - Avoid re-writing unchanged content
    - Avoid re-computing embeddings over unchanged content
    The index will work even with documents that have gone through several transformation steps
    (e.g., via text chunking) with respect to the original source documents.

    Args:
            db_name (str): the name of the sub-directory where the vectot store and index will be saved
            vector_store (_VectorStore, optional): The vectore store to store the embeddings.
                                                   Defaults to Chroma with OpenAI Embeddings.
    """

    namespace: str
    location: Path
    vector_store: EmbeddingStore
    record_manager: SQLRecordManager

    @classmethod
    def connect(
        cls, db_name: str, vector_store: Optional[EmbeddingStore] = None, **kwargs: Any
    ) -> "DocumentDB":
        namespace = db_name
        location = DB_DIRECTORY / db_name
        vector_store = vector_store or ChromaStore.connect(
            index_name=(location / "vector_store").as_posix(), **kwargs
        )
        location.mkdir(parents=True, exist_ok=True)
        record_manager = SQLRecordManager(
            namespace,
            db_url=f"sqlite:///{location.as_posix()}/record_manager_cache.sql",
        )
        record_manager.create_schema()
        return cls(store=vector_store.store, vector_store=vector_store, record_manager=record_manager, namespace=namespace, location=location)

    def __del__(self) -> None:
        self.vector_store.embedding.save()
        pass

    @exception_handler
    def get_langchain_store(self) -> VectorStore:
        return self.store

    @exception_handler
    def get_embedding_function(self) -> Embeddings:
        return self.vector_store.embedding.model

    @exception_handler
    def get_index(self, *args: Any, **kwargs: Any) -> Any:
        return self.vector_store.get_index(*args, **kwargs)

    @exception_handler
    def size(self) -> int:
        return self.vector_store.size()

    @exception_handler
    def delete_index(self, *args: Any, **kwargs: Any) -> bool:
        """
        Deletes the index, vector store and enclosing direectory.
        """
        self.vector_store.delete_index(*args, **kwargs)
        shutil.rmtree(f"{self.location.as_posix()}")
        return True

    @exception_handler
    def fetch_entries(self, ids: Union[str, list[str]], **kwargs: Any) -> dict:
        return self.vector_store.fetch_entries(ids, **kwargs)

    @exception_handler
    def clean(
        self,
        id_key: str = "id",
        return_ids: bool = False,
    ) -> dict:
        """
        Delete all the entries in the index and the associates entries in the vector store.

        Args:
            id_key (str, optional): The key of the metadata entry use to index the documents. Defaults to "id".
            return_ids (bool, optional): Whether to return the ids of the entries deleted. Defaults to False.

        Returns:
            dict: A dictionary with the ids of the entries deleted or the number of entries deleted.
        """
        return index(  # type: ignore
            [],
            self.record_manager,
            self.store,
            cleanup="full",
            source_id_key="source",
            id_key=id_key,
            return_ids=return_ids,
        )

    @exception_handler
    def upsert_documents(
        self,
        docs: Union[Document, list[Document]],
        cleanup: str = "incremental",
        source_id_key: str = "source",
        id_key: str = "my_id",
        return_ids: bool = True,
    ) -> dict:
        """
        Add or updated documents into the vector store and index

        Args:
            docs (Union[Document, list[Document]]): Documents to add or update.
            cleanup (str, optional): Cleanup method. Defaults to "incremental".
                                    See: https://python.langchain.com/docs/modules/data_connection/indexing
            source_id_key (str, optional): The key of the metadata used to indetify the original document to create
                                           the chunks. Defaults to "source".
            id_key (str, optional): The key of the metadata entry use to index the documents. Defaults to "id".
            return_ids (bool, optional): Whether to return the ids of the entries deleted. Defaults to True.

        Returns:
            dict: A dictionary with the ids of the entries added, deleted or skipped or the number of entries added,
                  deleted or skipped.
        """
        if isinstance(docs, Document):
            docs = [docs]
        return index(  # type: ignore
            docs,
            self.record_manager,
            self.store,
            cleanup=cleanup,  # type: ignore
            source_id_key=source_id_key,
            id_key=id_key,
            return_ids=return_ids,
        )
