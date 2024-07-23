import uuid
from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)

from langchain.chains.base import Chain
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.llms.base import BaseLanguageModel
from langchain.output_parsers.openai_functions import JsonKeyOutputFunctionsParser
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers.multi_vector import (
    MultiVectorRetriever,
    SearchType,
)
from langchain.schema import BaseRetriever
from langchain.schema.vectorstore import VectorStore
from langchain.storage import InMemoryStore
from langchain_core.embeddings import Embeddings
from langchain_core.runnables.config import run_in_executor
from langchain_core.stores import (
    BaseStore,
    ByteStore,
)

from load_document import chunk_docs
from ids_db_sql import IDsDB


def load_question_chain(llm: BaseLanguageModel) -> Chain:
    functions = [
        {
            "name": "hypothetical_questions",
            "description": "Generate hypothetical questions",
            "parameters": {
                "type": "object",
                "properties": {
                    "questions": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["questions"],
            },
        }
    ]
    prompt = ChatPromptTemplate.from_template(
        """
        Generate a list of exactly {q} hypothetical questions that the below document could be used to answer:

        {doc}
        """
    )
    chain: Chain = (
        {
            "doc": lambda x: x["doc"].page_content,
            "q": lambda x: x["q"],
        }
        | prompt
        | llm.bind(
            functions=functions,
            function_call={"name": "hypothetical_questions"},
        )
        | JsonKeyOutputFunctionsParser(key_name="questions")
    )
    return chain


def _chunk(doc: Document, metadata: dict, **kwargs: Any) -> Document:
    return chunk_docs([doc], metadata=metadata, **kwargs)


def _sumarize(doc: Document, metadata: dict, chain: Chain) -> Document:
    return [
        Document(page_content=chain.invoke([doc])["output_text"], metadata=metadata)
    ]


def _generate_questions(
    doc: Document, metadata: dict, chain: Chain, q: int = 5
) -> list[Document]:
    questions = chain.invoke({"doc": doc, "q": q})
    return [
        Document(page_content=question, metadata=metadata)
        for question in questions
        if question is not None
    ]


class MultiVectorStore(VectorStore):
    """
    Args:
        vectorstore (VectorStore): VectorStore to use to store generated child documents.
        byte_store (ByteStore, optional): ByteStore to store the parent documents. Defaults to None.
        docstore (BaseStore[str, Document], optional): Docstore to store the parent documents. Defaults to None.
            If both `byte_store` and `docstore` are provided, `byte_store` will be used.
            If neither `byte_store` nor `docstore` is provided, an `InMemoryStore` will be used.
        id_key (str, optional): Key to use to identify the parent documents. Defaults to "doc_id".
        child_id_key (str, optional): Key to use to identify the child document. Defaults to "child_ids".
        functor (str | Callable, optional): Function to transform the parent document into the child documents.
            Defaults to chunking the parent documents into smaller chunks.
        func_kwargs (dict, optional): Keyword arguments to pass to the transformation function.
            Defaults to {"chunk_size": 500, "chunk_overlap": 50}.
        llm (BaseLanguageModel, optional): Language model to use for the transformation function of the parent documents.
            Defaults to None. If there is no language model provided and the transformation function rquires a LLM, an
            exception will be raised.
        max_retries (int, optional): Maximum number of retries to use when failing to transform douments.
            Defaults to 0.
        add_originals (bool): Whether to also add the parent documents to the vectorstore.
            Defaults to False.
        search_kwargs (dict, optional): Keyword arguments to pass to the MultiVectorRetriever.
        search_type (SearchType): Type of search to perform when using the retriever. Defaults to similarity.
        kwargs: Additional kwargs to pass to the MultiVectorRetriever.
    """

    def __init__(
        self,
        vectorstore: VectorStore,
        byte_store: Optional[ByteStore] = None,
        docstore: Optional[BaseStore[str, Document]] = None,
        ids_db_path: str = "",
        id_key: str = "doc_id",
        child_id_key: str = "child_ids",
        functor: Union[str, Callable] = None,
        func_kwargs: Optional[dict] = None,
        llm: Optional[BaseLanguageModel] = None,
        max_retries: int = 0,
        search_kwargs: Optional[dict] = None,
        search_type: SearchType = SearchType.similarity,
        **kwargs: Any,
    ) -> None:
        self.vectorstore = vectorstore
        self.byte_store = byte_store
        if not byte_store:
            docstore = docstore or InMemoryStore()
        self.docstore = docstore
        self.ids_db = IDsDB(ids_db_path)
        self.id_key = id_key
        self.child_id_key = child_id_key
        if not functor:
            functor = "chunk"
            func_kwargs = func_kwargs or {"chunk_size": 500, "chunk_overlap": 50}
        self.set_func(functor, func_kwargs, llm, max_retries)
        self.search_kwargs = search_kwargs or {}
        self.search_type = search_type
        self.retriever = self.as_retriever(
            search_kwargs=self.search_kwargs, search_type=self.search_type, **kwargs
        )

    def set_func(
        self,
        functor: Union[str, Callable],
        func_kwargs: Optional[dict] = None,
        llm: Optional[BaseLanguageModel] = None,
        max_retries: Optional[int] = None,
    ) -> None:
        self.functor, self.func_kwargs = self._get_func(functor, func_kwargs, llm)
        if max_retries is not None:
            self.max_retries = max_retries

    @classmethod
    def _get_func(
        cls,
        functor: Union[str, Callable],
        func_kwargs: Optional[dict],
        llm: Optional[BaseLanguageModel] = None,
    ) -> tuple[Callable, dict]:
        func_kwargs = func_kwargs or {}
        if callable(functor):
            return functor, func_kwargs
        if isinstance(functor, str):
            match functor:
                case "chunk":
                    return _chunk, func_kwargs
                case "summary":
                    if not llm:
                        raise ValueError("llm must be provided for summary")
                    chain = load_summarize_chain(llm)
                    func_kwargs.update({"chain": chain})
                    return _sumarize, func_kwargs
                case "question":
                    if not llm:
                        raise ValueError("llm must be provided for question")
                    chain = load_question_chain(llm)
                    func_kwargs.update({"chain": chain})
                    return _generate_questions, func_kwargs
                case "none":
                    return None, None
                case _:
                    pass
        raise ValueError("Bad functor for MultiVectorStore")

    @property
    def embeddings(self) -> Embeddings:
        self.vectorstore.embeddings

    def as_retriever(
        self,
        search_kwargs: Optional[dict] = None,
        search_type: Optional[SearchType] = None,
        **kwargs: Any,
    ) -> BaseRetriever:
        """
        Return a MultiVectorRetriever initialized from this MultiVectorStore.

        Args:
            search_type (str, optional): Defines the type of search tha the Retriever should perform.
                Can be "similarity" (default), "mmr", or "similarity_score_threshold".
            search_kwargs (dict, optional): Keyword arguments to pass to the search function.
                Can include the following:
                    k: Amount of documents to return (defaults to 4)
                    score_threshold: Minimum relevance threshold for similarity_score_threshold
                    fetch_k: Amount of documents to pass to MMR algorithm (defaults to 20)
                    lambda_mult: Diversity of results returned by MMR; 1 for minimum diversity and 0 for maximum.
                        defaults to 0.5.
                    filter: Filter by document metadata

        Returns:
            A MultiVectorRetriever initialized from this MultiVectorStore.
        """
        search_kwargs = search_kwargs or self.search_kwargs
        search_type = search_type or self.search_type
        tags = kwargs.pop("tags", None) or []
        tags.extend(self.vectorstore._get_retriever_tags())
        return MultiVectorRetriever(
            vectorstore=self.vectorstore,
            byte_store=self.byte_store,
            docstore=self.docstore,
            id_key=self.id_key,
            search_type=search_type,
            search_kwargs=search_kwargs,
            tags=tags,
            **kwargs,
        )

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        *args: Any,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> "MultiVectorStore":
        """Return a MultiVectorStore initialized from texts."""
        store = cls(*args, **kwargs)
        store.add_texts(texts, metadatas=metadatas)
        return store

    @classmethod
    async def afrom_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> "MultiVectorStore":
        """Return a MultiVectorStore initialized from texts."""
        return await run_in_executor(
            None, cls.from_texts, texts, embedding, metadatas, **kwargs
        )

    @classmethod
    def from_documents(
        cls, docs: list[Document], *args: Any, **kwargs: Any
    ) -> "MultiVectorStore":
        """Return a MultiVectorStore initialized from documents."""
        store = cls(*args, **kwargs)
        store.add_documents(docs)
        return store

    @classmethod
    async def afrom_documents(
        cls,
        documents: List[Document],
        embedding: Embeddings,
        **kwargs: Any,
    ) -> "MultiVectorStore":
        """Return a MultiVectorStore initialized from documents."""
        return await run_in_executor(None, cls.from_documents, documents, **kwargs)

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore."""
        return self.vectorstore.add_texts(texts, metadatas=metadatas, **kwargs)

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore."""
        return await self.vectorstore.aadd_texts(texts, metadatas, **kwargs)

    def _get_ids(self, ids: List[str], docs: Iterable[Document]) -> List[str]:
        if not ids:
            if len(docs) and self.id_key in docs[0].metadata:
                ids = [doc.metadata[self.id_key] for doc in docs]
            else:
                ids = [str(uuid.uuid4()) for _ in docs]
        return ids

    def add_documents(
        self,
        documents: Iterable[Document],
        ids: Optional[list[str]] = None,
        functor: Union[str, Callable] = None,
        func_kwargs: Optional[dict] = None,
        llm: Optional[BaseLanguageModel] = None,
        max_retries: Optional[int] = None,
        first_time: bool = True,
        add_originals: bool = False,
        **kwargs: Any,
    ) -> list[str]:
        """
        Run more documents through the document transformation function and add the resulting
        child documentsto the vectorstore.

        Args:
            documents (Iterable[Document]: Parent documents to process and generate child documents to be
                added to the vectorstore.
            functor (str | Callable, optional): Function to transform a parent document into child documents.
                Defaults to the function selected at the initialization of the class instance.
            func_kwargs (dict, optional): Keyword arguments to pass to the transformation function.
                Defaults to the keyword arguments selected at the initialization of the class instance.
            llm (BaseLanguageModel, optional): Language model to use for the transformation function.
                Defaults to None. If there is no language model provided and the transformation function rquires a LLM,
                an exception will be raised.
            max_retries (int, optional): Maximum number of retries to use when failing to transfomation process.
                Defaults to the maximum number of retries selected at the initialization of the class instance.
            add_originals (bool): Whether to add the original documents to the vectorstore.
                Defaults to False.
            kwargs:
                Additional kwargs to pass to the vectorstore.

        Returns:
            List[str]: List of ids of the parent documents.
        """
        # configure processing function and arguments
        if functor:
            functor, func_kwargs = self._get_func(functor, func_kwargs, llm)
        else:
            functor = self.functor
            func_kwargs = self.func_kwargs
        max_retries = max_retries or self.max_retries

        # generate ids for the parent documents
        ids = self._get_ids(ids, documents)

        # generate child document using the processing function and
        # add cross reference ids between the parent documents and their childs
        # add the child documents to the vector store
        for i, doc in enumerate(documents):
            doc_id = ids[i]
            doc.metadata[self.id_key] = doc_id

            if functor:
            # try to generate child documents using the processing function
                retries = 0
                sub_docs = None
                while not sub_docs and retries <= self.max_retries:
                    sub_docs = functor(doc, metadata={self.id_key: doc_id}, **func_kwargs)
                    if sub_docs:
                        child_ids = self.vectorstore.add_documents(sub_docs, **kwargs)
                        self.ids_db.add_ids(doc_id, child_ids, "child_ids")
                    retries += 1

            if add_originals:
                alias = self.vectorstore.add_documents([doc], ids=[doc_id], **kwargs)
                self.ids_db.add_ids(doc_id, alias, "aliases")

        # add the original documents to the document store for index retrieval
        if first_time:
            self.docstore.mset(list(zip(ids, documents)))

        return ids

    async def aadd_documents(
        self, documents: List[Document], **kwargs: Any
    ) -> List[str]:
        """
        Run more documents through the document transformation function and add the resulting
        child documentsto the vectorstore.

        Args:
            documents (Iterable[Document]: Parent documents to process and generate child documents to be
                added to the vectorstore.
            functor (str | Callable, optional): Function to transform a parent document into child documents.
                Defaults to the function selected at the initialization of the class instance.
            func_kwargs (dict, optional): Keyword arguments to pass to the transformation function.
                Defaults to the keyword arguments selected at the initialization of the class instance.
            llm (BaseLanguageModel, optional): Language model to use for the transformation function.
                Defaults to None. If there is no language model provided and the transformation function rquires a LLM,
                an exception will be raised.
            max_retries (int, optional): Maximum number of retries to use when failing to transfomation process.
                Defaults to the maximum number of retries selected at the initialization of the class instance.
            add_originals (bool): Whether to add the original documents to the vectorstore.
                Defaults to False.
            kwargs:
                Additional kwargs to pass to the vectorstore.

        Returns:
            List[str]: List of ids of the parent documents.
        """
        return await run_in_executor(None, self.add_documents, documents, **kwargs)

    def _expand_func_list(
        self, func_list: list[Union[str, Callable, Tuple[Union[str, Callable], dict]]]
    ) -> list[Union[str, Callable]]:
        if not func_list:
            return []
        func = func_list[0]
        if isinstance(func, tuple):
            return [func] + self._expand_func_list(func_list[1:])
        return [(func, {})] + self._expand_func_list(func_list[1:])

    def add_documents_multiple(
        self,
        documents: Iterable[Document],
        func_list: List[Union[str, Callable, Tuple[Union[str, Callable], dict]]],
        ids: Optional[list[str]] = None,
        add_originals: bool = False,
        llm: Optional[BaseLanguageModel] = None,
        max_retries: Optional[int] = None,
        **kwargs: Any,
    ) -> list[str]:
        """
        Run documents through the document transformation functions and add the resulting child documents to
        the vectorstore.

        Args:
            documents (Iterable[Document]: Paremt documents to process and generate the child documents to be
                added to the vectorstore.
            func_list (list[str | Callable | tuple[str | Callable, dict]]): List of functions and kwargs to transform a
                parent document into child documents.
            llm (Optional[BaseLanguageModel]): Language model to use for the transformation function.
                Defaults to None. If there is no language model provided and a transformation function rquires a LLM,
                an exception will be raised.
            max_retries (Optional[int]): Maximum number of retries to use when failing to transfomation process.
                Defaults to the maximum number of retries selected at the initialization of the class instance.
            add_originals (bool): Whether to add the original documents to the vectorstore.
                Defaults to False.
            kwargs:
                Additional kwargs to pass to the vectorstore.

        Returns:
            List[str]: List of ids of the parent documents.
        """
        ids = self._get_ids(ids, documents)
        func_list = self._expand_func_list(func_list)
        self.add_documents(
            documents,
            ids=ids,
            functor=func_list[0][0],
            func_kwargs=func_list[0][1],
            first_time=True,
            add_originals=add_originals,
            llm=llm,
            max_retries=max_retries,
            **kwargs,
        )
        for f, fk in func_list[1:]:
            self.add_documents(
                documents,
                ids=ids,
                functor=f,
                func_kwargs=fk,
                first_time=False,
                add_originals=False,
                llm=llm,
                max_retries=max_retries,
                **kwargs,
            )
        return ids

    async def aadd_documents_multiple(
        self,
        documents: list[Document],
        **kwargs: Any,
    ) -> list[str]:
        """
        Run documents through the document transformation function and add the resulting child documents to
        the vectorstore.

        Args:
            documents (Iterable[Document]: Paremt documents to process and generate the child documents to be
                added to the vectorstore.
            func_list (list[str | Callable | tuple[str | Callable, dict]]): List of functions and kwargs to transform a
                parent document into child documents.
            llm (Optional[BaseLanguageModel]): Language model to use for the transformation function.
                Defaults to None. If there is no language model provided and a transformation function rquires a LLM,
                an exception will be raised.
            max_retries (Optional[int]): Maximum number of retries to use when failing to transfomation process.
                Defaults to the maximum number of retries selected at the initialization of the class instance.
            add_originals (bool): Whether to add the original documents to the vectorstore.
                Defaults to False.
            kwargs:
                Additional kwargs to pass to the vectorstore.

        Returns:
            List[str]: List of ids of the parent documents.
        """
        return await run_in_executor(
            None, self.add_documents_multiple, documents, **kwargs
        )

    def delete(self, ids: Optional[list[str]] = None, **kwargs: Any) -> bool:
        """
        Delete by vector id or other criteria.

        Args:
            ids: List of ids of documents to delete.
            kwargs: Other keyword arguments that the vectorstore might use to delete documents.

        Returns:
            bool: True if deletion is successful. False otherwise.
        """
        try:
            documents = self.docstore.mget(ids)
            for doc in documents:
                if doc:
                    doc_id = doc.metadata[self.id_key]
                    child_ids = self.ids_db.get_ids(doc_id, "child_ids")
                    self.vectorstore.delete(child_ids, **kwargs)
                    self.ids_db.delete_ids(doc_id, "child_ids")
                    doc_aliases = self.ids_db.get_ids(doc_id, "aliases")
                    if doc_aliases:
                        self.vectorstore.delete(doc_aliases, **kwargs)
                        self.ids_db.delete_ids(doc_id, "aliases")
            self.docstore.mdelete(ids)
            return True
        except Exception as e:
            return False

    async def adelete(
        self, ids: Optional[List[str]] = None, **kwargs: Any
    ) -> Optional[bool]:
        """
        Delete by vector id or other criteria.

        Args:
            ids: List of ids to delete.
            kwargs: Other keyword arguments that the vectorstore might use.

        Returns:
            Optional[bool]: True if deletion is successful. False otherwise.
        """
        return await run_in_executor(None, self.delete, ids, **kwargs)

    def search(self, query: str, search_type: str, **kwargs: Any) -> List[Document]:
        """Return docs most similar to query using specified search type."""
        return self.vectorstore.search(query, search_type, **kwargs)

    async def asearch(
        self, query: str, search_type: str, **kwargs: Any
    ) -> List[Document]:
        """Return docs most similar to query using specified search type."""
        return await self.vectorstore.asearch(query, search_type, **kwargs)

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Return docs most similar to query."""
        return self.vectorstore.similarity_search(query, k, **kwargs)

    async def asimilarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Return docs most similar to query."""
        return await self.vectorstore.asimilarity_search(query, k, **kwargs)

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """
        The 'correct' relevance function
        may differ depending on a few things, including:
        - the distance / similarity metric used by the VectorStore
        - the scale of your embeddings (OpenAI's are unit normed. Many others are not!)
        - embedding dimensionality
        - etc.

        Vectorstores should define their own selection based method of relevance.
        """
        raise self.vectorstore._select_relevance_score_fn()

    def similarity_search_with_score(
        self, *args: Any, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """Run similarity search with distance."""
        return self.vectorstore.similarity_search_with_score(*args, **kwargs)

    async def asimilarity_search_with_score(
        self, *args: Any, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """Run similarity search with distance asynchronously."""
        return await self.vectorstore.asimilarity_search_with_score(*args, **kwargs)

    def similarity_search_by_vector(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query vector.
        """
        return self.vectorstore.similarity_search_by_vector(embedding, k, **kwargs)

    async def asimilarity_search_by_vector(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Return docs most similar to embedding vector."""
        return await self.vectorstore.asimilarity_search_by_vector(
            embedding, k, **kwargs
        )

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        return self.vectorstore.max_marginal_relevance_search(
            query, k, fetch_k, lambda_mult, **kwargs
        )

    async def amax_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        return await self.vectorstore.amax_marginal_relevance_search(
            query, k, fetch_k, lambda_mult, **kwargs
        )

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        return self.vectorstore.max_marginal_relevance_search_by_vector(
            embedding, k, fetch_k, lambda_mult, **kwargs
        )

    async def amax_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance."""
        return await self.vectorstore.amax_marginal_relevance_search_by_vector(
            embedding, k, fetch_k, lambda_mult, **kwargs
        )

    def get_by_ids(self, ids: List[str]) -> List[Document]:
        """Return documents with the given ids."""
        return self.docstore.mget(ids)

    def get_child_ids(self, id: str) -> List[str]:
        doc = self.docstore.mget([id])[0]
        if doc:
            doc_id = doc.metadata[self.id_key]
            assert doc_id == id
            return self.ids_db.get_ids(doc_id, "child_ids")
        return []

    def get_aliases(self, id: str) -> List[str]:
        doc = self.docstore.mget([id])[0]
        if doc:
            doc_id = doc.metadata[self.id_key]
            assert doc_id == id
            return self.ids_db.get_ids(doc_id, "aliases")
        return []
