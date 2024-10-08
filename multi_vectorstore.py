import uuid
from typing import (
    Any,
    Callable,
    Dict,
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

from document_loaders.text_splitter import chunk_docs
from ids_db_sql import IDsDB

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100


def load_question_chain(llm: BaseLanguageModel) -> Chain:
    """
    Load a chain for generating hypothetical questions from a document. The chain uses the variables `doc` and `q` to
    generate questions. The `q` variable is the number of questions to generate.

    Args:
        llm (BaseLanguageModel): Language model to use for generating questions.

    Returns:
        Chain: Chain for generating hypothetical questions.
    """
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


def _chunk(doc: Document, metadata: dict, **kwargs: Any) -> List[Document]:
    """
    Chunk a document into smaller documents.

    Args:
        doc (Document): Document to chunk.
        metadata (dict): Metadata to add to the chunks.
        **kwargs: Additional keyword arguments for the text splitter.

    Returns:
        list[Document]: Chunked document.
    """
    return chunk_docs([doc], metadata=metadata, **kwargs)


def _sumarize(doc: Document, metadata: dict, chain: Chain) -> List[Document]:
    """
    Summarize a document using a language model. The chain uses the variables `doc` to generate a summary.

    Args:
        doc (Document): Document to summarize.
        metadata (dict): Metadata to add to the summary.
        chain (Chain): Chain to use for summarization.

    Returns:
        list[Document]: Summarized document.
    """
    return [
        Document(page_content=chain.invoke([doc])["output_text"], metadata=metadata)
    ]


def _generate_questions(
    doc: Document, metadata: dict, chain: Chain, q: int = 5
) -> list[Document]:
    """
    Generate hypothetical questions for a document using a language model. The chain uses the variables `doc` and
    `q` to generate questions. The `q` variable is the number of questions to generate.

    Args:
        doc (Document): Document to generate questions for.
        metadata (dict): Metadata to add to the questions.
        chain (Chain): Chain to use for generating questions.
        q (int, optional): Number of questions to generate. Defaults to 5.

    Returns:
        list[Document]: Hypothetical questions.
    """
    questions = chain.invoke({"doc": doc, "q": q})
    return [
        Document(page_content=question, metadata=metadata)
        for question in questions
        if question is not None
    ]


FunctorType = Union[
    str,
    Callable,
    List[Union[str, Callable, Tuple[Union[str, Callable], Dict]]],
]


class MultiVectorStore(VectorStore):
    """
    This class lets you create multiple vectors per document. There are multiple use cases where this is beneficial.
    LangChain has a base MultiVectorRetriever which makes querying this type of setup easy.
    The methods to create multiple vectors per document include:
        -Smaller chunks: split a document into smaller chunks, and embed those.
        -Summary: create a summary for each document, embed that along with (or instead of) the document.
        -Hypothetical questions: create hypothetical questions that each document would be appropriate to answer,
                                 embed those along with (or instead of) the document.
        -Custom: use a custom function to transform the document into multiple documents, embed those.
    This allows the retriever to use the child document embeddings for the search for the best match, but then return
    the parent documents that have more content.

    Args:
        vectorstore (VectorStore): VectorStore to use to store generated child documents and their embeddings.
        byte_store (ByteStore, optional): ByteStore to store the parent documents. Defaults to None.
        docstore (BaseStore[str, Document], optional): Docstore to store the parent documents. Defaults to None.
            If both `byte_store` and `docstore` are provided, `byte_store` will be used.
            If neither `byte_store` nor `docstore` is provided, an `InMemoryStore` will be used.
        id_key (str, optional): Key to use to identify the parent documents. Defaults to "doc_id".
        child_id_key (str, optional): Key to use to identify the child document. Defaults to "child_ids".
        functor: Function to transform the parent document into the child documents.
            Possible values include:
                - "chunk": Split the parent document into smaller chunks.
                - "summary": Create a summary for the parent document.
                - "question": Create hypothetical questions for the parent document.
                - Callable: Custom function to transform the parent document into child documents. It must have the
                            signature `func(doc: Document, metadata: dict, **kwargs: Any) -> list[Document]`.
                - List[Tuple]: List of functors and their keyword arguments. If a functor does not require keyword
                               arguments, the functor can be passed instead of a tuple.
            Defaults to "chunk".
        func_kwargs (dict, optional): Keyword arguments to pass to the transformation function.
            Defaults to {"chunk_size": 500, "chunk_overlap": 50}.
        llm (BaseLanguageModel, optional): Language model to use for the transformation function of the parent documents.
            Defaults to None. If there is no language model provided and the transformation function rquires a LLM, an
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
        vectorstore: VectorStore,
        *,
        byte_store: Optional[ByteStore] = None,
        docstore: Optional[BaseStore[str, Document]] = None,
        ids_db_path: str = "",
        id_key: str = "doc_id",
        child_id_key: str = "child_ids",
        functor: Optional[FunctorType] = None,
        func_kwargs: Optional[dict] = None,
        llm: Optional[BaseLanguageModel] = None,
        max_retries: int = 0,
        search_kwargs: Optional[Dict] = None,
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
            func_kwargs = func_kwargs or {
                "text_splitter": "recursive",
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
            }
        self.set_func(functor, func_kwargs, llm, max_retries)
        self.search_kwargs = search_kwargs or {}
        self.search_type = search_type
        self.retriever = self.as_retriever(
            search_kwargs=self.search_kwargs, search_type=self.search_type, **kwargs
        )

    def set_func(
        self,
        functor: FunctorType,
        func_kwargs: Optional[Dict] = None,
        llm: Optional[BaseLanguageModel] = None,
        max_retries: Optional[int] = None,
    ) -> None:
        """
        Set the transformation function for the parent documents into child documents.

        Args:
            functor: Function to transform the parent document into child documents.
                Possible values include:
                    - "chunk": Split the parent document into smaller chunks.
                    - "summary": Create a summary for the parent document.
                    - "question": Create hypothetical questions for the parent document.
                    - Callable: Custom function to transform the parent document into child documents. It must have the
                                signature `func(doc: Document, metadata: dict, **kwargs: Any) -> list[Document]`.
                    - List[Tuple]: List of functors and their keyword arguments. If a functor does not require keyword
                                   arguments, the functor can be passed instead of a tuple.
                Defaults to "chunk".
            func_kwargs (dict, optional): Keyword arguments to pass to the transformation function. Defaults to {}.
            llm (BaseLanguageModel, optional): Language model to use for the transformation function of the parent
                documents. Defaults to None. If there is no language model provided and the transformation function
                rquires a LLM, an exception will be raised.
            max_retries (int, optional): Maximum number of retries to use when failing to transform douments.
                Defaults to 0.
        """
        self.func_list = self._get_func_list(functor, func_kwargs, llm)
        if max_retries is not None:
            self.max_retries = max_retries

    @classmethod
    def _get_func_list(
        cls,
        functor: FunctorType,
        func_kwargs: Optional[Dict] = None,
        llm: Optional[BaseLanguageModel] = None,
    ) -> List[Tuple[Callable, dict]]:
        if isinstance(functor, list):
            func_list = cls._expand_func_list(functor)
        else:
            func_list = [(functor, func_kwargs or {})]
        return [
            cls._get_func(functor, func_kwargs, llm)
            for functor, func_kwargs in func_list
        ]

    @classmethod
    def _expand_func_list(
        cls, func_list: List[Union[str, Callable, Tuple[Union[str, Callable], Dict]]]
    ) -> List[Tuple[Union[str, Callable], Dict]]:
        if not func_list:
            return []
        func = func_list[0]
        if isinstance(func, tuple):
            return [func] + cls._expand_func_list(func_list[1:])
        return [(func, {})] + cls._expand_func_list(func_list[1:])

    @classmethod
    def _get_func(
        cls,
        functor: Union[str, Callable],
        func_kwargs: Dict,
        llm: Optional[BaseLanguageModel] = None,
    ) -> Tuple[Callable, dict]:
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
                    return None, {}
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
                Can be "similarity" (default) or "mmr".
            search_kwargs (dict, optional): Keyword arguments to pass to the search function.
                Can include the following:
                    k: Amount of documents to return (defaults to 4)
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
        functor: Optional[FunctorType] = None,
        func_kwargs: Optional[dict] = None,
        llm: Optional[BaseLanguageModel] = None,
        max_retries: Optional[int] = None,
        add_originals: bool = False,
        **kwargs: Any,
    ) -> list[str]:
        """
        Run more documents through the document transformation function and add the resulting
        child documentsto the vectorstore.

        Args:
            documents (Iterable[Document]: Parent documents to process and generate child documents to be
                added to the vectorstore.
            functor: Function to transform the parent document into child documents.
                Possible values include:
                    - "chunk": Split the parent document into smaller chunks.
                    - "summary": Create a summary for the parent document.
                    - "question": Create hypothetical questions for the parent document.
                    - Callable: Custom function to transform the parent document into child documents. It must have the
                                signature `func(doc: Document, metadata: dict, **kwargs: Any) -> list[Document]`.
                    - List[Tuple]: List of functors and their keyword arguments. If a functor does not require keyword
                                   arguments, the functor can be passed instead of a tuple.
                Defaults to the function(s) selected at the initialization of the class instance.
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
        func_list = (
            self._get_func_list(functor, func_kwargs, llm)
            if functor
            else self.func_list
        )
        max_retries = max_retries or self.max_retries

        # generate ids for the parent documents
        ids = self._get_ids(ids, documents)

        # generate child document using the processing functions and
        # add cross reference ids between the parent documents and their childs
        # add the child documents to the vector store
        for i, doc in enumerate(documents):
            doc_id = ids[i]
            doc.metadata[self.id_key] = doc_id

            for functor, func_kwargs in func_list:
                if not functor:
                    continue
                # try to generate child documents using the processing function
                retries = 0
                sub_docs = None
                while not sub_docs and retries <= self.max_retries:
                    sub_docs = functor(
                        doc, metadata={self.id_key: doc_id}, **func_kwargs
                    )
                    if sub_docs:
                        child_ids = self.vectorstore.add_documents(sub_docs, **kwargs)
                        self.ids_db.add_ids(doc_id, child_ids, "child_ids")
                    retries += 1

            if add_originals:
                alias = self.vectorstore.add_documents([doc], ids=[doc_id], **kwargs)
                self.ids_db.add_ids(doc_id, alias, "aliases")

        # add the parent documents to the document store for index retrieval
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
            functor: Function to transform the parent document into child documents.
                Possible values include:
                    - "chunk": Split the parent document into smaller chunks.
                    - "summary": Create a summary for the parent document.
                    - "question": Create hypothetical questions for the parent document.
                    - Callable: Custom function to transform the parent document into child documents. It must have the
                                signature `func(doc: Document, metadata: dict, **kwargs: Any) -> list[Document]`.
                    - List[Tuple]: List of functors and their keyword arguments. If a functor does not require keyword
                                   arguments, the functor can be passed instead of a tuple.
                Defaults to the function(s) selected at the initialization of the class instance.
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
