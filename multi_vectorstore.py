import uuid
from typing import (
    Any,
    Callable,
    Optional,
    Union,
)

from langchain.chains.base import Chain
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.llms.base import BaseLanguageModel
from langchain.output_parsers.openai_functions import JsonKeyOutputFunctionsParser
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers.multi_vector import MultiVectorRetriever, SearchType
from langchain.schema import BaseRetriever
from langchain.schema.vectorstore import VectorStore
from langchain.storage import InMemoryStore
from langchain_core.stores import (
    BaseStore,
    ByteStore,
)

from ..document.chunk import chunk_docs
from ..exception.exception import (
    do_retry,
    exception_handler,
)
from ..model import get_chat
from ..store.faiss_store import FaissStore


class MultiVectorStore:
    def __init__(
        self,
        vectorstore: Optional[VectorStore] = None,
        byte_store: Optional[ByteStore] = None,
        docstore: Optional[BaseStore[str, Document]] = None,
        llm: Optional[BaseLanguageModel] = None,
        func: Union[str, Callable] = "chunk",
        key: str = "doc_id",
        search_kwargs: Optional[dict] = None,
        search_type: SearchType = SearchType.similarity,
        max_retries: int = 0,
        **kwargs: Any,
    ) -> None:
        vectorstore = vectorstore or FaissStore.connect()
        if not byte_store:
            docstore = docstore or InMemoryStore()
        llm = llm or get_chat(temperature=0)
        search_kwargs = search_kwargs or {}
        self.func, self.kwargs = self._get_func(llm, func, kwargs)
        self.key = key
        self.max_retries = max_retries
        self.retriever = MultiVectorRetriever(
            vectorstore=vectorstore,
            byte_store=byte_store,
            docstore=docstore,  # type: ignore
            id_key=key,
            search_kwargs=search_kwargs,
            search_type=search_type,
        )

    @exception_handler
    def set_func(
        self,
        func: Union[str, Callable],
        llm: Optional[BaseLanguageModel] = None,
        max_retries: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self.func, self.kwargs = self._get_func(llm, func, kwargs)
        if max_retries:
            self.max_retries = max_retries

    @classmethod
    def _get_func(
        cls, llm: Optional[BaseLanguageModel], func: Union[str, Callable], kwargs: dict
    ) -> tuple[Callable, dict]:
        if callable(func):
            return func, kwargs
        if isinstance(func, str):
            match func:
                case "chunk":
                    return cls._chunk, kwargs
                case "summary":
                    llm = llm or get_chat(temperature=0)
                    chain = load_summarize_chain(llm)
                    kwargs.update({"chain": chain})
                    return cls._sumarize, kwargs
                case "question":
                    llm = llm or get_chat(max_retries=0)
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
                        "Generate a list of exactly {q} hypothetical questions that the below document could be used to answer:\n\n{doc}"
                    )
                    chain: RunnableSerializable = (  # type: ignore
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
                    kwargs.update({"chain": chain})
                    return cls._generate_questions, kwargs
                case _:
                    pass
        raise ValueError("Bad functor for MultiVectorStore")

    @exception_handler
    @staticmethod
    def _chunk(doc: Document, metadata: dict, **kwargs: Any) -> Document:
        return chunk_docs([doc], metadata=metadata, **kwargs)

    @exception_handler
    @staticmethod
    def _sumarize(doc: Document, metadata: dict, chain: Chain) -> Document:
        return Document(page_content=chain.run([doc]), metadata=metadata)

    @exception_handler
    @staticmethod
    def _generate_questions(
        doc: Document, metadata: dict, chain: Chain, q: int = 5
    ) -> list[Document]:
        questions = chain.invoke({"doc": doc, "q": q})
        return [
            Document(page_content=question, metadata=metadata)
            for question in questions
            if question is not None
        ]

    def as_retriever(self) -> BaseRetriever:
        return self.retriever

    @exception_handler
    @do_retry()
    def add_documents(
        self,
        docs: list[Document],
        docs_ids: Optional[list[str]] = None,
        func: Optional[Union[str, Callable]] = None,
        func_kwargs: Optional[dict] = None,
        first_time: bool = True,
        add_originals: bool = False,
        llm: Optional[BaseLanguageModel] = None,
        max_retries: Optional[int] = None,
        **kwargs: Any,
    ) -> int:
        func_kwargs = func_kwargs or {}
        max_retries = max_retries if max_retries else self.max_retries
        # configure processing function and arguments
        if func:
            func, func_kwargs = self._get_func(llm, func, func_kwargs)
        else:
            func = self.func
            func_kwargs = self.kwargs
        # generated ids for the original documents
        if not docs_ids:
            doc_ids = [str(uuid.uuid4()) for _ in docs]
        # generate sub document using the processing function and
        # add reference id for the original documents
        sub_docs = []
        for i, doc in enumerate(docs):
            _docs = None
            retries = 0
            while not _docs and retries <= max_retries:
                _docs = func(doc, metadata={self.key: doc_ids[i]}, **func_kwargs)
                if _docs:
                    if isinstance(_docs, list):
                        sub_docs.extend(_docs)
                    else:
                        sub_docs.append(_docs)
                retries += 1
        # add generated sub documents to the vector store
        self.retriever.vectorstore.add_documents(sub_docs, **kwargs)
        # add original douments to the vector store if required
        if add_originals:
            self.retriever.vectorstore.add_documents(
                docs, id_key=self.key, doc_ids=doc_ids, **kwargs
            )
        # add the original documents to the document store for index retrieval
        if first_time:
            self.retriever.docstore.mset(list(zip(doc_ids, docs)))

        return len(sub_docs)

    @exception_handler
    def add_documents_multiple(
        self,
        docs: list[Document],
        func: list[Union[str, Callable]],
        func_kwargs: list[dict],
        add_originals: bool = False,
        llm: Optional[BaseLanguageModel] = None,
        max_retries: Optional[int] = None,
        **kwargs: Any,
    ) -> int:
        doc_ids = [str(uuid.uuid4()) for _ in docs]
        count = self.add_documents(
            docs,
            doc_ids=doc_ids,
            func=func[0],
            func_kwargs=func_kwargs[0],
            first_time=True,
            add_originals=add_originals,
            llm=llm,
            max_retries=max_retries,
            **kwargs,
        )
        for f, fk in zip(func[1:], func_kwargs[1:]):
            count += self.add_documents(
                docs,
                doc_ids=doc_ids,
                func=f,
                func_kwargs=fk,
                first_time=False,
                add_originals=False,
                llm=llm,
                max_retries=max_retries,
                **kwargs,
            )
        return count

    @exception_handler
    def delete(self, ids: Optional[list[str]] = None, **kwargs: Any) -> None:
        self.retriever.vectorstore.delete(ids, **kwargs)
        self.retriever.docstore.mdelete(ids)

    @exception_handler
    def get_relevant_documents(self, query: str, **kwargs: Any) -> list[Document]:
        return self.retriever.get_relevant_documents(query, **kwargs)
