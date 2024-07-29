from functools import wraps
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Union,
)

from langchain.docstore.document import Document
from langchain.text_splitter import (
    CharacterTextSplitter,
    HTMLHeaderTextSplitter,
    MarkdownHeaderTextSplitter,
    Language,
    RecursiveCharacterTextSplitter,
    RecursiveJsonSplitter,
    TextSplitter,
)
from langchain_experimental.text_splitter import SemanticChunker

DocumentSplitterType = Union[
    CharacterTextSplitter,
    HTMLHeaderTextSplitter,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
    RecursiveJsonSplitter,
    TextSplitter,
    SemanticChunker
]

DocumentSplitter = (
    CharacterTextSplitter,
    HTMLHeaderTextSplitter,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
    RecursiveJsonSplitter,
    SemanticChunker
)

def _get_splitter(text_splitter: Union[str, DocumentSplitterType], **kwargs) -> DocumentSplitterType:
    if isinstance(text_splitter, DocumentSplitter):
        return text_splitter
    if isinstance(text_splitter, str):
        match text_splitter:
            case "character":
                return CharacterTextSplitter(**kwargs)
            case "html":
                return HTMLHeaderTextSplitter(**kwargs)
            case "markdown":
                return MarkdownHeaderTextSplitter(**kwargs)
            case "recursive":
                return RecursiveCharacterTextSplitter(**kwargs)
            case "json":
                return RecursiveJsonSplitter(**kwargs)
            case "semantic":
                return SemanticChunker(**kwargs)
            case _:
                pass
    raise ValueError("Bad text_splitter for chunk_docs")

def chunk_docs(
    documents: Iterable[Document],
    text_splitter: Union[str, DocumentSplitterType] = "recursive",
    metadata: Optional[Dict] = None,
    **kwargs: Any,
) -> List[Document]:
    """
    Split a list of Langchain documents into chunks.

    Recursively tries to split by different characters to find one that works.

    Args:
        documents (Iterable[Document]): List of Langchain documents to chunk.
        text_splitter (Union[str, DocumentSplitterType], optional): Text splitter to use.
            Possible values are:
                "character" or CharacterTextSplitter
                "html" or HTMLHeaderTextSplitter
                "markdown" or MarkdownHeaderTextSplitter
                "recursive" or RecursiveCharacterTextSplitter
                "json" or RecursiveJsonSplitter
                "semantic" or SemanticChunker
        metadata (Optional[Dict], optional): Metadata to add to chunks. Defaults to None.
        **kwargs: Additional keyword arguments to pass to the text splitter.

    Returns:
        List of Langchain documents.
    """
    text_splitter = _get_splitter(text_splitter, **kwargs)
    chunks = text_splitter.split_documents(documents)
    if metadata:
        for chunk in chunks:
            chunk.metadata.update(metadata)
    return chunks

def chunkable(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(
        *args: Any,
        text_splitter: Union[str, DocumentSplitterType] = None,
        metadata: Optional[dict] = None,
        splitter_kwargs: Optional[dict] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Decorator that adds chunking capability to a function that returns a list of lamngchain Documents.

        Args:
            text_splitter (Union[str, DocumentSplitterType], optional): Text splitter to use.
                Possible values are:
                    "character" or CharacterTextSplitter
                    "html" or HTMLHeaderTextSplitter
                    "markdown" or MarkdownHeaderTextSplitter
                    "recursive" or RecursiveCharacterTextSplitter
                    "json" or RecursiveJsonSplitter
                    "semantic" or SemanticChunker
            metadata (dict, optional): Metadata to add to each chunk. Defaults to None.
            splitter_kwargs (dict, optional): Keyword arguments to pass to the text splitter. Defaults to None.
        """
        result = func(*args, **kwargs)
        if result and text_splitter:
            return chunk_docs(
                result,
                text_splitter=text_splitter,
                metadata=metadata,
                **splitter_kwargs,
            )
        return result

    return wrapper
