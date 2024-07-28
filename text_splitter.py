from functools import wraps
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
)

from langchain.docstore.document import Document
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    TextSplitter,
)

def chunk_docs(
    documents: Iterable[Document],
    text_splitter: Optional[TextSplitter] = None,
    metadata: Optional[Dict] = None,
    **kwargs: Any,
) -> List[Document]:
    """
    Split a list of Langchain documents into chunks.

    Recursively tries to split by different characters to find one that works.

    Args:
        documents (Iterable[Document]): List of Langchain documents to chunk.
        metadata (Optional[dict], optional): Metadata to add to chunks. Defaults to None.
        text_splitter (TextSplitter, optional): Text splitter to use. Defaults to RecursiveCharacterTextSplitter.
        chunk_size (int, optional): Maximum size of chunks to return, Defaults to 4000.
        chunk_overlap (int, optional): Overlap in characters between chunks. Defaults to 200.
        separators (list, optional): List of strings with separators. Defaults to None. If None uses ["\n\n", "\n", " "]
        length_function (func, optional): Function that measures the length of given chunks. Defaults to len.
        keep_separator (bool, optional): Whether to keep the separator in the chunks. Defaults to True.
        is_separator_regex (bool, optional): Wheter the separator is a regular expression. Defaults to False.
        add_start_index (bool, optional): If True, includes chunk's start index in metadata. Defaults to False.

    Returns:
        List of Langchain documents or None if an error was encountered.
    """

    text_splitter = text_splitter or RecursiveCharacterTextSplitter(**kwargs)
    chunks = text_splitter.split_documents(documents)
    if metadata:
        for chunk in chunks:
            chunk.metadata.update(metadata)
    return chunks


def chunkable(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(
        *args: Any,
        text_spliter: Optional[TextSplitter] = None,
        metadata: Optional[dict] = None,
        chunk_it: bool = False,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
        length_function: Callable = len,
        keep_separator: bool = True,
        is_separator_regex: bool = False,
        add_start_index: bool = False,
        **kwargs: Any,
    ) -> Any:
        """
        Decorator that adds chunking capability to a function that returns a list of lamngchain Documents.

        Args:
            chunk_it (bool, optional): Whether to chuck the documents. Defaults to False.
            chunk_size (int, optional): Maximum size of chunks to return, Defaults to 4000.
            chunk_overlap (int, optional): Overlap in characters between chunks. Defaults to 200.
            separators (list, optional): List of strings with separators. Defaults to None.
                                         If None uses ["\n\n", "\n", " "]
            length_function (func, optional): Function that measures the length of given chunks. Defaults to len.
            keep_separator (bool, optional): Whether to keep the separator in the chunks. Defaults to True.
            is_separator_regex (bool, optional): Wheter the separator is a regular expression. Defaults to False.
            add_start_index (bool, optional): If True, includes chunk's start index in metadata. Defaults to False.
            metadata (dict, optional): _description_. Defaults to None.
        """
        result = func(*args, **kwargs)
        if result and chunk_it:
            return chunk_docs(
                result,
                text_splitter=text_spliter,
                metadata=metadata,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=separators,
                length_function=length_function,
                keep_separator=keep_separator,
                is_separator_regex=is_separator_regex,
                add_start_index=add_start_index,
            )
        return result

    return wrapper
