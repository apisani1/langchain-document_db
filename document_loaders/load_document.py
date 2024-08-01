import os
import logging
from pathlib import Path
from typing import (
    Any,
    Iterator,
    Optional,
    Union,
)

from langchain.docstore.document import Document
from langchain_community.document_loaders.base import BaseLoader

from .text_splitter import chunk_docs, DocumentSplitterType
from document_loaders import loaders_config


def _get_document_loader(file_path: Union[str, Path], **kwargs: Any) -> Any:
    _, extension = os.path.splitext(file_path)
    extension = extension.lower().split(".")[-1]
    if extension in loaders_config.document_loaders:
        loader = loaders_config.document_loaders[extension]["class"]
        loader_kwargs = loaders_config.document_loaders[extension]["kwargs"]
    else:
        if "default" in loaders_config.document_loaders:
            loader = loaders_config.document_loaders["default"]["class"]
            loader_kwargs = loaders_config.document_loaders["default"]["kwargs"]
        else:
            logging.warning(
                f"No loader found for extension {extension} and no default loader configured, using UnstructuredFileLoader"
            )
            try:
                from langchain.document_loaders.unstructured import (
                    UnstructuredFileLoader,
                )

                loader = UnstructuredFileLoader
                loader_kwargs = {}
            except ImportError:
                raise ImportError(
                    "The unstructured package is not installed. Please install it with `pip install unstructured`"
                )

    loader_kwargs.update(kwargs) # may produce kwargs conflicts

    return loader(file_path, **loader_kwargs)


def load_document_lazy(
    file_path: Union[str, Path],
    *,
    text_splitter: Union[str, DocumentSplitterType] = None,
    metadata: Optional[dict] = None,
    splitter_kwargs: Optional[dict] = None,
    **kwargs: Any,
) -> Iterator[Document]:
    """
    Generator that loads an individual file and covert it into a list of Lanchain documents.

    The corresponding document loader is selected acording to the file extension. If a specific document loader
    is not available tries with the UnstructuredFileLoader.

    Args:
        file_path (str | Path): File path of the document to load.
        text_splitter (Union[str, DocumentSplitterType], optional): Text splitter to use to chunks the document into
            smaller documents. Possible values are:
                "auto": Automatically determine the splitter based on the file extension.
                "character" or CharacterTextSplitter,
                "markdown" or MarkdownHeaderTextSplitter,
                "recursive" or RecursiveCharacterTextSplitter,
                "semantic" or SemanticChunker,
                "c", "cpp", "csharp", "cobol",  "elixir", "go", "haskell", "html", "java", "js", "json", "kotlin",
                "latex", "lua", "php", "perl", "proto", "python", "rst", "ruby", "rust", "scala", "sol", "swift" "ts"
        splitter_kwargs (dict, optional): Keyword arguments to pass to the text splitter. Defaults to None.
    Yiels:
        Langchain documents generated from the file.
    """
    loader = _get_document_loader(file_path, **kwargs)
    loader_method = loader.lazy_load if hasattr(loader, "lazy_load") else loader.load
    if text_splitter:
        for doc in loader_method():
            for sub_doc in chunk_docs(
                [doc],
                text_splitter=text_splitter,
                metadata=metadata,
                **(splitter_kwargs or {}),
            ):
                yield sub_doc
    else:
        for doc in loader_method():
            yield doc


def load_document(
    file_path: Union[str, Path],
    *,
    text_splitter: Union[str, DocumentSplitterType] = None,
    metadata: Optional[dict] = None,
    splitter_kwargs: Optional[dict] = None,
    **kwargs: Any,
) -> list[Document]:
    """
    Load an individual file and covert it into a list of Lanchain documents.

    The corresponding document loader is selected acording to the file extension. If a specific document loader
    is not available tries with the UnstructuredFileLoader.

    Args:
        file_path (str | Path): File path of the document to load.
        text_splitter (Union[str, DocumentSplitterType], optional): Text splitter to use to chunks the document into
            smaller documents. Possible values are:
                "auto": Automatically determine the splitter based on the file extension.
                "character" or CharacterTextSplitter,
                "markdown" or MarkdownHeaderTextSplitter,
                "recursive" or RecursiveCharacterTextSplitter,
                "semantic" or SemanticChunker,
                "c", "cpp", "csharp", "cobol",  "elixir", "go", "haskell", "html", "java", "js", "json", "kotlin",
                "latex", "lua", "php", "perl", "proto", "python", "rst", "ruby", "rust", "scala", "sol", "swift" "ts"
        splitter_kwargs (dict, optional): Keyword arguments to pass to the text splitter. Defaults to None.
    Returns:
        List of Langchain documents.
    """
    docs = _get_document_loader(file_path, **kwargs).load()
    if text_splitter:
        return chunk_docs(
            docs,
            text_splitter=text_splitter,
            metadata=metadata,
            **(splitter_kwargs or {}),
        )
    return docs


class DocumentLoader(BaseLoader):
    def __init__(
        self,
        file_path: Union[str, Path],
        *,
        text_splitter: Union[str, DocumentSplitterType] = None,
        metadata: Optional[dict] = None,
        splitter_kwargs: Optional[dict] = None,
        **kwargs: Any,
    ) -> None:
        self.file_path = file_path
        self.text_splitter = text_splitter
        self.metadata = metadata or {}
        self.splitter_kwargs = splitter_kwargs or {}
        self.kwargs = kwargs

    def lazy_load(self) -> Iterator[Document]:
        for doc in load_document_lazy(
            self.file_path,
            text_splitter=self.text_splitter,
            metadata=self.metadata,
            splitter_kwargs=self.splitter_kwargs,
            **self.kwargs,
        ):
            yield doc

    def load(self) -> list[Document]:
        return load_document(
            self.file_path,
            text_splitter=self.text_splitter,
            metadata=self.metadata,
            splitter_kwargs=self.splitter_kwargs,
            **self.kwargs,
        )
