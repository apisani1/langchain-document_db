import os
from pathlib import Path
from typing import (
    Any,
    Iterator,
    Optional,
    Union,
)

from langchain.docstore.document import Document
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.text import TextLoader

from .text_splitter import chunk_docs, DocumentSplitterType


def _get_document_loader(file_path: Union[str, Path], **kwargs: Any) -> Any:
    if "mode" in kwargs and kwargs["mode"] == "raw":
        kwargs.pop("mode")
        if "encoding" not in kwargs and "autodetect_encoding" not in kwargs:
            kwargs["autodetect_encoding"] = True

        return TextLoader(file_path, **kwargs)

    _, extension = os.path.splitext(file_path)
    match extension:
        case ".txt":
            if "encoding" not in kwargs and "autodetect_encoding" not in kwargs:
                kwargs["autodetect_encoding"] = True

            return TextLoader(file_path, **kwargs)
        case ".pdf":
            from langchain.document_loaders.unstructured import (
                UnstructuredFileLoader,
            )

            if "mode" not in kwargs:
                kwargs["mode"] = "single"
            if "strategy" not in kwargs:
                kwargs["strategy"] = "fast"
            if "load_tables" not in kwargs:
                kwargs["load_tables"] = False
            elif kwargs["load_tables"]:
                kwargs["skip_infer_table_types"] = []
                kwargs["pdf_infer_table_structure"] = True

            return UnstructuredFileLoader(file_path, **kwargs)
        case ".docx" | ".doc":
            from langchain.document_loaders.word_document import (
                UnstructuredWordDocumentLoader,
            )

            return UnstructuredWordDocumentLoader(file_path, **kwargs)
        case ".csv":
            from langchain.document_loaders.csv_loader import CSVLoader

            return CSVLoader(file_path, **kwargs)
        case ".eml":
            from langchain.document_loaders.email import UnstructuredEmailLoader

            return UnstructuredEmailLoader(file_path, **kwargs)
        case ".epub":
            from langchain.document_loaders.epub import UnstructuredEPubLoader

            return UnstructuredEPubLoader(file_path, **kwargs)
        case ".xlsx" | ".xls":
            # data = load_document(file_path_name, mode='elements')
            from langchain.document_loaders.excel import UnstructuredExcelLoader

            return UnstructuredExcelLoader(file_path, **kwargs)
        case ".pptx" | ".ppt":
            # data = load_document(file_path_name, mode='elements')
            from langchain.document_loaders.powerpoint import (
                UnstructuredPowerPointLoader,
            )

            return UnstructuredPowerPointLoader(file_path, **kwargs)
        case ".srt":
            from langchain.document_loaders.srt import SRTLoader

            return SRTLoader(file_path, **kwargs)
        case ".html":
            from langchain.document_loaders.html import UnstructuredHTMLLoader

            if "mode" not in kwargs:
                kwargs["mode"] = "single"
            if "strategy" not in kwargs:
                kwargs["strategy"] = "fast"

            return UnstructuredHTMLLoader(file_path, **kwargs)
        case ".json" | ".jsonl":
            from langchain.document_loaders.json_loader import JSONLoader

            if "jq_schema" not in kwargs:
                kwargs["jq_schema"] = "."
            if "text_content" not in kwargs:
                kwargs["text_content"] = False

            return JSONLoader(file_path, **kwargs)
        case ".md":
            from langchain.document_loaders.markdown import (
                UnstructuredMarkdownLoader,
            )

            return UnstructuredMarkdownLoader(file_path, **kwargs)
        case ".ipynb":
            from langchain.document_loaders.notebook import NotebookLoader

            return NotebookLoader(file_path, **kwargs)
        case _:
            from langchain.document_loaders.unstructured import (
                UnstructuredFileLoader,
            )

            return UnstructuredFileLoader(file_path, **kwargs)


def load_document_lazy(
    file_path: Union[str, Path],
    *,
    text_splitter: Union[str, DocumentSplitterType] = None,
    metadata: Optional[dict] = None,
    splitter_kwargs: Optional[dict] = None,
    **kwargs: Any,
) -> Iterator[Document]:
    """
    Generatior that loads an individual file and covert it into a list of Lanchain documents.

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
    splitter_kwargs = splitter_kwargs or {}
    loader_method = loader.lazy_load if hasattr(loader, "lazy_load") else loader.load
    if text_splitter:
        for doc in loader_method():
            for sub_doc in chunk_docs(
                [doc],
                text_splitter=text_splitter,
                metadata=metadata,
                **splitter_kwargs,
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
    return list(
        load_document_lazy(
            file_path,
            text_splitter=text_splitter,
            metadata=metadata,
            splitter_kwargs=splitter_kwargs,
            **kwargs,
        )
    )


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
        self.loader = _get_document_loader(file_path, **kwargs)
        self.text_splitter = text_splitter
        self.metadata = metadata or {}
        self.splitter_kwargs = splitter_kwargs or {}

    def lazy_load(self) -> Iterator[Document]:
        for doc in load_document_lazy(
            self.loader,
            text_splitter=self.text_splitter,
            metadata=self.metadata,
            splitter_kwargs=self.splitter_kwargs,
        ):
            yield doc
