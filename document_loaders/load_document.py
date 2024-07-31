import os
from pathlib import Path
from typing import (
    Any,
    Union,
)

from langchain.docstore.document import Document
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.text import TextLoader

from .text_splitter import chunkable


@chunkable
def load_document(
    file_path: Union[str, Path], *args: Any, **kwargs: Any
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
    loader: BaseLoader

    if "mode" in kwargs and kwargs["mode"] == "raw":
        kwargs.pop("mode")
        if "encoding" not in kwargs and "autodetect_encoding" not in kwargs:
            kwargs["autodetect_encoding"] = True

        loader = TextLoader(file_path, *args, **kwargs)
    else:
        _, extension = os.path.splitext(file_path)
        match extension:
            case ".txt":
                if "encoding" not in kwargs and "autodetect_encoding" not in kwargs:
                    kwargs["autodetect_encoding"] = True

                loader = TextLoader(file_path, *args, **kwargs)
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

                loader = UnstructuredFileLoader(file_path, *args, **kwargs)
            case ".docx" | ".doc":
                from langchain.document_loaders.word_document import Docx2txtLoader

                loader = Docx2txtLoader(file_path, *args, **kwargs)
            case ".csv":
                from langchain.document_loaders.csv_loader import CSVLoader

                loader = loader = CSVLoader(file_path, *args, **kwargs)
            case ".eml":
                from langchain.document_loaders.email import UnstructuredEmailLoader

                loader = UnstructuredEmailLoader(file_path, *args, **kwargs)
            case ".epub":
                from langchain.document_loaders.epub import UnstructuredEPubLoader

                loader = UnstructuredEPubLoader(file_path, *args, **kwargs)
            case ".xlsx" | ".xls":
                # data = load_document(file_path_name, mode='elements')
                from langchain.document_loaders.excel import UnstructuredExcelLoader

                loader = UnstructuredExcelLoader(file_path, *args, **kwargs)
            case ".pptx" | ".ppt":
                # data = load_document(file_path_name, mode='elements')
                from langchain.document_loaders.powerpoint import (
                    UnstructuredPowerPointLoader,
                )

                loader = UnstructuredPowerPointLoader(file_path, *args, **kwargs)
            case ".srt":
                from langchain.document_loaders.srt import SRTLoader

                loader = SRTLoader(file_path, *args, **kwargs)
            case ".html":
                from langchain.document_loaders.html import UnstructuredHTMLLoader

                if "mode" not in kwargs:
                    kwargs["mode"] = "single"
                if "strategy" not in kwargs:
                    kwargs["strategy"] = "fast"

                loader = UnstructuredHTMLLoader(file_path, *args, **kwargs)
            case ".json":
                from langchain.document_loaders.json_loader import JSONLoader

                if "jq_schema" not in kwargs:
                    kwargs["jq_schema"] = "."
                if "text_content" not in kwargs:
                    kwargs["text_content"] = False

                loader = JSONLoader(file_path, *args, **kwargs)
            case ".md":
                from langchain.document_loaders.markdown import (
                    UnstructuredMarkdownLoader,
                )

                loader = UnstructuredMarkdownLoader(file_path, *args, **kwargs)
            case ".ipynb":
                from langchain.document_loaders.notebook import NotebookLoader

                loader = NotebookLoader(file_path, *args, **kwargs)
            case _:
                from langchain.document_loaders.unstructured import (
                    UnstructuredFileLoader,
                )

                loader = UnstructuredFileLoader(file_path, *args, **kwargs)
    return loader.load()
