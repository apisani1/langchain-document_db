import os
from typing import (
    Any,
    Optional,
)

from langchain.docstore.document import Document
from langchain_community.document_loaders.base import BaseLoader

from text_splitter import chunkable


@chunkable
def load_document(file_path: str, *args: Any, **kwargs: Any) -> list[Document]:
    """
    Load individual files an covert it into a list of Lanchain documents.

    The corresponding document loader is selected acording to the file extension. If a specific document loader
    is not available tries with the Unstructured file loader.

    Args:
        file_path (str): File path or url address of the document to load.
        metadata (Optional[dict], optional): Metadata to add to chunks. Defaults to None.
        chunk_it (bool, optional): Whether to chuck the documents. Defaults to False.
        text_splitter (TextSplitter, optional): Text splitter to use. Defaults to RecursiveCharacterTextSplitter.
        chunk_size (int, optional): Maximum size of chunks to return, Defaults to 4000.
        chunk_overlap (int, optional): Overlap in characters between chunks. Defaults to 200.
        separators (list, optional): List of strings with separators. Defaults to None.
                                        If None uses ["\n\n", "\n", " "]
        length_function (func, optional): Function that measures the length of given chunks. Defaults to len.
        keep_separator (bool, optional): Whether to keep the separator in the chunks. Defaults to True.
        is_separator_regex (bool, optional): Wheter the separator is a regular expression. Defaults to False.
        add_start_index (bool, optional): If True, includes chunk's start index in metadata. Defaults to False.

    Returns:
        List of Langchain documents.
    """
    loader: BaseLoader

    _, extension = os.path.splitext(file_path)
    match extension:
        case ".txt":
            from langchain_community.document_loaders.text import TextLoader

            loader = TextLoader(file_path, *args, **kwargs)
        case ".pdf":
            from langchain_community.document_loaders.pdf import PyPDFLoader

            loader = PyPDFLoader(file_path, *args, **kwargs)
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

            loader = UnstructuredHTMLLoader(file_path, *args, **kwargs)
        case ".json":
            # data = load_document(file_path_name, jq_schema='.', text_content=False)
            from langchain.document_loaders.json_loader import JSONLoader

            loader = JSONLoader(file_path, *args, **kwargs)
        case ".md":
            # data = load_document(file_path_name, mode='elements')
            from langchain.document_loaders.markdown import UnstructuredMarkdownLoader

            loader = UnstructuredMarkdownLoader(file_path, *args, **kwargs)
        case ".ipynb":
            from langchain.document_loaders.notebook import NotebookLoader

            loader = NotebookLoader(file_path, *args, **kwargs)
        case _:
            from langchain.document_loaders.unstructured import UnstructuredFileLoader

            loader = UnstructuredFileLoader(file_path, *args, **kwargs)
    return loader.load()


@chunkable
def load_unstructured_document(
    file_path: str,
    mode: str = "single",
    load_tables: bool = False,
    chunking_strategy: Optional[str] = None,
    **kwargs: Any,
) -> list[Document]:
    """
    The file loader uses the unstructured partition function and will automatically detect the file
    type.

    Args:
        mode: You can run the loader in one of three modes: "single", "paged", and "elements".
              If you use "single" mode, the document will be returned as a single langchain Document object.
              If you use "paged" mode, the document will be splitted by page.
              If you use "elements" mode, the unstructured library will split the document into elements
              such as Title and NarrativeText.
        load_tables: Set it to True for unstructured infering the structure of tables automatically.
        metadata (Optional[dict], optional): Metadata to add to chunks. Defaults to None.
        chunk_it (bool, optional): Whether to chuck the documents. Defaults to False.
        text_splitter (TextSplitter, optional): Text splitter to use. Defaults to RecursiveCharacterTextSplitter.
        chunk_size (int, optional): Maximum size of chunks to return, Defaults to 4000.
        chunk_overlap (int, optional): Overlap in characters between chunks. Defaults to 200.
        separators (list, optional): List of strings with separators. Defaults to None.
                                        If None uses ["\n\n", "\n", " "]
        length_function (func, optional): Function that measures the length of given chunks. Defaults to len.
        keep_separator (bool, optional): Whether to keep the separator in the chunks. Defaults to True.
        is_separator_regex (bool, optional): Wheter the separator is a regular expression. Defaults to False.
        add_start_index (bool, optional): If True, includes chunk's start index in metadata. Defaults to False.

        You can pass in additional unstructured kwargs after mode to apply different unstructured settings. The
        following arguments manage the chunking strategy of unstructed. It may be redundant to use both.

        chunking_strategy: Strategy used for chunking text into larger or smaller elements. Defaults to `None` with
                           optional arg of 'by_title'
        multipage_sections: If True, sections can span multiple pages. Defaults to True.
        combine_text_under_n_chars: Combines elements (for example a series of titles) until a section reaches
                                    a length of n characters.
        new_after_n_chars: Cuts off new sections once they reach a length of n characters, a soft max.
        max_characters: Chunks elements text and text_as_html (if present) into chunks of length n characters,
                        a hard max.
    Returns:
        List of Langchain documents.
    """
    from langchain.document_loaders.unstructured import UnstructuredFileLoader

    if load_tables:
        kwargs.update({"skip_infer_table_types": [], "pdf_infer_table_structure": True})
    loader = UnstructuredFileLoader(
        file_path, mode=mode, chunking_strategy=chunking_strategy, **kwargs
    )
    return loader.load()
