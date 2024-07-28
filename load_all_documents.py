from typing import (
    Any,
    Iterator,
    Literal,
    Union,
)

from langchain.docstore.document import Document
from langchain.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.base import (
    BaseBlobParser,
    BaseLoader,
)

from text_splitter import chunkable


def load_all_documents_lazy(
    dir_path: str, extension: str, *args: Any, recursive: bool = False, **kwargs: Any
) -> Iterator[Document]:
    """
    Generator that loads all the files in a given directory with a given extension and yields them
    as Lanchain documents.

    The corresponding document parser is selected acording to the given extension. If a specific document
    parser is not available tries with the default parser.

    Args:
    dir_path (str): Directory to load files from.
    extension (str): Extension o files to load. Currently '.py', '.js', '.html', '.txt', '.pdf', '.doc', and 'docx'
                     have specific parsers implemented.
    recursie (bool, optional): if True the subdirectories will be included too.

    Returns:
    A generator of Langchain documents.
    """

    parser: Union[BaseBlobParser, Literal["default"]]

    glob = "**/*" if recursive else "*"
    match extension:
        case ".py":
            from langchain.document_loaders.parsers import LanguageParser

            parser = LanguageParser(language="python")
        case ".js":
            from langchain.document_loaders.parsers import LanguageParser

            parser = LanguageParser(language="js")
        case ".html":
            from langchain.document_loaders.parsers import BS4HTMLParser

            parser = BS4HTMLParser()
        case ".txt":
            from langchain.document_loaders.parsers.txt import TextParser

            parser = TextParser()
        case ".pdf.":
            from langchain.document_loaders.parsers.pdf import PyPDFParser

            parser = PyPDFParser()
        case ".doc" | ".docx":
            from langchain.document_loaders.parsers.msword import MsWordParser

            parser = MsWordParser()
        case _:
            parser = "default"
    loader = GenericLoader.from_filesystem(
        dir_path, *args, glob=glob, suffixes=[extension], parser=parser, **kwargs
    )
    return loader.lazy_load()


@chunkable
def load_all_documents(
    dir_path: str, extension: str, *args: Any, recursive: bool = False, **kwargs: Any
) -> list[Document]:
    """
    Load all the files in a given directory with a given extension and covert them into a list of Lanchain documents.

    The corresponding document parser is selected acording to the given extension. If a specific document parser is
    not available tries with the default parser.

    Args:
    dir_path (str): Directory to load files from.
    extension (str): Extension o files to load. Currently '.py', '.js', '.html', '.txt', '.pdf', '.doc', and 'docx'
                     have specific parsers implemented.
    recursie (bool, optional): if True the subdirectories will be included too.

    Returns:
    List of Langchain documents or None if an error was encountered.
    """

    parser: Union[BaseBlobParser, Literal["default"]]

    glob = "**/*" if recursive else "*"
    match extension:
        case ".py":
            from langchain.document_loaders.parsers import LanguageParser

            parser = LanguageParser(language="python")
        case ".js":
            from langchain.document_loaders.parsers import LanguageParser

            parser = LanguageParser(language="js")
        case ".html":
            from langchain.document_loaders.parsers import BS4HTMLParser

            parser = BS4HTMLParser()
        case ".txt":
            from langchain.document_loaders.parsers.txt import TextParser

            parser = TextParser()
        case ".pdf.":
            from langchain.document_loaders.parsers.pdf import PyPDFParser

            parser = PyPDFParser()
        case ".doc" | ".docx":
            from langchain.document_loaders.parsers.msword import MsWordParser

            parser = MsWordParser()
        case _:
            parser = "default"
    loader = GenericLoader.from_filesystem(
        dir_path, *args, glob=glob, suffixes=[extension], parser=parser, **kwargs
    )
    return loader.load()


class LoadAllDocuments(BaseLoader):
    """
    Loader that uses the Generic Loader to load all the files in a given directory with a given extension.
    It selects the right parser using the given extension. If a specific document parser is not available tries
    with the default parser.

    Args:
    dir_path (str): Directory to load files from.
    extension (str): Extension o files to load. Currently '.py', '.js', '.html', '.txt', '.pdf', '.doc', and 'docx'
                     have specific parsers implemented.
    recursie (bool, optional): if True the subdirectories will be included too.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        self.args = args
        self.kwargs = kwargs

    def lazy_load(self) -> Iterator[Document]:
        for doc in load_all_documents_lazy(*self.args, **self.kwargs):
            yield doc

    def load(self) -> list[Document]:
        return load_all_documents(*self.args, **self.kwargs)
