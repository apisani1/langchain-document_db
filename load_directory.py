import fnmatch
import os
from typing import (
    Any,
    Callable,
    Iterator,
    List,
    Optional,
)

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

from load_document import load_document
from text_splitter import chunkable


@chunkable
def load_directory(dir_path: str, *args: Any, **kwargs: Any) -> List[Document]:
    """
    Load all the documents in a directory.

    Args:
    dir_path (str): Path to the directory to be scanned.
    glob (str, optional): Filter to control wich files to load. eg: *.txt. Uses the glob library syntax.
                          Defaults to "**/[!.]*" (all files except hidden).
    show_progress (bool, optional): Show progress bar using tqdm.
    recursive (bool, optional): Whether to recursively search for files. Defaults to False.
    use_multithreading (bool, optional): Set to True to utilize several threads.
    silent_errors (bool, optional): Skip the files which could not be loaded and continue the load process.
    loader_cls (loader class, optional): Loader class to be used eg: TexLoader.
    By default this uses the UnstructuredLoader class.
    loader_kwargs (dict, optional): Arguments for the loader eg: {'autodetect_encoding': True}

    Returns:
     List of Langchain documents or None if an error was encountered.

    Examples:

    docs = load_directory(dir_path, glob="**/*.txt", loader_cls=TextLoader, silent_errors=True,
               loader_kwargs={'autodetect_encoding': True})

    docs = load_directory(dir_path, glob="**/*.py", loader_cls=PythonLoader, show_progress=True)
    """
    from langchain.document_loaders import DirectoryLoader

    loader = DirectoryLoader(dir_path, *args, **kwargs)
    return loader.load()


def scan_load_directory(
    dir_path: str,
    *args: Any,
    recursive: bool = True,
    file_filter: Optional[str] = None,
    pre_process: Optional[Callable] = None,
    topdown: bool = True,
    followlinks: bool = False,
    on_error: Optional[Callable] = None,
    **kwargs: Any,
) -> Iterator[tuple[str, Any]]:
    """
    Generator that scans files and subdirectories in the given directory path and calls the load_document function on
    each file that matches the file_filter.

    Args:
    dir_path (str): Path to the directory to be scanned.
    recursive (bool, optional): if True, the search will include subdirectories. If False, only the top directory is
                                scanned. Defaults to True.
    file_filter (str, optional): A glob-style pattern that files must match to be included. Default is None.
    pre_process (func, optional): A function to call on each file before calling load_document on them.
    top_down (bool, optional): Whether the scan is done top-down or bottoms-up. Defaults to True.
    follow_links (bool, optional): Whether to follow simbolic links. Defaults to False.
    onerror (bool, optional): A function to be called if an error is raised while scanning the directory. It will be
                              called with one argument, an OSError instance.
                              It can report the error to continue with the walk, or raise the exception to abort the
                              walk.  Note that the filename is available as the filename attribute of the exception
                              object.

    Yields:
    A tupple with the file path and the list of Lanchain Documents generated from it.

    Example usage: Only process  files ending with ".txt"
    for file, docs in scan_load_dir(directory_path, recursive=True, file_filter='*.txt'):
        for doc in docs:
            doc
    """
    for root, _, files in os.walk(
        dir_path, topdown=topdown, followlinks=followlinks, onerror=on_error
    ):
        for file in files:
            file_path = os.path.join(root, file)
            if file_filter is None or fnmatch.fnmatch(file, file_filter):
                if pre_process:
                    pre_process(file_path)
                yield file_path, (load_document(file_path, *args, **kwargs) or [])

        if not recursive:
            break  # Stop recursion if not required


class ScanLoadDirectory(BaseLoader):
    """
    Loader that scans files and subdirectories in the given directory path and calls the load_document function on
    each file that matches the file_filter.

    Args:
    dir_path (str): Path to the directory to be scanned.
    recursive (bool, optional): if True, the search will include subdirectories. If False, only the top directory is
                                scanned. Defaults to True.
    file_filter (str, optional): A glob-style pattern that files must match to be included. Default is None.
    pre_process (func, optional): A function to call on each file before calling load_document on them.
    top_down (bool, optional): Whether the scan is done top-down or bottoms-up. Defaults to True.
    follow_links (bool, optional): Whether to follow simbolic links. Defaults to False.
    onerror (bool, optional): A function to be called if an error is raised while scanning the directory. It will be
                              called with one argument, an OSError instance. It can report the error to continue with
                              the walk, or raise the exception to abort the walk.  Note that the filename is available
                              as the filename attribute of the exception object.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        self.args = args
        self.kwargs = kwargs

    def lazy_load(self) -> Iterator[Document]:
        for _, docs in scan_load_directory(*self.args, **self.kwargs):
            for doc in docs:
                yield doc

    def load(self) -> List[Document]:
        return List(self.lazy_load())
