import fnmatch
import os
from typing import (
    Any,
    Callable,
    Iterator,
    Iterable,
    List,
    Optional,
)

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

from .load_document import load_document_lazy


def load_directory_lazy(
    dir_path: str,
    *,
    recursive: bool = True,
    dir_filter: Optional[str] = None,
    file_filter: Optional[str] = None,
    pre_process: Optional[Callable] = None,
    post_process: Optional[Callable] = None,
    on_file_error: Optional[Callable] = None,
    topdown: bool = True,
    followlinks: bool = False,
    on_os_error: Optional[Callable] = None,
    **kwargs: Any,
) -> Iterator[Document]:
    """
    Generator that scans files and subdirectories in the given directory path and calls the load_document function on
    each file that matches the file_filter.

    Args:
    dir_path (str): Path to the directory to be scanned.
    recursive (bool, optional): if True, the search will include subdirectories. If False, only the top directory is
                                scanned. Defaults to True.
    dir_filter (str, optional): fpattern that directories must match to be included. Default is None.
    file_filter (str, optional): fpattern that files must match to be included. Default is None.
    pre_process (callable, optional): A function to call on each file before calling load_document on them. Deaults to None.
    post_process (callable, optional): A function to call on each doc after calling load_document on them. Deaults to None.
    on_file_error (callable, optional): A function to be called if an error is raised while loading a file. It will be
                              called with two arguments, the file path and the exception. Defaults to None.
    top_down (bool, optional): Whether the scan is done top-down or bottoms-up. Defaults to True.
    follow_links (bool, optional): Whether to follow simbolic links. Defaults to False.
    on_os_error (callable, optional): A function to be called if an error is raised while scanning the directory. It will be
                              called with one argument, an OSError instance.
                              It can report the error to continue with the walk, or raise the exception to abort the
                              walk.  Note that the filename is available as the filename attribute of the exception
                              object.
    kwargs: Additional keyword arguments to pass to load_document.
    Yields:
    Lanchain Documents generated from it.
    """
    for root, _, files in os.walk(
        dir_path, topdown=topdown, followlinks=followlinks, onerror=on_os_error
    ):
        if (
            root != "./"
            and dir_filter is not None
            and not fnmatch.fnmatch(root, dir_filter)
        ):
            continue

        for file in files:
            file_path = os.path.join(root, file)
            print(f"File: {file_path}")
            if file_filter is None or fnmatch.fnmatch(file, file_filter):
                if pre_process:
                    pre_process(file_path)
                try:
                    for doc in load_document_lazy(file_path, **kwargs):
                        if post_process:
                            post_process(file_path, doc)
                        yield doc
                except Exception as e:
                    if on_file_error:
                        on_file_error(file_path, e)

        if not recursive:
            break


def load_directory(
    dir_path: str,
    *,
    recursive: bool = True,
    dir_filter: Optional[str] = None,
    file_filter: Optional[str] = None,
    pre_process: Optional[Callable] = None,
    post_process: Optional[Callable] = None,
    on_file_error: Optional[Callable] = None,
    topdown: bool = True,
    followlinks: bool = False,
    on_os_error: Optional[Callable] = None,
    **kwargs: Any,
) -> List[Document]:
    """
    Scan for files and subdirectories in the given directory path and calls the load_document function on
    each file that matches the file_filter.

    Args:
    dir_path (str): Path to the directory to be scanned.
    recursive (bool, optional): if True, the search will include subdirectories. If False, only the top directory is
                                scanned. Defaults to True.
    dir_filter (str, optional): fpattern that directories must match to be included. Default is None.
    file_filter (str, optional): fpattern that files must match to be included. Default is None.
    pre_process (callable, optional): A function to call on each file before calling load_document on them. Deaults to None.
    post_process (callable, optional): A function to call on each doc after calling load_document on them. Deaults to None.
    on_file_error (callable, optional): A function to be called if an error is raised while loading a file. It will be
                              called with two arguments, the file path and the exception. Defaults to None.
    top_down (bool, optional): Whether the scan is done top-down or bottoms-up. Defaults to True.
    follow_links (bool, optional): Whether to follow simbolic links. Defaults to False.
    on_os_error (callable, optional): A function to be called if an error is raised while scanning the directory. It will be
                              called with one argument, an OSError instance.
                              It can report the error to continue with the walk, or raise the exception to abort the
                              walk.  Note that the filename is available as the filename attribute of the exception
                              object.
    kwargs: Additional keyword arguments to pass to load_document.

    Returns:
    A list of Lanchain Documents generated from load_document.
    """
    return list(
        load_directory_lazy(
            dir_path,
            recursive=recursive,
            dir_filter=dir_filter,
            file_filter=file_filter,
            pre_process=pre_process,
            post_process=post_process,
            on_file_error=on_file_error,
            topdown=topdown,
            followlinks=followlinks,
            on_os_error=on_os_error,
            **kwargs,
        )
    )


class DirectoryLoader(BaseLoader):
    """
    Loader that scans files and subdirectories in the given directory path and calls the load_document function on
    each file that matches the file_filter.

    Args:
    dir_path (str): Path to the directory to be scanned.
    recursive (bool, optional): if True, the search will include subdirectories. If False, only the top directory is
                                scanned. Defaults to True.
    file_filter (str, optional): A glob-style pattern that files must match to be included. Default is None.
    pre_process (func, optional): A function to call on each file before calling load_document on them.
    post_process (func, optional): A function to call on each doc after calling load_document on them. Deaults to None
    top_down (bool, optional): Whether the scan is done top-down or bottoms-up. Defaults to True.
    follow_links (bool, optional): Whether to follow simbolic links. Defaults to False.
    onerror (bool, optional): A function to be called if an error is raised while scanning the directory. It will be
                              called with one argument, an OSError instance. It can report the error to continue with
                              the walk, or raise the exception to abort the walk.  Note that the filename is available
                              as the filename attribute of the exception object.
    kwargs: Additional keyword arguments to pass to load_document.
    """

    def __init__(self, dir_path: str, **kwargs: Any):
        self.dir_path = dir_path
        self.kwargs = kwargs

    def lazy_load(self) -> Iterator[Document]:
        for doc in load_directory_lazy(self.dir_path, **self.kwargs):
            yield doc


def load_files_lazy(
    files: Iterable[str],
    pre_process: Optional[Callable] = None,
    post_process: Optional[Callable] = None,
    on_file_error: Optional[Callable] = None,
    **kwargs: Any,
) -> Iterator[Document]:
    """
    Generator that scans files and subdirectories in the given directory path and calls the load_document function on
    each file that matches the file_filter.

    Args:
    file_path (str): Path to the file to be processed
    pre_process (callable, optional): A function to call on each file before calling load_document on them. Deaults to None.
    post_process (callable, optional): A function to call on each doc after calling load_document on them. Deaults to None.
    on_file_error (callable, optional): A function to be called if an error is raised while loading a file. It will be
                              called with two arguments, the file path and the exception. Defaults to None.
    kwargs: Additional keyword arguments to pass to load_document.
    Yields:
    Lanchain Documents generated from it.
    """

    for file in files:
        if pre_process:
            pre_process(file)
        try:
            for doc in load_document_lazy(file, **kwargs):
                if post_process:
                    post_process(file, doc)
                yield doc
        except Exception as e:
            if on_file_error:
                on_file_error(file, e)

def load_files(
    files: Iterable[str],
    pre_process: Optional[Callable] = None,
    post_process: Optional[Callable] = None,
    on_file_error: Optional[Callable] = None,
    **kwargs: Any,
) -> List[Document]:
    """
    Calls the load_document function on each file path provided.

    Args:
    file_path (str): Path to the file to be processed
    pre_process (callable, optional): A function to call on each file before calling load_document on them. Deaults to None.
    post_process (callable, optional): A function to call on each doc after calling load_document on them. Deaults to None.
    on_file_error (callable, optional): A function to be called if an error is raised while loading a file. It will be
                              called with two arguments, the file path and the exception. Defaults to None.
    kwargs: Additional keyword arguments to pass to load_document.

    Returns:
    Lanchain Documents generated from it.
    """

    return list(
        load_files_lazy(
            files=files,
            pre_process=pre_process,
            post_process=post_process,
            on_file_error=on_file_error,
            **kwargs,
        )
    )
