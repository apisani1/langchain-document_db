from functools import wraps
from typing import (
    Any,
    Callable,
    Optional,
    Union,
)

from langchain.docstore.document import Document
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from text_splitter import chunkable


def do_retry(
    wait: int = 30,
    multiplier: int = 1,
    attempts: int = 3,
    exception: Optional[Union[BaseException, list[BaseException]]] = None,
) -> Any:
    """
    Decorator to handle exceptions with retries.
    See: https://tenacity.readthedocs.io/en/latest/

    Args:
        wait (int, optional): Exponential backoff maximum wait time. Defaults to 30.
        multiplier (int, optional): Exponential backoff multiplier. Defaults to 1.
        attempts (int, optional): Number of attemps before giving up Defaults to 3.
        exception (Union[BaseException, list[BaseException]], optional): Exceptions to retry.
                                Defaults to None meaning all.
    """
    if exception:
        if isinstance(exception, list):
            exception_type = retry_if_exception_type(exception[0])  # type: ignore
            for e in exception[1:]:
                exception_type = exception_type | retry_if_exception_type(e)  # type: ignore
        else:
            exception_type = retry_if_exception_type(exception)  # type: ignore

    def retried(func: Callable) -> Any:
        if exception:

            @retry(
                wait=wait_random_exponential(multiplier=multiplier, max=wait),
                stop=stop_after_attempt(attempts),
                retry=exception_type,
            )
            @wraps(func)
            def wrapped(*args: Any, **kwargs: Any) -> Any:
                return func(*args, **kwargs)

        else:

            @retry(
                wait=wait_random_exponential(multiplier=multiplier, max=wait),
                stop=stop_after_attempt(attempts),
            )
            @wraps(func)
            def wrapped(*args: Any, **kwargs: Any) -> Any:
                return func(*args, **kwargs)

        return wrapped

    return retried


@do_retry()
@chunkable
def load_from_wikipedia(query: str, *args: Any, **kwargs: Any) -> list[Document]:
    """
    Load a Wikipedia page an convert it into a Langchain document.

    Each wiki page represents one Document. The hard limit on the number of downloaded Documents is 300 for now.

    Args:
    query (str): The query string to search on Wikipedia.
    lang (str, optional): The language code for the Wikipedia language edition. Defaults to "en".
    load_max_docs (int, optional): The maximum number of documents to load. Defaults to 100.
    load_all_available_meta (bool, optional): Indicates whether to load all available metadata for each document.
                                              Defaults to False.
    doc_content_chars_max (int, optional): The maximum number of characters for the document content. Defaults to 4000.

    Returns:
     List of Langchain documents or None if an error was encountered.
    """
    from langchain.document_loaders import WikipediaLoader

    loader = WikipediaLoader(query=query, *args, **kwargs)
    return loader.load()


@do_retry()
@chunkable
def load_from_web(web_path: str, *args: Any, **kwargs: Any) -> list[Document]:
    """
    Load web pages, parse them with BeautifulSoup and convert them into Langchain documents.

    Args:
    web_path (str or seq[str]):  Web paths to load from.
    requests_per_second (int = 2): Max number of concurrent requests to make.
    default_parser (str = "html.parser"): Default parser to use for BeautifulSoup.
    requests_kwargs (dict, optional): kwargs for requests
    raise_for_status (bool = False): Raise an exception if http status code denotes an error.
    bs_get_text_kwargs (dict, optional): kwargs for beatifulsoup4 get_text
    bs_kwargs (dict, optional): kwargs for beatifulsoup4 web page parsing

    Returns:
     List of Langchain documents or None if an error was encountered.
    """
    from langchain.document_loaders import WebBaseLoader

    loader = WebBaseLoader(web_path, *args, **kwargs)
    return loader.load()


@do_retry()
@chunkable
def load_with_chromium(
    web_paths: Union[str, list[str]], transform: bool = True, **kwargs: Any
) -> list[Document]:
    """
    Loads a list of web pages using chromium browser suppoting javascript.

    Args:
        web_paths (list[str]): the list of urls to load.
        transform (bool, optional): Whether to use Beautiful Soup to transform the html code. Defaults to True.

    Returns:
        List of Langchain documents or None if an error was encountered.
    """
    import nest_asyncio
    from langchain.document_loaders import AsyncChromiumLoader
    from langchain.document_transformers import BeautifulSoupTransformer

    nest_asyncio.apply()
    if not isinstance(web_paths, list):
        web_paths = [web_paths]
    loader = AsyncChromiumLoader(web_paths)
    html = loader.load()
    if transform:
        transformer = BeautifulSoupTransformer()
        return transformer.transform_documents(html, **kwargs)
    else:
        return html


@do_retry()
@chunkable
def crawl_with_apify(web_path: str, **kwargs: Any) -> list[Document]:
    """
    Recursively loads a web page using Apify website content crawl actor.

    Args:
        web_path (str): the url of the web page to load.

    Returns:
        List of Langchain documents or None if an error was encountered.
    """
    from langchain.utilities.apify import ApifyWrapper

    apify = ApifyWrapper()  # type: ignore
    loader = apify.call_actor(
        actor_id="apify/website-content-crawler",
        run_input={"startUrls": [{"url": web_path}]},
        dataset_mapping_function=lambda item: Document(
            page_content=item["text"] or "", metadata={"source": item["url"]}
        ),
        **kwargs,
    )
    return loader.load()
