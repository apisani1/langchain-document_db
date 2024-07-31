import json
import os
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
    Language,
    MarkdownTextSplitter,
    RecursiveCharacterTextSplitter,
    RecursiveJsonSplitter,
    TextSplitter,
)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters.html import HTMLSectionSplitter


class HTMLSplitter(HTMLSectionSplitter):
    def __init__(self, *args, **kwargs):
        if "headers_to_split_on" not in kwargs:
            kwargs["headers_to_split_on"] = [
                ("h1", "Header 1"),
                ("h2", "Header 2"),
                ("h3", "Header 3"),
                ("h4", "Header 4"),
                ("h5", "Header 5"),
                ("h6", "Header 6"),
            ]
        # if "return_each_element" not in kwargs:
        #     kwargs["return_each_element"] = True
        super().__init__(*args, **kwargs)

    def split_documents(self, documents: Iterable[Document]) -> List[Document]:
        html_docs = []
        for doc in documents:
            try:
                sub_docs = self.split_text(doc.page_content)
                for sub_doc in sub_docs:
                    sub_doc.metadata.update(doc.metadata)
                html_docs.extend(sub_docs)
            except Exception:
                html_docs.append(doc)
        return html_docs


class JsonSplitter(RecursiveJsonSplitter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def split_documents(self, documents: Iterable[Document]) -> List[Document]:
        json_docs = []
        for doc in documents:
            try:
                json_data = json.loads(doc.page_content)
                sub_docs = [
                    Document(page_content=json.dumps(chunk), metadata=doc.metadata)
                    for chunk in self.split_json(json_data)
                ]
                json_docs.extend(sub_docs)
            except json.JSONDecodeError:
                json_docs.append(doc)
        return json_docs


DocumentSplitterType = Union[
    CharacterTextSplitter,
    HTMLSplitter,
    JsonSplitter,
    MarkdownTextSplitter,
    RecursiveCharacterTextSplitter,
    SemanticChunker,
]

DocumentSplitter = (
    CharacterTextSplitter,
    HTMLSplitter,
    JsonSplitter,
    MarkdownTextSplitter,
    RecursiveCharacterTextSplitter,
    SemanticChunker,
)

extension_to_type = {
    '.html': "html",
    '.json': "json",
    '.jsonl': "json",
    '.md': "markdown",
    '.go': "go",
    '.java': "java",
    '.kt': "kotlin",
    '.js': "js",
    '.ts': "ts",
    '.php': "php",
    '.proto': "proto",
    '.py': "python",
    '.rst': "rst",
    '.rb': "ruby",
    '.rs': "rust",
    '.scala': "scala",
    '.swift': "swift",\
    '.tex': "latex",
    '.sol': "sol",
    '.cs': "csharp",
    '.cob': "cobol",
    '.c': "c",
    '.lua': "lua",
    '.pl': "perl",
    '.hs': "haskell",
    '.ex': "elixir",
    '.exs': "elixir"
}


def _find_splitter_type(documents: Iterable[Document]) -> str:
    if len(documents) != 0 and "source" in documents[0].metadata:
        _, extension = os.path.splitext(documents[0].metadata["source"])
        if extension in extension_to_type:
            return extension_to_type[extension]
    return "recursive"


def _get_splitter(
    text_splitter: Union[str, DocumentSplitterType],
    documents: Iterable[Document],
    **kwargs,
) -> DocumentSplitterType:
    if isinstance(text_splitter, DocumentSplitter):
        return text_splitter
    if isinstance(text_splitter, str):
        if text_splitter == "auto":
            text_splitter = _find_splitter_type(documents)
        match text_splitter:
            case "character":
                return CharacterTextSplitter(**kwargs)
            case "html":
                return HTMLSplitter(**kwargs)
            case "json":
                return JsonSplitter(**kwargs)
            case "markdown":
                return MarkdownTextSplitter(**kwargs)
            case "recursive":
                return RecursiveCharacterTextSplitter(**kwargs)
            case "semantic":
                return SemanticChunker(**kwargs)
            case _:
                if text_splitter in Language:
                    return RecursiveCharacterTextSplitter.from_language(
                        language=Language(text_splitter), **kwargs
                    )
    raise ValueError(f"Bad text_splitter for chunk_docs: {text_splitter}")


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
        text_splitter (Union[str, DocumentSplitterType], optional): Text splitter to use to chunks the document into
            smaller documents. Possible values are:
                "auto": Automatically determine the splitter based on the file extension.
                "character" or CharacterTextSplitter,
                "markdown" or MarkdownHeaderTextSplitter,
                "recursive" or RecursiveCharacterTextSplitter,
                "semantic" or SemanticChunker,
                "c", "cpp", "csharp", "cobol",  "elixir", "go", "haskell", "html", "java", "js", "json", "kotlin",
                "latex", "lua", "php", "perl", "proto", "python", "rst", "ruby", "rust", "scala", "sol", "swift" "ts".
        **kwargs: Additional keyword arguments to pass to the text splitter.

    Returns:
        List of Langchain documents.
    """
    text_splitter = _get_splitter(text_splitter, documents, **kwargs)
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
        """
        splitter_kwargs = splitter_kwargs or {}
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
