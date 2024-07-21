import os
import pickle
from pathlib import Path
from typing import (
    Any,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from langchain.docstore.document import Document
from langchain.storage import LocalFileStore
from langchain_community.storage import RedisStore
from langchain_core.stores import BaseStore


class NillStore(BaseStore[str, Document]):
    def mget(self, keys: Sequence[str]) -> List[Optional[Document]]:
        return [None] * len(keys)

    def mset(self, key_doc_pairs: Sequence[Tuple[str, Document]]) -> None:
        return

    def mdelete(self, keys: Sequence[str]) -> None:
        return

    def yield_keys(self, *, prefix: str | None = None) -> Iterator[str]:
        return


class CachedDocStore(BaseStore[str, Document]):
    """
    A document store that saves documents locally in the file system and caches them in a Redis server.

    Args:
        root_path (str | Path]): The root path of the file store. All keys are interpreted as paths relative
            to this root.
        chmod_file: (int, optional): If specified, sets permissions for newly created files, overriding the
            current `umask` if needed. Defaults to `None`
        chmod_dir: (int, optional): If specified, sets permissions for newly created dirs, overriding the
            current `umask` if needed. Defaults to `None`
        update_atime: (bool, optional): If `True`, updates the filesystem access time (but not the modified time)
            when a file is read. This allows MRU/LRU cache policies to be implemented for filesystems where access
            time updates are disabled. Defaults to `False`
        redis_client (optional): A Redis connection instance. If not provided, a new connection will be created.
            Defaults to `None`
        redis_url (str, optional): redis url to create a new connection if no redis_client is provided. If not provided,
            it looks for the environmt variable REDIS_URI. if the environment variable is not found it tries to connect
            to a local using redis://localhost:6379. Defaults to `None`
        client_kwargs (dic, optional): Keyword arguments to pass to the Redis client
        ttl (int, optional): time to expire keys in seconds if provided, if None keys will never expire.
            Defaults to `None`
        prefix (str, optional): if provided, all keys will be prefixed with this prefix. Defaults to None.

    """
    def __init__(
        self,
        root_path: Optional[Union[str, Path]] = None,
        *,
        cached: bool = True,
        chmod_file: Optional[int] = None,
        chmod_dir: Optional[int] = None,
        update_atime: bool = False,
        redis_client: Any = None,
        redis_url: Optional[str] = None,
        client_kwargs: Optional[dict] = None,
        ttl: Optional[int] = None,
        prefix: Optional[str] = None,
    ):
        self.root_path = root_path if root_path else "./.docstore"
        self.update_atime = update_atime
        self.file_store = LocalFileStore(
            self.root_path,
            chmod_file=chmod_file,
            chmod_dir=chmod_dir,
            update_atime=update_atime,
        )
        if cached:
            if not redis_client and not redis_url:
                redis_url = os.getenv("REDIS_URI", None)
                if not redis_url:
                    redis_url = "redis://localhost:6379"
            self.redis_store = RedisStore(
                client=redis_client,
                redis_url=redis_url,
                client_kwargs=client_kwargs,
                ttl=ttl,
            )
        else:
            self.redis_store = NillStore()
        self.prefix = prefix + "/" if prefix else ""

    def _get_full_path(self, key: str) -> Path:
        pkey = self.prefix + key
        return self.file_store._get_full_path(pkey)

    def mget(self, keys: Sequence[str]) -> list[Optional[Document]]:
        """Get the values associated with the given keys.

        Args:
            keys (Sequence[str]): A sequence of keys.

        Returns:
            A sequence of optional Documents associated with the keys.
            If a key is not found, the corresponding Document will be None.
        """
        docs: list[Optional[Document]] = []
        for key in keys:
            pkey = self.prefix + key
            value = self.redis_store.mget([pkey])[0]
            if not value or self.update_atime:
                value = self.file_store.mget([pkey])[0]
                if value:
                    self.redis_store.mset([(pkey, value)])
            doc = None if not value else pickle.loads(value)
            docs.append(doc)
        return docs

    def mset(self, key_doc_pairs: Sequence[tuple[str, Document]]) -> None:
        """Set the documents for the given keys.

        Args:
            key_doc_pairs (Sequence[tuple[str, Document]]): A sequence of key-document pairs.
        """
        for key, doc in key_doc_pairs:
            pkey = self.prefix + key
            value = pickle.dumps(doc)
            self.file_store.mset([(pkey, value)])
            self.redis_store.mset([(pkey, value)])

    def mdelete(self, keys: Sequence[str]) -> None:
        """Delete the given keys and their associated Documents.

        Args:
            keys (Sequence[str]): A sequence of keys to delete.
        """
        pkeys = [self.prefix + key for key in keys]
        self.file_store.mdelete(pkeys)
        self.redis_store.mdelete(pkeys)

    def yield_keys(self, prefix: Optional[str] = None) -> Iterator[str]:
        """Get an iterator over keys that match the given prefix.

        Args:
            prefix (str): The prefix to match.

        Yields:
            Iterator[str]: An iterator over keys that match the given prefix.
        """
        prefix = self.prefix + prefix if prefix else self.prefix
        for pkey in self.file_store.yield_keys(prefix=prefix):
            key = pkey.replace(self.prefix, "", 1)
            yield key
