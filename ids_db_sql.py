import os
from sqlalchemy import create_engine, Column, String
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
from typing import List

Base = declarative_base()


class IDEntry(Base):
    __tablename__ = "id_entries"

    key = Column(String, primary_key=True)
    namespace = Column(String, primary_key=True)
    ids = Column(String)


class IDsDB:
    """
    A class to manage IDs in a database. It associates a string key with a list of string IDs.
    It uses SQLAlchemy to interact with the database.

    Args:
        db_path (str, optional): The path to the database. Defaults to the current directory.
        db_name (str, optional): The name of the database. Defaults to "ids_db.sqlite".
    """

    def __init__(self, db_path: str = "", db_name: str = "ids_db.sqlite") -> None:
        db_url = f"sqlite:///{os.path.join(db_path, db_name)}"
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def add_ids(self, key: str, ids: List[str], namespace: str = "") -> None:
        """
        Add IDs to the database.

        Args:
            key (str): The key for the IDs to add.
            ids (List[str]): List of IDs to add.
            namespace (str, optional): The namespace for the IDs. Defaults to "".

        Return:
            None
        """
        if ids and isinstance(ids, list):
            session = self.Session()
            try:
                entry = (
                    session.query(IDEntry)
                    .filter_by(key=key, namespace=namespace)
                    .first()
                )
                if entry:
                    existing_ids = entry.ids.split(",") if entry.ids else []
                    existing_ids.extend(ids)
                    entry.ids = ",".join(existing_ids)
                else:
                    new_entry = IDEntry(key=key, namespace=namespace, ids=",".join(ids))
                    session.add(new_entry)
                session.commit()
            finally:
                session.close()

    def get_ids(self, key: str, namespace: str = "") -> List[str]:
        """
        Get IDs from the database.

        Args:
            key (str): The key for the IDs to get.
            namespace (str, optiona): The namespace for the IDs. Defaults to "".

        Return:
            List[str]: List of IDs associated with the key.
        """
        session = self.Session()
        try:
            entry = (
                session.query(IDEntry).filter_by(key=key, namespace=namespace).first()
            )
            if entry and entry.ids:
                return entry.ids.split(",")
            return []
        finally:
            session.close()

    def delete_ids(self, key: str, namespace: str = "") -> None:
        """
        Delete IDs from the database.

        Args:
            key (str): The key for the IDs to delete.
            namespace (str, optional): The namespace for the IDs. Defaults to "".

        Return:
            None
        """
        session = self.Session()
        try:
            entry = (
                session.query(IDEntry).filter_by(key=key, namespace=namespace).first()
            )
            if entry:
                session.delete(entry)
                session.commit()
        finally:
            session.close()
