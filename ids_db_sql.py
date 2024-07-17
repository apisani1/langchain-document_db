import os
from sqlalchemy import create_engine, Column, String, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from typing import List

Base = declarative_base()

class IDEntry(Base):
    __tablename__ = 'id_entries'

    key = Column(String, primary_key=True)
    namespace = Column(String, primary_key=True)
    ids = Column(String)

class IDsDB:
    def __init__(self, db_path=''):
        db_name = 'ids_db.sqlite'
        db_url = f'sqlite:///{os.path.join(db_path, db_name)}'
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def add_ids(self, key: str, ids: List[str], namespace: str = ""):
        if ids and isinstance(ids, list):
            session = self.Session()
            try:
                entry = session.query(IDEntry).filter_by(key=key, namespace=namespace).first()
                if entry:
                    existing_ids = entry.ids.split(',') if entry.ids else []
                    existing_ids.extend(ids)
                    entry.ids = ','.join(existing_ids)
                else:
                    new_entry = IDEntry(key=key, namespace=namespace, ids=','.join(ids))
                    session.add(new_entry)
                session.commit()
            finally:
                session.close()

    def get_ids(self, key: str, namespace: str = "") -> List[str]:
        session = self.Session()
        try:
            entry = session.query(IDEntry).filter_by(key=key, namespace=namespace).first()
            if entry and entry.ids:
                return entry.ids.split(',')
            return []
        finally:
            session.close()

    def delete_ids(self, key: str, namespace: str = ""):
        session = self.Session()
        try:
            entry = session.query(IDEntry).filter_by(key=key, namespace=namespace).first()
            if entry:
                session.delete(entry)
                session.commit()
        finally:
            session.close()
