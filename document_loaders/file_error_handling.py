import traceback
from datetime import (
    datetime,
    timezone,
)
from typing import (
    Dict,
    List,
)

from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import (
    Session,
    sessionmaker,
)


Base = declarative_base()


class FileError(Base):
    __tablename__ = "file_errors"

    id = Column(Integer, primary_key=True)
    timestamp = Column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    root = Column(String(255))
    file = Column(String(255))
    file_extension = Column(String(50))
    error_type = Column(String(100))
    error_message = Column(Text)
    error_traceback = Column(Text)


class FileErrorDB:
    def __init__(self, db_file):
        self.engine = create_engine(f"sqlite:///{db_file}", echo=False)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def add_file_error(self, root, file, error):
        error_type = type(error).__name__
        error_message = str(error)
        error_traceback = traceback.format_exc()
        file_extension = file.split(".")[-1] if "." in file else ""

        self._insert_error(
            root, file, file_extension, error_type, error_message, error_traceback
        )

    def _insert_error(
        self, root, file, file_extension, error_type, error_message, error_traceback
    ):
        session = self.Session()
        try:
            new_error = FileError(
                root=root,
                file=file,
                file_extension=file_extension,
                error_type=error_type,
                error_message=error_message,
                error_traceback=error_traceback,
            )
            session.add(new_error)
            session.commit()
        except Exception as e:
            session.rollback()
            print(f"Error inserting into database: {str(e)}")
        finally:
            session.close()

    def delete_error_by_id(self, error_id):
        session = self.Session()
        try:
            error = session.query(FileError).filter_by(id=error_id).first()
            if error:
                session.delete(error)
                session.commit()
                return True
            else:
                return False
        except Exception as e:
            session.rollback()
            print(f"Error deleting entry from database: {str(e)}")
            return False
        finally:
            session.close()

    def clean_database(self):
        session = self.Session()
        try:
            session.query(FileError).delete()
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            print(f"Error cleaning database: {str(e)}")
            return False
        finally:
            session.close()

    def _row_to_dict(self, row: FileError) -> Dict:
        return {
            "id": row.id,
            "timestamp": row.timestamp,
            "root": row.root,
            "file": row.file,
            "file_extension": row.file_extension,
            "error_type": row.error_type,
            "error_message": row.error_message,
            "error_traceback": row.error_traceback,
        }

    def get_all_errors(self) -> List[Dict]:
        session: Session = self.Session()
        try:
            rows = session.query(FileError).all()
            return [self._row_to_dict(row) for row in rows]
        finally:
            session.close()

    def get_errors_by_type(self, error_type: str) -> List[Dict]:
        session: Session = self.Session()
        try:
            rows = session.query(FileError).filter_by(error_type=error_type).all()
            return [self._row_to_dict(row) for row in rows]
        finally:
            session.close()

    def get_errors_by_extension(self, file_extension: str) -> List[Dict]:
        session: Session = self.Session()
        try:
            rows = (
                session.query(FileError).filter_by(file_extension=file_extension).all()
            )
            return [self._row_to_dict(row) for row in rows]
        finally:
            session.close()

    def get_errors_by_date_range(
        self, start_date: datetime, end_date: datetime
    ) -> List[Dict]:
        session: Session = self.Session()
        try:
            rows = (
                session.query(FileError)
                .filter(
                    FileError.timestamp >= start_date, FileError.timestamp <= end_date
                )
                .all()
            )
            return [self._row_to_dict(row) for row in rows]
        finally:
            session.close()
