import logging
import os
import traceback
from datetime import (
    datetime,
    timezone,
)
from typing import (
    Dict,
    List,
    Union,
)

from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.exc import NoSuchColumnError
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
    dir = Column(String(255))
    file = Column(String(255))
    extension = Column(String(50))
    error_type = Column(String(100))
    error_message = Column(Text)
    error_traceback = Column(Text)


class FileErrorDB:
    def __init__(self, db_file):
        self.engine = create_engine(f"sqlite:///{db_file}", echo=False)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def add_file_error(self, file_path, error):
        dir, file = os.path.split(file_path)
        extension = os.path.splitext(file)[1][1:]
        error_type = type(error).__name__
        error_message = str(error)
        error_traceback = traceback.format_exc()

        print(f"***Error: {error_message}")

        self._insert_error(
            dir, file, extension, error_type, error_message, error_traceback
        )

    def _insert_error(
        self, dir, file, extension, error_type, error_message, error_traceback
    ):
        session = self.Session()
        try:
            new_error = FileError(
                dir=dir,
                file=file,
                extension=extension,
                error_type=error_type,
                error_message=error_message,
                error_traceback=error_traceback,
            )
            session.add(new_error)
            session.commit()
        except Exception as e:
            session.rollback()
            logging.error(f"Error inserting into database: {str(e)}")
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
            logging.error(f"Error deleting entry from database: {str(e)}")
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
            logging.error(f"Error cleaning database: {str(e)}")
            return False
        finally:
            session.close()

    def _row_to_dict(self, row: FileError) -> Dict:
        return {
            "id": row.id,
            "timestamp": row.timestamp,
            "dir": row.dir,
            "file": row.file,
            "extension": row.extension,
            "error_type": row.error_type,
            "error_message": row.error_message,
            "error_traceback": row.error_traceback,
        }

    def _group_results(self, rows, group_by):
        if group_by not in FileError.__table__.columns:
            raise ValueError(f"Invalid group_by field: {group_by}")

        grouped_results = {}
        for row in rows:
            key = getattr(row, group_by)
            if key not in grouped_results:
                grouped_results[key] = []
            grouped_results[key].append(self._row_to_dict(row))
        return grouped_results

    def get_all_errors(
        self, group_by: str = None
    ) -> Union[List[Dict], Dict[str, List[Dict]]]:
        session: Session = self.Session()
        try:
            query = session.query(FileError)
            if group_by:
                query = query.group_by(getattr(FileError, group_by))
            rows = query.all()
            if group_by:
                return self._group_results(rows, group_by)
            return [self._row_to_dict(row) for row in rows]
        except NoSuchColumnError:
            raise ValueError(f"Invalid group_by field: {group_by}")
        finally:
            session.close()

    def get_errors_by_type(
        self, error_type: str, group_by: str = None
    ) -> Union[List[Dict], Dict[str, List[Dict]]]:
        session: Session = self.Session()
        try:
            query = session.query(FileError).filter_by(error_type=error_type)
            if group_by:
                query = query.group_by(getattr(FileError, group_by))
            rows = query.all()
            if group_by:
                return self._group_results(rows, group_by)
            return [self._row_to_dict(row) for row in rows]
        except NoSuchColumnError:
            raise ValueError(f"Invalid group_by field: {group_by}")
        finally:
            session.close()

    def get_errors_by_extension(
        self, file_extension: str, group_by: str = None
    ) -> Union[List[Dict], Dict[str, List[Dict]]]:
        session: Session = self.Session()
        try:
            query = session.query(FileError).filter_by(file_extension=file_extension)
            if group_by:
                query = query.group_by(getattr(FileError, group_by))
            rows = query.all()
            if group_by:
                return self._group_results(rows, group_by)
            return [self._row_to_dict(row) for row in rows]
        except NoSuchColumnError:
            raise ValueError(f"Invalid group_by field: {group_by}")
        finally:
            session.close()

    def get_errors_by_date_range(
        self, start_date: datetime, end_date: datetime, group_by: str = None
    ) -> Union[List[Dict], Dict[str, List[Dict]]]:
        session: Session = self.Session()
        try:
            query = session.query(FileError).filter(
                FileError.timestamp >= start_date, FileError.timestamp <= end_date
            )
            if group_by:
                query = query.group_by(getattr(FileError, group_by))
            rows = query.all()
            if group_by:
                return self._group_results(rows, group_by)
            return [self._row_to_dict(row) for row in rows]
        except NoSuchColumnError:
            raise ValueError(f"Invalid group_by field: {group_by}")
        finally:
            session.close()
