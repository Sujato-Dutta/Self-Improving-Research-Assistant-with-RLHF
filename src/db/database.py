import os
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session
from src.config import config

engine = create_engine(
    config.db_url,
    connect_args={"check_same_thread": False} if config.db_url.startswith("sqlite") else {},
    echo=False
)

SessionLocal = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    from src.db import models  # Ensure all models are registered
    Base.metadata.create_all(bind=engine)
