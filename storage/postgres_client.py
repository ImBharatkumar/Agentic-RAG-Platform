# External libraries
from sqlalchemy import (
create_engine,
Column,
Integer,
Text,
JSON,
event,
text as sql_text,
)
from sqlalchemy.orm import declarative_base, sessionmaker
from pgvector.sqlalchemy import Vector
from pgvector.psycopg import register_vector
from config.settings import DATABASE_URL, EMBED_DIM
# ---------- DB setup ----------
Base = declarative_base()


class Chunk(Base):
    __tablename__ = "chunks"
    id = Column(Integer, primary_key=True)
    doc_id = Column(Text, index=True)
    chunk_id = Column(Integer, index=True)
    content = Column(Text)
    metadatas = Column(JSON)
    embedding = Column(Vector(EMBED_DIM))


# Create engine and register vector type on connect
engine = create_engine(DATABASE_URL)


@event.listens_for(engine, "connect")
def _register_vector(conn, rec):
# register_vector adapts python lists to pgvector, required for psycopg
    try:
        register_vector(conn)
    except Exception:
    # if already registered or using a different driver, ignore
        pass


SessionLocal = sessionmaker(bind=engine)
