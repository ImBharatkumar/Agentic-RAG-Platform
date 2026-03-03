# External libraries
from sqlalchemy import (
create_engine,
Column,
Integer,
Text,
JSON,
event,
)
from sqlalchemy.orm import declarative_base, sessionmaker
from pgvector.sqlalchemy import Vector
from pgvector.psycopg import register_vector
from enterprice_rag.config.settings import DATABASE_URL, EMBED_DIM
# ---------- DB setup ----------
Base = declarative_base()


class Chunk(Base):
    __tablename__ = "chunks"
    id = Column(Integer, primary_key=True)
    doc_id = Column(Text, index=True)
    chunk_id = Column(Integer, index=True)
    content = Column(Text)
    context = Column(Text)  # Anthropic Contextual Retrieval
    metadatas = Column(JSON)
    embedding = Column(Vector(EMBED_DIM))
    search_vector = Column(Text)  # For keyword search (PostgreSQL tsvector)


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
