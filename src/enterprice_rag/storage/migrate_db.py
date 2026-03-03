from sqlalchemy import create_engine, text
from enterprice_rag.config.settings import DATABASE_URL

def migrate():
    engine = create_engine(DATABASE_URL)
    with engine.begin() as conn:
        print("Adding 'context' column...")
        conn.execute(text("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS context TEXT"))
        print("Adding 'search_vector' column...")
        conn.execute(text("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS search_vector TEXT"))
        print("Migration complete.")

if __name__ == "__main__":
    migrate()
