import os
from dotenv import load_dotenv

load_dotenv()

DB_BACKEND = os.getenv("DB_BACKEND", "sqlite").lower()


def get_db_connection():
    """
    Returns a database connection based on DB_BACKEND.
    Currently supports: sqlite (mysql).
    """
    if DB_BACKEND == "sqlite":
        from db.db_sqlite import get_connection
        return get_connection()
    else:
        raise ValueError(f"Unsupported DB_BACKEND: {DB_BACKEND}")