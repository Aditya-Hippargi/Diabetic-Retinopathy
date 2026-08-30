import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

SUPABASE_DB_HOST = os.getenv("SUPABASE_DB_HOST")
SUPABASE_DB_PORT = os.getenv("SUPABASE_DB_PORT", "5432")
SUPABASE_DB_NAME = os.getenv("SUPABASE_DB_NAME", "postgres")
SUPABASE_DB_USER = os.getenv("SUPABASE_DB_USER", "postgres")
SUPABASE_DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD")


def get_connection():
    """
    Returns a live connection to the Supabase Postgres database,
    built from individual env vars to avoid DSN string parsing issues.
    """
    missing = [name for name, val in [
        ("SUPABASE_DB_HOST", SUPABASE_DB_HOST),
        ("SUPABASE_DB_PASSWORD", SUPABASE_DB_PASSWORD),
    ] if not val]
    if missing:
        raise EnvironmentError(f"Missing required env vars: {', '.join(missing)}")

    return psycopg2.connect(
        host=SUPABASE_DB_HOST,
        port=SUPABASE_DB_PORT,
        dbname=SUPABASE_DB_NAME,
        user=SUPABASE_DB_USER,
        password=SUPABASE_DB_PASSWORD,
    )