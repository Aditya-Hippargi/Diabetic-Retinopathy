import sqlite3
import os

SQLITE_PATH = os.path.join(os.getcwd(), 'retinascan_ai.db')


def get_connection():
    """
    Returns a SQLite connection to the app's database.
    Mirrors the path logic in src/new_database.py exactly.
    """
    return sqlite3.connect(SQLITE_PATH, check_same_thread=False)