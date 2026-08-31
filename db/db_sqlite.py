import sqlite3
import os
import sys

# Ensure src/ is importable regardless of where the process is launched from
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

from new_database import (
    get_connection as _get_connection,
    setup_new_database,
    insert_new_scan,
    get_all_new_scans,
    get_new_stats,
    search_new_scans,
    delete_new_scan,
    get_new_scan_by_id,
    register_user,
    verify_user,
)

SQLITE_PATH = os.path.join(os.getcwd(), 'retinascan_ai.db')


def get_connection():
    """
    Returns a database connection using the existing dual-backend
    logic in src/new_database.py (MySQL with SQLite fallback).
    """
    return _get_connection()