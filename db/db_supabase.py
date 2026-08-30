"""
db_supabase.py
===============
Production database layer for the Supabase (Postgres) backend.
Implements the schema defined in deploy/schema_supabase.sql.
"""

from auth_supabase import determine_initial_approval, VALID_ROLES
from auth_supabase import can_view_all_records


import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")


def get_connection():
    """
    Returns a live connection to the Supabase Postgres database.
    Requires SUPABASE_DB_URL to be set in .env (see .env.example).
    """
    if not SUPABASE_DB_URL:
        raise EnvironmentError(
            "SUPABASE_DB_URL is not set. Add it to your .env file."
        )
    return psycopg2.connect(SUPABASE_DB_URL)


def register_user(conn, username: str, password_hash: str, role: str = 'patient') -> bool:
    """
    Insert a new user with role-aware approval defaults.
    Returns True on success, False on failure (e.g. duplicate username).
    """
    if role not in VALID_ROLES:
        raise ValueError(f"Invalid role: {role}")

    is_approved = determine_initial_approval(role)

    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO users (username, password_hash, role, is_approved)
            VALUES (%s, %s, %s, %s)
            """,
            (username, password_hash, role, is_approved),
        )
        conn.commit()
        return True
    except Exception as e:
        print(f"[Supabase] User registration error: {e}")
        return False
    finally:
        cursor.close()


def verify_user(conn, username: str, password_hash: str):
    """
    Verify credentials and return the user's row (id, role, is_approved)
    on success, or None on failure. Unlike the legacy verify_user, this
    returns role/approval info since the app needs it immediately after login.
    """
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, role, is_approved FROM users
            WHERE username = %s AND password_hash = %s
            """,
            (username, password_hash),
        )
        row = cursor.fetchone()
        return row  # (id, role, is_approved) or None
    except Exception as e:
        print(f"[Supabase] Verify user error: {e}")
        return None
    finally:
        cursor.close()

def insert_scan(conn, patient_name, patient_age, eye_side, grade, grade_name,
                 confidence, all_probabilities, gradcam_path, model_version,
                 risk_level, notes, created_by_id):
    """Insert a new scan record, linked to the user who ran it."""
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO scans
                (patient_name, patient_age, eye_side, grade, grade_name,
                 confidence, all_probabilities, gradcam_path, model_version,
                 risk_level, notes, created_by)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (patient_name, patient_age, eye_side, grade, grade_name,
             confidence, all_probabilities, gradcam_path, model_version,
             risk_level, notes, created_by_id),
        )
        record_id = cursor.fetchone()[0]
        conn.commit()
        return record_id
    except Exception as e:
        print(f"[Supabase] Insert scan error: {e}")
        return None
    finally:
        cursor.close()


def get_scans(conn, requesting_user_id: int, requesting_role: str):
    """
    Fetch scan records, automatically scoped by role:
      - doctor/researcher/admin: all records
      - patient: only records they created
    This scoping happens here, not in the UI layer, so every
    caller gets it correctly without needing to remember to filter.
    """
    try:
        cursor = conn.cursor()
        if can_view_all_records(requesting_role):
            cursor.execute("SELECT * FROM scans ORDER BY scan_date DESC")
        else:
            cursor.execute(
                "SELECT * FROM scans WHERE created_by = %s ORDER BY scan_date DESC",
                (requesting_user_id,),
            )
        return cursor.fetchall()
    except Exception as e:
        print(f"[Supabase] Get scans error: {e}")
        return []
    finally:
        cursor.close()