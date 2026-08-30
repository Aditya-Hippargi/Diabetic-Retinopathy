"""
db_supabase.py
===============
Production database layer for the Supabase (Postgres) backend.
Implements the schema defined in deploy/schema_supabase.sql.
"""

from auth_supabase import determine_initial_approval, VALID_ROLES


def get_connection():
    """
    Placeholder — real implementation added later,
    once a live Supabase project and connection string exist.
    """
    raise NotImplementedError(
        "Supabase connection not yet configured — see deploy plan Section C."
    )


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