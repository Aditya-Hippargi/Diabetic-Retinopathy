"""
auth_supabase.py
=================
Role-aware registration and access logic for the Supabase-backed
production path. Mirrors new_database.py's auth functions but adds
role + approval handling per the schema in deploy/schema_supabase.sql.
"""

VALID_ROLES = ('patient', 'doctor', 'researcher', 'admin')


def determine_initial_approval(role: str) -> bool:
    """
    Patients are auto-approved (they only ever see their own results).
    Doctor/researcher accounts require manual admin approval before
    they can use Scan & Predict. Admin accounts are seeded separately,
    never through self-registration.
    """
    if role == 'patient':
        return True
    if role in ('doctor', 'researcher'):
        return False
    raise ValueError(f"Invalid role: {role}")


def can_access_scan_and_predict(role: str, is_approved: bool) -> bool:
    """Gate for the Scan & Predict page."""
    if role == 'patient':
        return False
    return role in ('doctor', 'researcher') and is_approved


def can_view_all_records(role: str) -> bool:
    """
    Patients only ever see their own records (queries get filtered
    by created_by elsewhere). Doctors/researchers see everything.
    """
    return role in ('doctor', 'researcher', 'admin')