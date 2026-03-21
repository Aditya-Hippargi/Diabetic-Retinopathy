"""
RetinaScan AI — MySQL Database Module
Handles all database operations for patient scan records.
CRUD: Create, Read, Update, Delete
"""

import mysql.connector
from mysql.connector import Error
import datetime

# ── Database Configuration ────────────────────────────────────────────────────
DB_CONFIG = {
    'host'    : 'localhost',
    'port'    : 3306,
    'user'    : 'root',
    'password': 'jeet123',
    'database': 'retinascan_db'
}

GRADE_NAMES = [
    'No DR',
    'Mild DR',
    'Moderate DR',
    'Severe DR',
    'Proliferative DR'
]

# ── Connection ────────────────────────────────────────────────────────────────
def get_connection():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except Error as e:
        print(f"Connection error: {e}")
        return None


# ── Setup ─────────────────────────────────────────────────────────────────────
def setup_database():
    try:
        conn = mysql.connector.connect(
            host     = DB_CONFIG['host'],
            port     = DB_CONFIG['port'],
            user     = DB_CONFIG['user'],
            password = DB_CONFIG['password']
        )
        cursor = conn.cursor()
        cursor.execute("CREATE DATABASE IF NOT EXISTS retinascan_db")
        cursor.execute("USE retinascan_db")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                id           INT AUTO_INCREMENT PRIMARY KEY,
                patient_name VARCHAR(100) NOT NULL,
                patient_age  INT,
                eye_side     VARCHAR(20),
                grade        INT NOT NULL,
                grade_name   VARCHAR(50) NOT NULL,
                confidence   FLOAT NOT NULL,
                scan_date    DATETIME NOT NULL,
                notes        TEXT
            )
        """)
        conn.commit()
        cursor.close()
        conn.close()
        print("Database connected successfully!")
    except Error as e:
        print(f"Setup error: {e}")


# ── CREATE ────────────────────────────────────────────────────────────────────
def insert_scan(patient_name, patient_age, eye_side, grade, confidence, notes=None):
    conn = get_connection()
    if not conn:
        return None
    try:
        cursor = conn.cursor()
        query = """
            INSERT INTO patients
                (patient_name, patient_age, eye_side, grade, grade_name, confidence, scan_date, notes)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        values = (
            patient_name, patient_age, eye_side, grade,
            GRADE_NAMES[grade], round(confidence, 4),
            datetime.datetime.now(), notes
        )
        cursor.execute(query, values)
        conn.commit()
        record_id = cursor.lastrowid
        print(f"Record inserted! ID: {record_id} | Patient: {patient_name} | Grade: {GRADE_NAMES[grade]}")
        return record_id
    except Error as e:
        print(f"Insert error: {e}")
        return None
    finally:
        cursor.close()
        conn.close()


# ── READ ──────────────────────────────────────────────────────────────────────
def get_all_scans():
    conn = get_connection()
    if not conn:
        return []
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM patients ORDER BY scan_date DESC")
        return cursor.fetchall()
    except Error as e:
        print(f"Read error: {e}")
        return []
    finally:
        cursor.close()
        conn.close()


def get_scan_by_id(record_id):
    conn = get_connection()
    if not conn:
        return None
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM patients WHERE id = %s", (record_id,))
        return cursor.fetchone()
    except Error as e:
        print(f"Read error: {e}")
        return None
    finally:
        cursor.close()
        conn.close()


def get_scans_by_name(patient_name):
    conn = get_connection()
    if not conn:
        return []
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM patients WHERE patient_name LIKE %s ORDER BY scan_date DESC",
            (f"%{patient_name}%",)
        )
        return cursor.fetchall()
    except Error as e:
        print(f"Search error: {e}")
        return []
    finally:
        cursor.close()
        conn.close()


def get_stats():
    conn = get_connection()
    if not conn:
        return {}
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM patients")
        total = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM patients WHERE grade > 0")
        dr_detected = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM patients WHERE grade >= 3")
        severe = cursor.fetchone()[0]
        cursor.execute("SELECT AVG(confidence) FROM patients")
        avg_conf = cursor.fetchone()[0]
        return {
            'total'      : total,
            'dr_detected': dr_detected,
            'severe'     : severe,
            'avg_conf'   : round(avg_conf, 4) if avg_conf else 0
        }
    except Error as e:
        print(f"Stats error: {e}")
        return {}
    finally:
        cursor.close()
        conn.close()


# ── UPDATE ────────────────────────────────────────────────────────────────────
def update_notes(record_id, notes):
    conn = get_connection()
    if not conn:
        return False
    try:
        cursor = conn.cursor()
        cursor.execute("UPDATE patients SET notes = %s WHERE id = %s", (notes, record_id))
        conn.commit()
        if cursor.rowcount > 0:
            print(f"Record {record_id} updated successfully!")
            return True
        else:
            print(f"Record {record_id} not found.")
            return False
    except Error as e:
        print(f"Update error: {e}")
        return False
    finally:
        cursor.close()
        conn.close()


def update_scan(record_id, patient_name=None, patient_age=None, eye_side=None, notes=None):
    conn = get_connection()
    if not conn:
        return False
    try:
        cursor = conn.cursor()
        fields, values = [], []
        if patient_name is not None:
            fields.append("patient_name = %s"); values.append(patient_name)
        if patient_age is not None:
            fields.append("patient_age = %s");  values.append(patient_age)
        if eye_side is not None:
            fields.append("eye_side = %s");     values.append(eye_side)
        if notes is not None:
            fields.append("notes = %s");        values.append(notes)
        if not fields:
            print("Nothing to update.")
            return False
        values.append(record_id)
        cursor.execute(f"UPDATE patients SET {', '.join(fields)} WHERE id = %s", values)
        conn.commit()
        if cursor.rowcount > 0:
            print(f"Record {record_id} updated successfully!")
            return True
        else:
            print(f"Record {record_id} not found.")
            return False
    except Error as e:
        print(f"Update error: {e}")
        return False
    finally:
        cursor.close()
        conn.close()


# ── DELETE ────────────────────────────────────────────────────────────────────
def delete_scan(record_id):
    conn = get_connection()
    if not conn:
        return False
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM patients WHERE id = %s", (record_id,))
        conn.commit()
        if cursor.rowcount > 0:
            print(f"Record {record_id} deleted successfully!")
            return True
        else:
            print(f"Record {record_id} not found.")
            return False
    except Error as e:
        print(f"Delete error: {e}")
        return False
    finally:
        cursor.close()
        conn.close()


def delete_all_scans():
    conn = get_connection()
    if not conn:
        return False
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM patients")
        conn.commit()
        print(f"All records deleted. Rows affected: {cursor.rowcount}")
        return True
    except Error as e:
        print(f"Delete error: {e}")
        return False
    finally:
        cursor.close()
        conn.close()


# ── Display Helper ────────────────────────────────────────────────────────────
def print_all_scans():
    rows = get_all_scans()
    if not rows:
        print("No records found.")
        return
    print("\n" + "="*95)
    print(f"{'ID':<5} {'Name':<20} {'Age':<5} {'Eye':<12} {'Grade':<20} {'Confidence':<12} {'Date'}")
    print("="*95)
    for row in rows:
        rid, name, age, eye, grade, grade_name, conf, date, notes = row
        print(f"{rid:<5} {str(name):<20} {str(age):<5} {str(eye):<12} {str(grade_name):<20} {conf*100:<11.1f}% {date}")
    print("="*95)
    print(f"Total records: {len(rows)}\n")


# ── Interactive Menu ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\nRetinaScan AI — Patient Database")
    print("="*40)

    setup_database()

    while True:
        print("\nOptions:")
        print("1. Add new patient scan")
        print("2. View all records")
        print("3. Search by patient name")
        print("4. Update patient notes")
        print("5. Delete a record")
        print("6. View statistics")
        print("7. Clear all records")
        print("8. Exit")

        choice = input("\nEnter choice (1-8): ").strip()

        if choice == '1':
            print("\n--- Add New Patient ---")
            name  = input("Patient name                        : ").strip()
            age   = int(input("Patient age                         : ").strip())
            eye   = input("Eye side (Left Eye/Right Eye/Both)  : ").strip()
            print("DR Grade: 0=No DR, 1=Mild, 2=Moderate, 3=Severe, 4=Proliferative")
            grade = int(input("DR Grade (0-4)                      : ").strip())
            conf  = float(input("Confidence (0.0 - 1.0)             : ").strip())
            notes = input("Notes (press Enter to skip)         : ").strip()
            insert_scan(name, age, eye, grade, conf, notes or None)

        elif choice == '2':
            print("\n--- All Patient Records ---")
            print_all_scans()

        elif choice == '3':
            name    = input("Enter patient name to search: ").strip()
            results = get_scans_by_name(name)
            if results:
                print(f"\nFound {len(results)} record(s):")
                for r in results:
                    rid, rname, age, eye, grade, grade_name, conf, date, notes = r
                    print(f"  ID:{rid} | {rname} | Age:{age} | {grade_name} | {conf*100:.1f}% | {date}")
            else:
                print("No records found.")

        elif choice == '4':
            print_all_scans()
            record_id = int(input("Enter record ID to update: ").strip())
            notes     = input("Enter new notes           : ").strip()
            update_notes(record_id, notes)

        elif choice == '5':
            print_all_scans()
            record_id = int(input("Enter record ID to delete: ").strip())
            confirm   = input(f"Are you sure? (yes/no)   : ").strip()
            if confirm.lower() == 'yes':
                delete_scan(record_id)
            else:
                print("Delete cancelled.")

        elif choice == '6':
            stats = get_stats()
            print("\n--- Database Statistics ---")
            print(f"  Total scans    : {stats['total']}")
            print(f"  DR detected    : {stats['dr_detected']}")
            print(f"  Severe cases   : {stats['severe']}")
            print(f"  Avg confidence : {stats['avg_conf']*100:.1f}%")

        elif choice == '7':
            confirm = input("Delete ALL records? This cannot be undone. (yes/no): ").strip()
            if confirm.lower() == 'yes':
                delete_all_scans()
            else:
                print("Cancelled.")

        elif choice == '8':
            print("Goodbye!")
            break

        else:
            print("Invalid choice. Please enter 1-8.")
