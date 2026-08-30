-- RetinaScan AI — Supabase (Postgres) Production Schema

-- ── Users ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS users (
    id              BIGSERIAL PRIMARY KEY,
    username        TEXT UNIQUE NOT NULL,
    password_hash   TEXT NOT NULL,
    role            TEXT NOT NULL DEFAULT 'patient'
                        CHECK (role IN ('patient', 'doctor', 'researcher', 'admin')),
    is_approved     BOOLEAN NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Admins and the seed account should be approved by default;
-- self-registered doctor/researcher/patient accounts start unapproved.

-- ── Scans ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS scans (
    id                  BIGSERIAL PRIMARY KEY,
    patient_name        TEXT NOT NULL,
    patient_age         INTEGER,
    eye_side            TEXT,
    grade               INTEGER NOT NULL,
    grade_name          TEXT NOT NULL,
    confidence          REAL NOT NULL,
    all_probabilities   JSONB,
    gradcam_path         TEXT,   -- storage URL (Supabase Storage), NOT a local filesystem path
    model_version       TEXT NOT NULL DEFAULT 'EfficientNetB4_v82pct',
    risk_level          TEXT,
    scan_date           TIMESTAMPTZ NOT NULL DEFAULT now(),
    notes               TEXT,
    created_by          BIGINT REFERENCES users(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_scans_patient_name ON scans (patient_name);
CREATE INDEX IF NOT EXISTS idx_scans_grade        ON scans (grade);
CREATE INDEX IF NOT EXISTS idx_scans_model        ON scans (model_version);
CREATE INDEX IF NOT EXISTS idx_scans_scan_date    ON scans (scan_date);
CREATE INDEX IF NOT EXISTS idx_scans_created_by   ON scans (created_by);