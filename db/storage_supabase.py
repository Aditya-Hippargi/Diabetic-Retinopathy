"""
storage_supabase.py
====================
Uploads Grad-CAM composite images to Supabase Storage and returns
a public URL, replacing local filesystem paths (which don't
persist on Cloud Run).
"""

import os
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
BUCKET_NAME = "gradcam-images"

_client = None


def get_storage_client():
    global _client
    if _client is None:
        if not SUPABASE_URL or not SUPABASE_ANON_KEY:
            raise EnvironmentError(
                "SUPABASE_URL and SUPABASE_ANON_KEY must be set in .env for Storage uploads."
            )
        _client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    return _client


def upload_gradcam_image(local_path: str, remote_filename: str) -> str:
    """
    Uploads a local image file to Supabase Storage and returns its public URL.

    Args:
        local_path: Path to the image file on local/container disk
        remote_filename: Desired filename in the bucket (should be unique)

    Returns:
        Public URL string, or None on failure
    """
    try:
        client = get_storage_client()
        with open(local_path, "rb") as f:
            client.storage.from_(BUCKET_NAME).upload(
                remote_filename, f, {"content-type": "image/png"}
            )
        return client.storage.from_(BUCKET_NAME).get_public_url(remote_filename)
    except Exception as e:
        print(f"[Storage] Upload error: {e}")
        return None