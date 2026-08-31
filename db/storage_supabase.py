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
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
BUCKET_NAME = "gradcam-images"

_client = None


def get_storage_client():
    global _client
    if _client is None:
        if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
            raise EnvironmentError(
                "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in .env for Storage uploads."
            )
        _client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    return _client


def upload_gradcam_image(local_path: str, remote_filename: str) -> str:
    """
    Uploads a local image file to Supabase Storage and returns its public URL.
    """
    try:
        client = get_storage_client()
        with open(local_path, "rb") as f:
            file_bytes = f.read()

        result = client.storage.from_(BUCKET_NAME).upload(
            path=remote_filename,
            file=file_bytes,
            file_options={"content-type": "image/png"},
        )
        print(f"[Storage] Upload result: {result}")

        url = client.storage.from_(BUCKET_NAME).get_public_url(remote_filename)
        print(f"[Storage] Public URL: {url}")
        return url

    except Exception as e:
        print(f"[Storage] Upload error: {type(e).__name__}: {e}")
        return None