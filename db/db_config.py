import os
from dotenv import load_dotenv

load_dotenv()

DB_BACKEND = os.getenv("DB_BACKEND", "sqlite").lower()