import os
from dotenv import load_dotenv
load_dotenv()
API_SECRET_KEY = os.getenv("API_SECRET_KEY", "")
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "25"))
MAX_INPUT_PIXELS = int(os.getenv("MAX_INPUT_PIXELS", "4000000"))
FREE_TIER_LIMIT = int(os.getenv("FREE_TIER_LIMIT", "5"))
FILE_TTL_SECONDS = int(os.getenv("FILE_TTL_SECONDS", "3600"))
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "http://localhost:8000")
