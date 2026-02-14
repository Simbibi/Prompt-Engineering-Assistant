# api.py — exposes variables used across the project
from settings import get_settings

settings = get_settings()

# Prefer `api_key` but fall back to `llm_vender_secret` if needed
OPENAI_API_KEY = settings.api_key or settings.llm_vender_secret
OPENAI_API_BASE = settings.llm_base_url

