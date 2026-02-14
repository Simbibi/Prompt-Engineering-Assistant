from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import AnyUrl, Field, SecretStr
from pydantic_settings import BaseSettings

# Base dir for locating the .env file (project root)
BASE_DIR = Path(__file__).parent


class Settings(BaseSettings):
    # secrets
    api_key: Optional[SecretStr] = Field(None, env="API_KEY")
    llm_vender_secret: Optional[SecretStr] = Field(None, env="LLM_VENDER_SECRET")

    # app
    port: int = Field(8000, ge=1, le=65535)
    sheet_template_id: str = Field(
        "1fvmDkDEBZPAdNO_SjzfVpBzqBWKeOjhcUWhSCmtdyjE"
    )

    # db
    db_schema: Optional[str] = Field(None, env="DB_SCHEMA")
    db_url: Optional[AnyUrl] = Field(None, env="DATABASE_URL")

    # llm
    llm_base_url: AnyUrl = Field("https://openrouter.ai/v1", env="LLM_BASE_URL")
    llm_model: str = Field("google/gemini-2.0-flash-001", env="LLM_MODEL")

    class Config:
        env_file = BASE_DIR / ".env"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
