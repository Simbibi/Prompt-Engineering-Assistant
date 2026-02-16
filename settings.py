from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import AnyUrl, Field, SecretStr
from pydantic_settings import BaseSettings

# Base dir for locating the .env file 
BASE_DIR = Path(__file__).parent


class Settings(BaseSettings):
    # secrets
    openai_api_key: Optional[SecretStr] = Field(None, env="OPENAI_API_KEY")
    openai_api_url: Optional[SecretStr] = Field(None, env="OPENAI_API_URL")

    # app
    port: int = Field(8000, ge=1, le=65535)
    sheet_template_id: str = Field(
        "1fvmDkDEBZPAdNO_SjzfVpBzqBWKeOjhcUWhSCmtdyjE"
    )

    # db
    chroma_db_dir: Path = Field("chroma_db", env="CHROMA_DB_DIR")

    class Config:
        env_file = BASE_DIR / ".env"
        extra = "ignore"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
