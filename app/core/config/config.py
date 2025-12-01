from functools import lru_cache
from typing import Optional

from pydantic import Field, HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    api_key: str = Field(..., alias="API_KEY", description="Shared secret for X-API-Key header")
    model_device: str = Field("cuda", description="Device to run the image model on")
    webhook_timeout_seconds: int = Field(15, ge=1, le=120, description="Timeout for webhook callbacks")
    max_parallel_jobs: int = Field(4, ge=1, description="Upper bound for concurrent jobs")
    model_id: str = Field(
        default="Qwen/Qwen-Image-Edit-2509",
        description="HuggingFace model id for the local image editor",
    )
    default_negative_prompt: str = Field(
        default="low quality, artifacts, blur",
        description="Default negative prompt fed into the model",
    )
    webhook_verify_ssl: bool = Field(True, description="Toggle SSL verification for webhook requests")
    sample_webhook_url: Optional[HttpUrl] = Field(
        default=None, description="Optional webhook used for local testing"
    )

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[arg-type]
