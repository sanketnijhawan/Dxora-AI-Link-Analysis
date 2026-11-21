"""Configuration is supplied by environment variables, never source code."""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.1-flash-lite"
    request_timeout: float = 35.0


def get_settings() -> Settings:
    return Settings(
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite"),
        request_timeout=float(os.getenv("LLM_TIMEOUT_SECONDS", "35")),
    )
