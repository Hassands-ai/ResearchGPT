from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    PROJECT_NAME: str = "ResearchGPT"
    API_V1_STR: str = "/api/v1"

    # ============================================================
    # DATABASE
    # ============================================================

    DATABASE_URL: str = ""

    POSTGRES_USER: str = "paperaxiom"
    POSTGRES_PASSWORD: str = ""
    POSTGRES_DB: str = "paperaxiom"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432

    @property
    def database_url_resolved(self) -> str:
        if self.DATABASE_URL:
            return self.DATABASE_URL

        return (
            f"postgresql://{self.POSTGRES_USER}:"
            f"{self.POSTGRES_PASSWORD}@"
            f"{self.POSTGRES_HOST}:"
            f"{self.POSTGRES_PORT}/"
            f"{self.POSTGRES_DB}"
        )

    # ============================================================
    # QDRANT
    # ============================================================

    QDRANT_URL: str = ""
    QDRANT_API_KEY: str = ""
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333

    # ============================================================
    # MINIO
    # ============================================================

    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "paperaxiom"
    MINIO_SECRET_KEY: str = ""
    MINIO_BUCKET: str = "papers"

    # ============================================================
    # REDIS
    # ============================================================

    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379

    # ============================================================
    # OPENROUTER
    # ============================================================

    OPENROUTER_API_KEYS: str = ""
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_MODELS: str = "qwen/qwen3-14b:free,openrouter/free"

    @property
    def api_keys_list(self) -> List[str]:
        return [
            key.strip()
            for key in self.OPENROUTER_API_KEYS.split(",")
            if key.strip()
        ]

    @property
    def models_list(self) -> List[str]:
        return [
            model.strip()
            for model in self.OPENROUTER_MODELS.split(",")
            if model.strip()
        ] or ["openrouter/free"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()
