from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    # ============================================================
    # PROJECT
    # ============================================================

    PROJECT_NAME: str = "PaperAxiom"
    API_V1_STR: str = "/api/v1"


    # ============================================================
    # DATABASE
    # ============================================================
    #
    # SQLite is used for the Render Free deployment.
    # This avoids requiring a paid PostgreSQL service.
    #

    DATABASE_URL: str = "sqlite:///./paperaxiom.db"


    # ============================================================
    # MINIO
    # ============================================================

    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "paperaxiom"
    MINIO_SECRET_KEY: str = "paperaxiom_secret"
    MINIO_BUCKET: str = "papers"


    # ============================================================
    # QDRANT
    # ============================================================

    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333


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
    OPENROUTER_MODELS: str = "meta-llama/llama-3.1-8b-instruct"


    # ============================================================
    # HELPERS
    # ============================================================

    @property
    def api_keys_list(self) -> List[str]:
        if not self.OPENROUTER_API_KEYS:
            return []

        return [
            key.strip()
            for key in self.OPENROUTER_API_KEYS.split(",")
            if key.strip()
        ]


    @property
    def models_list(self) -> List[str]:
        if not self.OPENROUTER_MODELS:
            return [
                "meta-llama/llama-3.1-8b-instruct"
            ]

        return [
            model.strip()
            for model in self.OPENROUTER_MODELS.split(",")
            if model.strip()
        ]


    # ============================================================
    # ENVIRONMENT CONFIGURATION
    # ============================================================

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()
