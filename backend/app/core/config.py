from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    # Project
    PROJECT_NAME: str = "PaperAxiom"
    API_V1_STR: str = "/api/v1"

    # Database
    POSTGRES_USER: str = "paperaxiom"
    POSTGRES_PASSWORD: str = "paperaxiom_secret"
    POSTGRES_DB: str = "paperaxiom"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432

    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    # MinIO
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "paperaxiom"
    MINIO_SECRET_KEY: str = "paperaxiom_secret"
    MINIO_BUCKET: str = "papers"

    # Qdrant
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333

    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379

    # OpenRouter
    OPENROUTER_API_KEYS: str = ""
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_MODELS: str = "meta-llama/llama-3.1-8b-instruct"

    @property
    def api_keys_list(self) -> List[str]:
        if not self.OPENROUTER_API_KEYS:
            return []
        return [k.strip() for k in self.OPENROUTER_API_KEYS.split(",") if k.strip()]

    @property
    def models_list(self) -> List[str]:
        if not self.OPENROUTER_MODELS:
            return ["meta-llama/llama-3.1-8b-instruct"]
        return [m.strip() for m in self.OPENROUTER_MODELS.split(",") if m.strip()]

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()