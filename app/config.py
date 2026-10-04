from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Model Config
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

    # LLM / LiteLLM
    LLM_BASE_URL: str
    LLM_API_KEY: str
    GENERATION_MODEL: str = "llama3-70b-8192"
    STRUCTURED_OUTPUT_MODEL: str = "llama3-70b-8192"

    # Qdrant
    QDRANT_URL: str
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_COLLECTION: str
    QDRANT_PORT: Optional[int] = 6333

    # PostgreSQL — required when PostgreSQL integration is active
    # Not yet consumed by application code; Optional until implemented.
    POSTGRES_HOST: Optional[str] = None
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: Optional[str] = None
    POSTGRES_USER: Optional[str] = None
    POSTGRES_PASSWORD: Optional[str] = None

    # Redis — required when Redis integration is active
    # Not yet consumed by application code; Optional until implemented.
    REDIS_HOST: Optional[str] = None
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: Optional[str] = None

    # ZeroClaw
    ZEROCLAW_WEBHOOK_SECRET: str
    ZEROCLAW_WEBHOOK_PATH: str = "/webhook/zeroclaw"
    ZEROCLAW_WEBHOOK_HOST: str = "0.0.0.0"
    ZEROCLAW_WEBHOOK_PORT: int = 8001
    ZEROCLAW_API_URL: Optional[str] = None
    ZEROCLAW_API_KEY: Optional[str] = None


    # JWT / Auth
    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 60

    # Ticketing
    TICKETING_API_URL: Optional[str] = None
    TICKETING_API_KEY: Optional[str] = None

    # Application settings
    ENVIRONMENT: str = "development"

    # Embeddings
    EMBEDDING_DEVICE: str = "auto"
    EMBEDDING_BATCH_SIZE: int = 32
    EMBEDDING_USE_COLBERT: bool = False


# Instantiate settings instance
settings = Settings()
