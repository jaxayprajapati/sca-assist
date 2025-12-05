from pydantic_settings import BaseSettings, SettingsConfigDict



class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file="app/config/.env",
        env_file_encoding="utf-8",
        frozen=True
    )

    DEBUG: bool
    OPENAI_API_KEY: str


settings = Settings()
