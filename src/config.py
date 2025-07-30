from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from src.settings import WebSearchAgentSettings


class Config(BaseSettings):
    url: str = Field(..., description='Базовая ссылка на api')
    api_key: SecretStr = Field(..., description="Секретный ключ")
    websearch: WebSearchAgentSettings = Field(default_factory=WebSearchAgentSettings)
    # tavily: TavilySearchSettings = Field(default_factory=TavilySearchSettings)


    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="BASE_",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
