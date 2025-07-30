from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class WebSearchAgentSettings(BaseSettings):
    model_name: str = Field(..., description="Название модели")
    temperature: int = Field(..., description="Температура модели")
    max_tokens: int = Field(..., description="Максимальное количество токенов")
    use_beam_search: bool = Field(..., description="Использование beam search")
    best_of: int = Field(..., description="Лучший из top k запросов")

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="WEBSEARCH_", case_sensitive=False, extra="ignore"
    )


# class TavilySearchSettings(BaseSettings):
#     max_results: int = Field(..., description="Количество результатов веб поиска")
#     topic: str = Field(..., description="Топик")

#     model_config = SettingsConfigDict(
#         env_file=".env", env_prefix="TAVILY_", case_sensitive=False, extra="ignore"
#     )
