from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "sqlite+aiosqlite:///./radar.db"
    redis_url: str = "redis://localhost:6379/0"

    market_data_provider: str = "binance"
    scan_interval_seconds: float = 2.0
    full_rescan_seconds: float = 60.0
    min_opportunity_score: float = 55.0
    enable_crypto: bool = True
    enable_stocks: bool = False
    mock_time_acceleration: float = 40.0
    universe_size: int = 48
    min_quote_volume_usdt: float = 15_000_000
    binance_rest_base: str = "https://data-api.binance.vision"
    binance_ws_base: str = "wss://data-stream.binance.vision/stream"

    ai_provider: str = "mock"
    ai_model: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    ai_min_score: float = 65.0

    cors_origins: str = "http://localhost:3000"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
