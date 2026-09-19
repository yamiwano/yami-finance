from app.market_data.base import MarketDataProvider


def create_provider() -> MarketDataProvider:
    from app.config import get_settings

    settings = get_settings()
    name = settings.market_data_provider.lower().strip()
    if name == "mock":
        from app.market_data.mock import MockProvider

        return MockProvider(acceleration=settings.mock_time_acceleration)
    from app.market_data.binance import BinanceProvider

    return BinanceProvider()
