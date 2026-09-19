from abc import ABC, abstractmethod
from datetime import datetime

from app.domain import AssetType, Bar, Quote, Timeframe


class MarketDataProvider(ABC):
    """Provider-agnostic market data interface.

    Swap implementations via MARKET_DATA_PROVIDER without changing the scanner.
    """

    @abstractmethod
    def universe(self) -> list[dict]:
        """Return [{symbol, name, asset_type, exchange}, ...]."""

    @abstractmethod
    async def historical(
        self,
        symbol: str,
        timeframe: Timeframe,
        limit: int = 300,
        *,
        closed_only: bool = False,
    ) -> list[Bar]:
        """OHLCV bars, oldest first. Forming bar included unless closed_only."""

    async def historical_range(
        self,
        symbol: str,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
    ) -> list[Bar]:
        """Closed OHLCV from start (inclusive) to end (exclusive). Default: RAM window."""
        bars = await self.historical(symbol, timeframe, 400, closed_only=True)
        return [b for b in bars if start <= b.ts < end]

    @abstractmethod
    async def quote(self, symbol: str) -> Quote:
        """Latest traded price snapshot."""

    @abstractmethod
    async def quotes(self) -> dict[str, Quote]:
        """Latest quotes for the full universe."""

    @abstractmethod
    async def tick(self) -> None:
        """Advance mock tape. No-op for live stream providers."""

    async def start(self) -> None:
        """Connect streams / seed history. Default no-op."""

    async def close(self) -> None:
        """Disconnect streams. Default no-op."""

    def drain_closed(self) -> list[tuple[str, Timeframe]]:
        """(symbol, timeframe) klines that closed since last drain."""
        return []

    def status(self) -> dict:
        return {"provider": type(self).__name__, "ready": True}

    def is_ready(self) -> bool:
        return True

    async def news_score(self, symbol: str, asset_type: AssetType) -> tuple[float, list[str]]:
        """Catalyst score 0-100. Default: unknown/neutral — never invent news."""
        return 50.0, [
            "No news/catalyst feed is wired. Catalyst component is scored as unknown (50), not as confirmation."
        ]
