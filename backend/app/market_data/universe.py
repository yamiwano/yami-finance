"""Liquid Binance USDT spot universe. No equities. No simulated names."""

from __future__ import annotations

from app.domain import AssetType

# Display names for well-known bases. Unknown bases use the ticker itself.
ASSET_NAMES: dict[str, str] = {
    "BTC": "Bitcoin",
    "ETH": "Ethereum",
    "SOL": "Solana",
    "XRP": "XRP",
    "DOGE": "Dogecoin",
    "ADA": "Cardano",
    "AVAX": "Avalanche",
    "LINK": "Chainlink",
    "DOT": "Polkadot",
    "ATOM": "Cosmos",
    "NEAR": "NEAR Protocol",
    "LTC": "Litecoin",
    "BCH": "Bitcoin Cash",
    "UNI": "Uniswap",
    "AAVE": "Aave",
    "SUI": "Sui",
    "APT": "Aptos",
    "ARB": "Arbitrum",
    "OP": "Optimism",
    "FIL": "Filecoin",
    "INJ": "Injective",
    "TIA": "Celestia",
    "SEI": "Sei",
    "PEPE": "Pepe",
    "WIF": "dogwifhat",
    "BONK": "Bonk",
    "SHIB": "Shiba Inu",
    "TRX": "TRON",
    "TON": "Toncoin",
    "HBAR": "Hedera",
    "XLM": "Stellar",
    "ETC": "Ethereum Classic",
    "ICP": "Internet Computer",
    "RENDER": "Render",
    "FET": "Fetch.ai",
    "TAO": "Bittensor",
    "ONDO": "Ondo",
    "ENA": "Ethena",
    "WLD": "Worldcoin",
    "JUP": "Jupiter",
    "PENGU": "Pudgy Penguins",
    "HYPE": "Hyperliquid",
}

# Quote-stable and fiat bases we never scan as the "asset".
STABLE_OR_FIAT_BASES = {
    "USDT",
    "USDC",
    "BUSD",
    "FDUSD",
    "TUSD",
    "DAI",
    "USDP",
    "USDD",
    "USD1",
    "USDE",
    "PYUSD",
    "UST",
    "AEUR",
    "EUR",
    "EURI",
    "TRY",
    "BRL",
    "ARS",
    "IDRT",
    "UAH",
    "NGN",
    "GBP",
    "AUD",
    "BIDR",
    "BVND",
}

LEVERAGED_SUFFIXES = ("UP", "DOWN", "BULL", "BEAR")

# Used only by MARKET_DATA_PROVIDER=mock (offline). Live tape never reads this list.
MOCK_CRYPTO = [
    {"symbol": "BTCUSDT", "name": "Bitcoin", "asset_type": AssetType.CRYPTO, "exchange": "BINANCE", "price": 67420.0},
    {"symbol": "ETHUSDT", "name": "Ethereum", "asset_type": AssetType.CRYPTO, "exchange": "BINANCE", "price": 3280.0},
    {"symbol": "SOLUSDT", "name": "Solana", "asset_type": AssetType.CRYPTO, "exchange": "BINANCE", "price": 148.5},
    {"symbol": "XRPUSDT", "name": "XRP", "asset_type": AssetType.CRYPTO, "exchange": "BINANCE", "price": 0.62},
    {"symbol": "DOGEUSDT", "name": "Dogecoin", "asset_type": AssetType.CRYPTO, "exchange": "BINANCE", "price": 0.148},
    {"symbol": "AVAXUSDT", "name": "Avalanche", "asset_type": AssetType.CRYPTO, "exchange": "BINANCE", "price": 28.4},
    {"symbol": "LINKUSDT", "name": "Chainlink", "asset_type": AssetType.CRYPTO, "exchange": "BINANCE", "price": 13.8},
    {"symbol": "ADAUSDT", "name": "Cardano", "asset_type": AssetType.CRYPTO, "exchange": "BINANCE", "price": 0.42},
    {"symbol": "ATOMUSDT", "name": "Cosmos", "asset_type": AssetType.CRYPTO, "exchange": "BINANCE", "price": 6.15},
    {"symbol": "NEARUSDT", "name": "NEAR Protocol", "asset_type": AssetType.CRYPTO, "exchange": "BINANCE", "price": 4.82},
]

REGIMES = {
    "BTCUSDT": "momentum",
    "ETHUSDT": "pullback",
    "SOLUSDT": "breakout",
    "XRPUSDT": "range",
    "DOGEUSDT": "parabolic",
    "AVAXUSDT": "failed_breakout",
    "LINKUSDT": "breakdown",
    "ADAUSDT": "range",
    "ATOMUSDT": "pullback",
    "NEARUSDT": "momentum",
}

# Back-compat aliases so older imports do not explode.
CRYPTO = MOCK_CRYPTO
UNIVERSE = MOCK_CRYPTO


def asset_name(base: str) -> str:
    return ASSET_NAMES.get(base.upper(), base.upper())


def is_leveraged_base(base: str) -> bool:
    u = base.upper()
    return any(u.endswith(sfx) for sfx in LEVERAGED_SUFFIXES)


def is_tradable_usdt_spot(info: dict) -> bool:
    if info.get("status") != "TRADING":
        return False
    if info.get("quoteAsset") != "USDT":
        return False
    if info.get("isSpotTradingAllowed") is False:
        return False
    base = str(info.get("baseAsset") or "")
    if base in STABLE_OR_FIAT_BASES:
        return False
    if is_leveraged_base(base):
        return False
    return True


def normalize_symbol(raw: str) -> str:
    s = (raw or "").strip().upper().replace("/", "").replace("-", "").replace("_", "")
    if s.endswith("USD") and not s.endswith("USDT") and not s.endswith("USDC"):
        s = s[:-3] + "USDT"
    return s


def resolve_symbol(raw: str, universe: set[str]) -> str | None:
    s = normalize_symbol(raw)
    if s in universe:
        return s
    if f"{s}USDT" in universe:
        return f"{s}USDT"
    if s.endswith("USD") and f"{s[:-3]}USDT" in universe:
        return f"{s[:-3]}USDT"
    return None
