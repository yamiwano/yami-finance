# Yami Financier

Decision-support scanner for **liquid Binance USDT spot** pairs. It ranks LONG / SHORT / WATCH setups from deterministic technical rules, stores a snapshot of the tape at detection, and tracks outcomes. It does **not** place trades and does **not** claim to predict markets.

Prices, candles, 24h volume, and bid/ask spreads come from **Binance spot** (REST seed + WebSocket). There is no simulated tape on the default path. Setups are evaluated on **closed candles** only.

## Desktop app

Install a launcher that appears in the app menu. Closing the window stops the UI and the scanner.

```bash
chmod +x scripts/yami-financier scripts/install-desktop.sh
./scripts/install-desktop.sh
```

Then open **Yami Financier** from the application menu, or run:

```bash
./scripts/yami-financier
```

## Run in a browser

**Docker (Postgres + Redis + both apps):**

```bash
cp .env.example .env
docker compose up --build
```

- UI: http://localhost:3000
- API: http://localhost:8000/docs
- WebSocket: ws://localhost:8000/ws

**Without Docker:** the API falls back to SQLite if `DATABASE_URL` is unset. Redis is optional.

```bash
cd backend && python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# separate terminal
cd frontend && npm install && npm run dev
```

Default market data is **Binance USDT spot** via the public market-data endpoints (`data-api.binance.vision` REST + `data-stream.binance.vision` WebSocket). No API key is required. The scanner fails over from `api.binance.com` when that host is geo-restricted (HTTP 451). It does **not** mix in Binance.US prices.

`MARKET_DATA_PROVIDER=mock` exists only for offline UI work. Do not use it for live research.

### Optional AI notes

Set `AI_PROVIDER=openai` or `anthropic` plus the matching key. The model only restates scanner facts; it cannot change scores or invent prices. `AI_PROVIDER=mock` (default) templates the same fields.

## Pipeline

Binance REST seed → WebSocket klines/bookTicker → indicators → six strategies (closed candles) → risk flags → transparent 0–100 score → Postgres snapshot → optional AI → WebSocket UI.

Universe: top USDT spot pairs by 24h quote volume (stables and leveraged tokens excluded). Most symbols carry **no signal**. Levels (entry / stop / targets) are hypothetical zones, not guarantees. The catalyst component is scored as unknown unless a real news feed is wired.

## Strategies

**Long:** volume-confirmed breakout, trend pullback, momentum continuation  
**Short:** failed breakout, parabolic exhaustion, breakdown continuation  

Timeframes: 5m, 15m, 1h.
