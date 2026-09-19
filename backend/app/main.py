import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.database import init_db
from app.redis_client import close_redis, get_redis
from app.api import router
from app.scanner import scanner
from app.ws import hub

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


@asynccontextmanager
async def lifespan(_: FastAPI):
    await init_db()
    try:
        redis = await get_redis()
        await redis.ping()
    except Exception:
        logging.getLogger("radar").warning("Redis unavailable — continuing without cache")
    await scanner.start()
    yield
    await scanner.stop()
    await close_redis()


app = FastAPI(title="Yami Financier", version="0.1.0", lifespan=lifespan)
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list or ["*"],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router, prefix="/api")


@app.get("/")
async def root():
    return {"name": "Yami Financier", "docs": "/docs"}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await hub.connect(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        hub.disconnect(ws)
