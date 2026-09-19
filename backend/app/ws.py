from __future__ import annotations

import json
from typing import Any

from fastapi import WebSocket


class Hub:
    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._clients.add(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self._clients.discard(ws)

    async def broadcast(self, event: str, payload: Any) -> None:
        message = json.dumps({"event": event, "payload": payload}, default=str)
        stale: list[WebSocket] = []
        for ws in list(self._clients):
            try:
                await ws.send_text(message)
            except Exception:
                stale.append(ws)
        for ws in stale:
            self.disconnect(ws)


hub = Hub()
