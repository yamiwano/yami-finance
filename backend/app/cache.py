from app.redis_client import get_redis


async def cache_set(key: str, value: str, ttl: int = 30) -> None:
    try:
        r = await get_redis()
        await r.set(key, value, ex=ttl)
    except Exception:
        return


async def cache_get(key: str) -> str | None:
    try:
        r = await get_redis()
        return await r.get(key)
    except Exception:
        return None
