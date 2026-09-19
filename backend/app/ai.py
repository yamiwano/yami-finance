"""Optional AI layer. Never computes indicators, prices, or scores — only restates facts."""

from __future__ import annotations

from typing import Any, Protocol

import httpx

from app.config import get_settings

SYSTEM = """You are a market analyst for Yami Financier, a decision-support tool (not a broker, not advice).
You receive ONLY structured facts already computed by a deterministic scanner.
Rules:
- Do not invent prices, indicators, news, timestamps, or statistics.
- Do not change or comment as if you computed the opportunity score.
- If a fact is missing, say it is not in the payload.
- Levels are hypothetical zones, not guarantees.
Return JSON with keys: thesis, evidence (array of strings), bear_case, what_to_watch, summary.
Keep thesis ≤ 3 sentences. Summary is plain English for a terminal footer.
"""


class Explainer(Protocol):
    async def explain(self, facts: dict[str, Any]) -> dict[str, Any]: ...


class MockExplainer:
    async def explain(self, facts: dict[str, Any]) -> dict[str, Any]:
        direction = facts.get("direction")
        strategy = facts.get("strategy_label") or facts.get("strategy")
        symbol = facts.get("symbol")
        reasons = facts.get("reasons") or []
        flags = facts.get("risk_flags") or []
        score = facts.get("score")
        tf = facts.get("timeframe")
        evidence = [str(r) for r in reasons[:6]]
        if not evidence:
            evidence = ["No rule-level evidence was supplied in the payload."]
        bear = "Invalidation / stop in the payload is the primary failure mode."
        if flags:
            bear = f"Primary risks disclosed by the scanner: {', '.join(f.get('code','?') for f in flags)}."
        watch = (
            f"Watch whether price holds the entry zone {facts.get('entry_low')}–{facts.get('entry_high')} "
            f"and whether {tf} structure remains consistent with a {direction} {strategy}."
        )
        thesis = (
            f"{symbol} screened as a {direction} {strategy} on {tf} with a model score of {score}/100. "
            f"The scanner (not this explanation) computed that score from stored components. "
            f"This is a setup description, not a prediction or a trade recommendation."
        )
        summary = (
            f"{symbol}: {direction} {strategy} ({tf}). Score {score}. "
            f"Entry {facts.get('entry_low')}–{facts.get('entry_high')}, stop {facts.get('stop')}, "
            f"T1 {facts.get('target_1')}. Not a guarantee."
        )
        return {
            "thesis": thesis,
            "evidence": evidence,
            "bear_case": bear,
            "what_to_watch": watch,
            "summary": summary,
            "provider": "mock",
        }


class OpenAIExplainer:
    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model or "gpt-4o-mini"

    async def explain(self, facts: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Facts JSON:\n{facts}"},
            ],
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
            )
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"]
        import json

        data = json.loads(text)
        data["provider"] = "openai"
        return data


class AnthropicExplainer:
    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model or "claude-sonnet-4-20250514"

    async def explain(self, facts: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": self.model,
                    "max_tokens": 800,
                    "system": SYSTEM,
                    "messages": [{"role": "user", "content": f"Facts JSON:\n{facts}"}],
                },
            )
            r.raise_for_status()
            text = r.json()["content"][0]["text"]
        import json
        import re

        match = re.search(r"\{.*\}", text, re.S)
        data = json.loads(match.group(0) if match else text)
        data["provider"] = "anthropic"
        return data


def create_explainer() -> Explainer:
    s = get_settings()
    name = s.ai_provider.lower().strip()
    if name == "openai" and s.openai_api_key:
        return OpenAIExplainer(s.openai_api_key, s.ai_model)
    if name == "anthropic" and s.anthropic_api_key:
        return AnthropicExplainer(s.anthropic_api_key, s.ai_model)
    return MockExplainer()
