"""Async client for any OpenAI-compatible chat endpoint (OpenRouter, NexLLM, ...).

Reliability features:
  * per-call timeout
  * retries with exponential backoff and jitter on timeouts, connection errors,
    HTTP 429 and 5xx (honours Retry-After); other 4xx errors fail fast
  * ordered fallback chain across models (Gateway.complete)
  * SQLite response cache keyed on (model, messages, params), so reruns are
    free and reproducible
  * token and USD cost accounting with a hard budget stop
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import random
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529}


@dataclass
class Price:
    usd_per_m_in: float = 0.0
    usd_per_m_out: float = 0.0


@dataclass
class CallResult:
    model: str
    ok: bool
    text: str | None = None
    latency_s: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    attempts: int = 0
    error: str | None = None
    finish_reason: str | None = None
    cached: bool = False
    served_by: str | None = None  # model that actually answered (after fallback)
    fallback_used: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class BudgetExceeded(RuntimeError):
    pass


class Cache:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.execute("CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, v TEXT)")
        self.db.commit()

    @staticmethod
    def key(model: str, messages: list[dict], params: dict) -> str:
        blob = json.dumps({"m": model, "msgs": messages, "p": params}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def get(self, k: str) -> dict | None:
        row = self.db.execute("SELECT v FROM cache WHERE k=?", (k,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, k: str, v: dict) -> None:
        self.db.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)", (k, json.dumps(v)))
        self.db.commit()


@dataclass
class ClientConfig:
    base_url: str
    api_key: str
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout_s: float = 90.0
    max_retries: int = 3
    backoff_base_s: float = 1.0
    concurrency: int = 8
    budget_usd: float = 5.0
    extra_headers: dict = field(default_factory=dict)
    extra_body: dict = field(default_factory=dict)  # e.g. {"reasoning": {"enabled": false}}


class Backend:
    """One HTTP round trip.

    Returns (status | None, json_body | None, error_str | None, retry_after | None).
    status None means the request never got an HTTP response (timeout, network).
    """

    async def post(self, model: str, messages: list[dict], params: dict, timeout: float):
        raise NotImplementedError


class HttpBackend(Backend):
    def __init__(self, cfg: ClientConfig):
        self.cfg = cfg
        self.http = httpx.AsyncClient(
            base_url=cfg.base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {cfg.api_key}", **cfg.extra_headers},
        )

    async def post(self, model, messages, params, timeout):
        body = {"model": model, "messages": messages, **params}
        try:
            r = await self.http.post("/chat/completions", json=body, timeout=timeout)
        except httpx.TimeoutException:
            return None, None, "timeout", None
        except httpx.TransportError as e:
            return None, None, f"transport:{type(e).__name__}", None
        retry_after = r.headers.get("retry-after")
        try:
            data = r.json()
        except ValueError:
            data = None
        if r.status_code != 200:
            msg = (data or {}).get("error", {}) if isinstance(data, dict) else {}
            msg = msg.get("message") if isinstance(msg, dict) else str(msg)
            return r.status_code, data, f"http_{r.status_code}:{(msg or '')[:120]}", retry_after
        # some gateways return 200 with an error object
        if isinstance(data, dict) and data.get("error"):
            return 502, data, f"provider_error:{str(data['error'])[:120]}", None
        return 200, data, None, None

    async def aclose(self):
        await self.http.aclose()


class Gateway:
    def __init__(self, cfg: ClientConfig, backend: Backend, prices: dict[str, Price],
                 cache: Cache | None = None, sleep=asyncio.sleep):
        self.cfg = cfg
        self.backend = backend
        self.prices = prices
        self.cache = cache
        self.spent_usd = 0.0
        self._sleep = sleep
        self._sem = asyncio.Semaphore(cfg.concurrency)
        self._rng = random.Random(0)

    def _params(self) -> dict:
        return {"temperature": self.cfg.temperature, "max_tokens": self.cfg.max_tokens,
                **self.cfg.extra_body}

    def _cost(self, model: str, data: dict, tin: int, tout: int) -> float:
        reported = (data.get("usage") or {}).get("cost")
        if isinstance(reported, (int, float)):
            return float(reported)
        p = self.prices.get(model, Price())
        return tin / 1e6 * p.usd_per_m_in + tout / 1e6 * p.usd_per_m_out

    async def call(self, model: str, messages: list[dict]) -> CallResult:
        """Single model, with cache, retries and budget accounting."""
        params = self._params()
        k = Cache.key(model, messages, params) if self.cache else None
        if k and (hit := self.cache.get(k)):
            res = CallResult(**hit)
            res.cached = True
            return res
        if self.spent_usd >= self.cfg.budget_usd:
            raise BudgetExceeded(f"budget {self.cfg.budget_usd} USD reached")

        res = CallResult(model=model, ok=False, served_by=model)
        t_first = time.perf_counter()
        async with self._sem:
            for attempt in range(self.cfg.max_retries + 1):
                res.attempts = attempt + 1
                t0 = time.perf_counter()
                status, data, err, retry_after = await self.backend.post(
                    model, messages, params, self.cfg.timeout_s)
                if status == 200 and data:
                    choice = (data.get("choices") or [{}])[0]
                    msg = choice.get("message") or {}
                    usage = data.get("usage") or {}
                    res.text = msg.get("content") or ""
                    res.finish_reason = choice.get("finish_reason")
                    res.tokens_in = int(usage.get("prompt_tokens") or 0)
                    res.tokens_out = int(usage.get("completion_tokens") or 0)
                    res.cost_usd = self._cost(model, data, res.tokens_in, res.tokens_out)
                    # latency of the successful attempt (mock backend reports a simulated value)
                    res.latency_s = data.get("_mock_latency", time.perf_counter() - t0)
                    res.ok = True
                    res.error = None
                    break
                res.error = err
                retryable = status is None or status in RETRYABLE_STATUS
                if not retryable or attempt == self.cfg.max_retries:
                    break
                delay = self.cfg.backoff_base_s * (2 ** attempt) * (0.5 + self._rng.random())
                if retry_after:
                    try:
                        delay = max(delay, float(retry_after))
                    except ValueError:
                        pass
                await self._sleep(min(delay, 60))
        if not res.ok:
            res.latency_s = time.perf_counter() - t_first
        self.spent_usd += res.cost_usd
        if res.ok and k:
            self.cache.put(k, res.to_dict())
        return res

    async def complete(self, chain: list[str], messages: list[dict]) -> CallResult:
        """Try models in order; the first successful call wins."""
        last = None
        for i, model in enumerate(chain):
            res = await self.call(model, messages)
            if res.ok:
                res.fallback_used = i > 0
                res.served_by = model
                return res
            last = res
        return last


async def fetch_prices(base_url: str, api_key: str) -> dict[str, Price]:
    """OpenRouter-style /models listing with per-token pricing. Empty dict if unavailable."""
    try:
        async with httpx.AsyncClient() as http:
            r = await http.get(base_url.rstrip("/") + "/models",
                               headers={"Authorization": f"Bearer {api_key}"}, timeout=30)
            r.raise_for_status()
            out = {}
            for m in r.json().get("data", []):
                p = m.get("pricing") or {}
                try:
                    out[m["id"]] = Price(float(p.get("prompt", 0)) * 1e6,
                                         float(p.get("completion", 0)) * 1e6)
                except (TypeError, ValueError):
                    out[m["id"]] = Price()
            return out
    except Exception:
        return {}
