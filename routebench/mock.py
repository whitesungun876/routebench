"""Simulated backend used ONLY to test the pipeline end to end without API keys.

Numbers produced in mock mode are synthetic and must never be reported as
results. Every mock report is stamped SIMULATED.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import random

from .client import Backend, Price
from .tasks import Task

# skill on a logit scale; price in USD per 1M tokens (in, out)
MOCK_MODELS = {
    "mock/tiny": dict(skill=1.2, lat=0.6, err=0.02, malformed=0.04, price=(0.05, 0.20)),
    "mock/small": dict(skill=2.2, lat=0.9, err=0.01, malformed=0.02, price=(0.15, 0.60)),
    "mock/medium": dict(skill=3.2, lat=1.6, err=0.01, malformed=0.01, price=(0.60, 2.40)),
    "mock/large": dict(skill=4.4, lat=3.5, err=0.03, malformed=0.0, price=(3.00, 15.00)),
}
_DIFF = {"easy": 0.0, "medium": 1.4, "hard": 2.9}
_TYPE = {"arith": 0.3, "date": 0.5, "extract": 0.2, "order": 0.4}


def mock_prices() -> dict[str, Price]:
    return {m: Price(*p["price"]) for m, p in MOCK_MODELS.items()}


def _seed(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:12], 16)


def _common_mistakes(task: Task) -> list[str]:
    """Plausible wrong answers shared across models, so errors are correlated."""
    rng = random.Random(_seed("mistake", task.id))
    if task.type == "arith":
        g = int(task.answer)
        return [str(g + rng.choice([-10, -1, 1, 10, 20])), str(g + rng.randint(-50, 50) or 7)]
    if task.type == "date":
        y, m, d = task.answer.split("-")
        dd = int(d)
        alt1 = f"{y}-{m}-{(dd % 28) + 1:02d}"
        alt2 = f"{y}-{m}-{((dd + 6) % 28) + 1:02d}"
        return [alt1, alt2]
    if task.type == "order":
        names = [w.strip(".") for w in task.prompt.split() if w.istitle() and w.strip(".").isalpha()]
        names = [n for n in dict.fromkeys(names) if n not in {"There", "Who", "Answer", "Think"}
                 and n != task.answer]
        return names[:2] or ["Nobody"]
    gold = json.loads(task.answer)
    wrong1 = dict(gold, total=round(gold["total"] * 1.25, 2))
    wrong2 = dict(gold, due_date="2000-01-01")
    return [json.dumps(wrong1), json.dumps(wrong2)]


class MockBackend(Backend):
    def __init__(self, tasks: list[Task], time_scale: float = 0.0):
        self.by_prompt = {t.prompt: t for t in tasks}
        self.time_scale = time_scale  # 0 = do not actually sleep
        self.calls = 0

    async def post(self, model, messages, params, timeout):
        self.calls += 1
        prof = MOCK_MODELS[model]
        task = self.by_prompt[messages[-1]["content"]]
        rng = random.Random(_seed(model, task.id, self.calls))  # per attempt (transient errors)
        stable = random.Random(_seed(model, task.id))             # per (model, task)
        latency = prof["lat"] * math.exp(stable.gauss(0, 0.35)) * (1 + 0.5 * _DIFF[task.difficulty])
        if self.time_scale:
            await asyncio.sleep(latency * self.time_scale)
        if rng.random() < prof["err"]:
            return (rng.choice([429, 503]), None, "http_simulated", None)
        logit = prof["skill"] - _DIFF[task.difficulty] - _TYPE[task.type]
        correct = stable.random() < 1 / (1 + math.exp(-logit))
        if stable.random() < prof["malformed"]:
            text = "I think the answer is probably around there."
        elif correct:
            text = f"Working...\nANSWER: {task.answer}"
        else:
            text = f"Working...\nANSWER: {stable.choice(_common_mistakes(task))}"
        tin = len(messages[-1]["content"]) // 4
        tout = int(60 + 220 * (1 + _DIFF[task.difficulty]) * (0.5 + stable.random()))
        body = {"choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": tin, "completion_tokens": tout},
                "_mock_latency": latency}
        return 200, body, None, None
