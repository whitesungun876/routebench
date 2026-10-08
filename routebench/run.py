"""Collect one response per (task, model). Resumable: finished rows are skipped."""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import yaml

from .client import (BudgetExceeded, Cache, ClientConfig, Gateway, HttpBackend, Price,
                     fetch_prices)
from .mock import MOCK_MODELS, MockBackend, mock_prices
from .tasks import generate, grade

EST_OUT_TOKENS = 600  # rough per-call output estimate for the pre-run cost check


def load_config(path: str, mock: bool) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    if mock:
        cfg["models"] = [{"id": m} for m in MOCK_MODELS]
    return cfg


def _messages(prompt: str) -> list[dict]:
    return [{"role": "user", "content": prompt}]


async def _resolve_prices(cfg: dict, mock: bool) -> dict[str, Price]:
    if mock:
        return mock_prices()
    prices: dict[str, Price] = {}
    listed = await fetch_prices(cfg["base_url"], os.environ.get(cfg["api_key_env"], ""))
    missing = []
    for m in cfg["models"]:
        if "price_in" in m:
            prices[m["id"]] = Price(float(m["price_in"]), float(m["price_out"]))
        elif m["id"] in listed:
            prices[m["id"]] = listed[m["id"]]
        else:
            missing.append(m["id"])
    if listed and missing:
        sys.exit(f"Model id(s) not found on {cfg['base_url']}: {missing}. "
                 "Run `python -m routebench models` to see valid ids.")
    if missing:
        print(f"[warn] no price for {missing}; cost will rely on usage.cost from the API if present")
    return prices


def estimate_cost(tasks, models, prices) -> dict[str, float]:
    avg_in = sum(len(t.prompt) for t in tasks) / len(tasks) / 4
    out = {}
    for m in models:
        p = prices.get(m, Price())
        out[m] = len(tasks) * (avg_in * p.usd_per_m_in + EST_OUT_TOKENS * p.usd_per_m_out) / 1e6
    return out


async def collect(config_path: str, out_dir: str, mock: bool = False, yes: bool = False,
                  n_per_type: int | None = None) -> Path:
    cfg = load_config(config_path, mock)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tasks = generate(n_per_type=n_per_type or cfg.get("n_per_type", 100), seed=cfg.get("seed", 20261008))
    with open(out / "tasks.jsonl", "w") as f:
        for t in tasks:
            f.write(json.dumps(t.to_dict(), ensure_ascii=False) + "\n")
    models = [m["id"] for m in cfg["models"]]
    prices = await _resolve_prices(cfg, mock)

    est = estimate_cost(tasks, models, prices)
    print(f"{len(tasks)} tasks x {len(models)} models = {len(tasks) * len(models)} calls")
    for m, c in est.items():
        print(f"  {m:45s} est. ${c:.2f}")
    total_est = sum(est.values())
    budget = float("inf") if mock else float(cfg.get("budget_usd", 5))
    print(f"  estimated total ${total_est:.2f} (budget ${budget:.2f}; estimate assumes "
          f"~{EST_OUT_TOKENS} output tokens per call, reasoning models can use far more)")
    if not mock and not yes:
        if input("Proceed? [y/N] ").strip().lower() != "y":
            sys.exit("aborted")

    ccfg = ClientConfig(
        base_url=cfg.get("base_url", "mock://"),
        api_key=os.environ.get(cfg.get("api_key_env", ""), "mock"),
        temperature=cfg.get("temperature", 0.0), max_tokens=cfg.get("max_tokens", 2048),
        timeout_s=cfg.get("timeout_s", 90), max_retries=cfg.get("max_retries", 3),
        concurrency=cfg.get("concurrency", 8), budget_usd=budget,
        extra_headers=cfg.get("extra_headers", {}) or {},
        extra_body=cfg.get("extra_body", {}) or {},
    )
    if not mock and not ccfg.api_key:
        sys.exit(f"Set the {cfg['api_key_env']} environment variable first.")
    backend = MockBackend(tasks) if mock else HttpBackend(ccfg)
    gw = Gateway(ccfg, backend, prices, cache=Cache(out / "cache.sqlite"),
                 sleep=(lambda s: asyncio.sleep(0)) if mock else asyncio.sleep)

    results_path = out / "results.jsonl"
    done = set()
    if results_path.exists():
        for line in results_path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["task_id"], r["model"]))
    todo = [(t, m) for t in tasks for m in models if (t.id, m) not in done]
    print(f"{len(done)} rows already collected, {len(todo)} to go")

    lock = asyncio.Lock()
    progress = {"n": 0, "t0": time.time(), "stopped": False}
    fh = open(results_path, "a")

    async def one(task, model):
        if progress["stopped"]:
            return
        try:
            res = await gw.call(model, _messages(task.prompt))
        except BudgetExceeded as e:
            if not progress["stopped"]:
                print(f"\n[stop] {e}; partial results kept, rerun with a higher budget to resume")
            progress["stopped"] = True
            return
        correct, well_formed = grade(task.type, task.answer, res.text) if res.ok else (False, False)
        row = {"task_id": task.id, "type": task.type, "difficulty": task.difficulty,
               "split": task.split, **res.to_dict(), "correct": correct, "well_formed": well_formed}
        async with lock:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            progress["n"] += 1
            if progress["n"] % 50 == 0 or progress["n"] == len(todo):
                print(f"  {progress['n']}/{len(todo)}  spent ${gw.spent_usd:.3f}  "
                      f"{time.time() - progress['t0']:.0f}s", flush=True)

    await asyncio.gather(*(one(t, m) for t, m in todo))
    fh.close()
    if isinstance(backend, HttpBackend):
        await backend.aclose()
    (out / "run_meta.json").write_text(json.dumps({
        "mock": mock, "models": models, "base_url": cfg.get("base_url"),
        "temperature": ccfg.temperature, "max_tokens": ccfg.max_tokens,
        "n_tasks": len(tasks), "seed": cfg.get("seed", 20261008),
        "prices_usd_per_m": {m: [prices.get(m, Price()).usd_per_m_in,
                                 prices.get(m, Price()).usd_per_m_out] for m in models},
        "spent_usd_this_session": gw.spent_usd,
        "completed": not progress["stopped"],
    }, indent=2))
    return out


async def list_models(config_path: str, contains: str | None):
    cfg = yaml.safe_load(Path(config_path).read_text())
    prices = await fetch_prices(cfg["base_url"], os.environ.get(cfg["api_key_env"], ""))
    if not prices:
        sys.exit("Could not list models (check base_url / key / network).")
    rows = sorted(prices.items(), key=lambda kv: kv[1].usd_per_m_out)
    for mid, p in rows:
        if contains and contains.lower() not in mid.lower():
            continue
        print(f"{mid:60s} in ${p.usd_per_m_in:7.3f}/M  out ${p.usd_per_m_out:7.3f}/M")
