"""Offline evaluation of routing strategies on recorded responses.

Protocol
  * Every model answered every task once (temperature 0). Strategies are replayed
    offline from those recorded responses, so all strategies see identical outputs.
  * Strategy parameters are chosen on the DEV split only. The TEST split is
    evaluated once with the frozen choices.
  * A failed call (after retries) counts as a wrong answer for that model.
  * Routers only see the prompt text: detected task family + a length bucket.
    The generator's difficulty label is never used for routing.
"""
from __future__ import annotations

import itertools
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .tasks import TASK_TYPES, canonical

BOOT = 2000
TARGET_GAP = 0.01   # a strategy "matches" the best model if dev accuracy >= best - 1pp
EPSILONS = (0.0, 0.02, 0.05, 0.10)


# ------------------------------------------------------------------ loading
@dataclass
class Data:
    tasks: dict            # task_id -> task dict
    models: list[str]
    R: dict                # (task_id, model) -> row
    mock: bool
    prices: dict = field(default_factory=dict)

    def ids(self, split: str) -> list[str]:
        return [t for t, d in self.tasks.items() if d["split"] == split]


def load(run_dir: str | Path) -> Data:
    run = Path(run_dir)
    tasks = {}
    for line in (run / "tasks.jsonl").read_text().splitlines():
        t = json.loads(line)
        tasks[t["id"]] = t
    meta = json.loads((run / "run_meta.json").read_text())
    R = {}
    for line in (run / "results.jsonl").read_text().splitlines():
        r = json.loads(line)
        R[(r["task_id"], r["model"])] = r
    models = meta["models"]
    complete = [t for t in tasks if all((t, m) in R for m in models)]
    tasks = {t: tasks[t] for t in complete}
    return Data(tasks, models, R, meta.get("mock", False), meta.get("prices_usd_per_m", {}))


# ----------------------------------------------------------------- features
_TYPE_PATTERNS = [
    ("extract", re.compile(r"invoice", re.I)),
    ("date", re.compile(r"\d{4}-\d{2}-\d{2}.*milestone|milestone.*\d{4}-\d{2}-\d{2}", re.I | re.S)),
    ("order", re.compile(r"taller|shorter", re.I)),
    ("arith", re.compile(r"how many", re.I)),
]


def detect_type(prompt: str) -> str:
    for name, pat in _TYPE_PATTERNS:
        if pat.search(prompt):
            return name
    return "other"


def length_buckets(data: Data, dev_ids: list[str]) -> dict[str, tuple[float, float]]:
    """Per detected type, tercile cut points of prompt length, fitted on dev."""
    cuts = {}
    for ty in list(TASK_TYPES) + ["other"]:
        lens = [len(data.tasks[t]["prompt"]) for t in dev_ids if detect_type(data.tasks[t]["prompt"]) == ty]
        if lens:
            cuts[ty] = tuple(np.quantile(lens, [1 / 3, 2 / 3]))
    return cuts


def cell(prompt: str, cuts) -> tuple[str, int]:
    ty = detect_type(prompt)
    lo, hi = cuts.get(ty, (float("inf"), float("inf")))
    n = len(prompt)
    return ty, 0 if n <= lo else (1 if n <= hi else 2)


# --------------------------------------------------------------- strategies
@dataclass
class Outcome:
    correct: np.ndarray
    cost: np.ndarray
    latency: np.ndarray
    calls: np.ndarray


def _row(data, t, m):
    return data.R[(t, m)]


def single(data: Data, ids, model) -> Outcome:
    rows = [_row(data, t, model) for t in ids]
    return Outcome(np.array([r["correct"] for r in rows], float),
                   np.array([r["cost_usd"] for r in rows]),
                   np.array([r["latency_s"] for r in rows]),
                   np.ones(len(ids)))


def oracle(data: Data, ids, by_price) -> Outcome:
    c, cost, lat = [], [], []
    for t in ids:
        pick = next((m for m in by_price if _row(data, t, m)["correct"]), by_price[0])
        r = _row(data, t, pick)
        c.append(r["correct"]); cost.append(r["cost_usd"]); lat.append(r["latency_s"])
    return Outcome(np.array(c, float), np.array(cost), np.array(lat), np.ones(len(ids)))


def fit_router(data: Data, dev_ids, eps: float, by_price, cuts) -> dict:
    """Per cell: cheapest model whose dev accuracy is within eps of the best in that cell."""
    cells: dict = {}
    for t in dev_ids:
        cells.setdefault(cell(data.tasks[t]["prompt"], cuts), []).append(t)
    table = {}
    for key, tids in cells.items():
        acc = {m: np.mean([_row(data, t, m)["correct"] for t in tids]) for m in by_price}
        best = max(acc.values())
        table[key] = next(m for m in by_price if acc[m] >= best - eps - 1e-12)
    return table


def apply_router(data: Data, ids, table, fallback_model, cuts) -> tuple[Outcome, list[str]]:
    picks = [table.get(cell(data.tasks[t]["prompt"], cuts), fallback_model) for t in ids]
    rows = [_row(data, t, m) for t, m in zip(ids, picks)]
    return Outcome(np.array([r["correct"] for r in rows], float),
                   np.array([r["cost_usd"] for r in rows]),
                   np.array([r["latency_s"] for r in rows]),
                   np.ones(len(ids))), picks


def agreement_cascade(data: Data, ids, a, b, strong) -> tuple[Outcome, float]:
    """Run cheap models a and b in parallel; accept if their canonical answers agree,
    otherwise escalate to the strong model."""
    c, cost, lat, calls, esc = [], [], [], [], 0
    for t in ids:
        ty = data.tasks[t]["type"]
        ra, rb = _row(data, t, a), _row(data, t, b)
        ca = canonical(ty, ra["text"]) if ra["ok"] else None
        cb = canonical(ty, rb["text"]) if rb["ok"] else None
        base_cost = ra["cost_usd"] + rb["cost_usd"]
        base_lat = max(ra["latency_s"], rb["latency_s"])
        if ca is not None and ca == cb:
            c.append(ra["correct"]); cost.append(base_cost); lat.append(base_lat); calls.append(2)
        else:
            rs = _row(data, t, strong)
            esc += 1
            c.append(rs["correct"]); cost.append(base_cost + rs["cost_usd"])
            lat.append(base_lat + rs["latency_s"]); calls.append(3)
    return Outcome(np.array(c, float), np.array(cost), np.array(lat), np.array(calls, float)), esc / len(ids)


def format_cascade(data: Data, ids, chain) -> tuple[Outcome, float]:
    """Try models in order; escalate only when the call fails or the answer is malformed."""
    c, cost, lat, calls, esc = [], [], [], [], 0
    for t in ids:
        tot_c = tot_l = 0.0
        for i, m in enumerate(chain):
            r = _row(data, t, m)
            tot_c += r["cost_usd"]; tot_l += r["latency_s"]
            if (r["ok"] and r["well_formed"]) or i == len(chain) - 1:
                esc += i > 0
                c.append(r["correct"]); cost.append(tot_c); lat.append(tot_l); calls.append(i + 1)
                break
    return Outcome(np.array(c, float), np.array(cost), np.array(lat), np.array(calls, float)), esc / len(ids)


# ------------------------------------------------------------------- stats
def boot_ci(x: np.ndarray, rng) -> tuple[float, float]:
    n = len(x)
    idx = rng.integers(0, n, size=(BOOT, n))
    m = x[idx].mean(axis=1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def paired_diff_ci(a: np.ndarray, b: np.ndarray, rng) -> tuple[float, float, float]:
    d = a - b
    lo, hi = boot_ci(d, rng)
    return float(d.mean()), lo, hi


def summarise(o: Outcome, rng) -> dict:
    lo, hi = boot_ci(o.correct, rng)
    return {"accuracy": float(o.correct.mean()), "acc_ci": [lo, hi],
            "cost_per_1k_usd": float(o.cost.mean() * 1000),
            "p50_latency_s": float(np.median(o.latency)),
            "p95_latency_s": float(np.quantile(o.latency, 0.95)),
            "mean_calls": float(o.calls.mean())}


# ---------------------------------------------------------------- pipeline
def evaluate(run_dir: str | Path) -> dict:
    data = load(run_dir)
    rng = np.random.default_rng(7)
    dev, test = data.ids("dev"), data.ids("test")
    models = data.models
    if min(len(dev), len(test)) < 20:
        raise SystemExit(f"Only {len(dev)} dev / {len(test)} test tasks have responses from every "
                         "model. That is a smoke test, not an evaluation: check results.jsonl for "
                         "errors, then run the full task set.")

    def total_cost(m, ids):
        return sum(_row(data, t, m)["cost_usd"] for t in ids)

    by_price = sorted(models, key=lambda m: total_cost(m, dev))          # cheapest first (dev)
    dev_acc = {m: single(data, dev, m).correct.mean() for m in models}
    best = max(models, key=lambda m: (dev_acc[m], -total_cost(m, dev)))   # chosen on dev
    target = dev_acc[best] - TARGET_GAP
    cuts = length_buckets(data, dev)

    # ---- candidate strategies, each evaluated on dev and test
    cands = []
    for m in models:
        cands.append({"family": "single", "name": m, "params": {"model": m},
                      "dev": single(data, dev, m), "test": single(data, test, m)})
    for eps in EPSILONS:
        table = fit_router(data, dev, eps, by_price, cuts)
        od, _ = apply_router(data, dev, table, best, cuts)
        ot, picks = apply_router(data, test, table, best, cuts)
        cands.append({"family": "rule_router", "name": f"rule router (eps={eps:.2f})",
                      "params": {"eps": eps, "table": {f"{k[0]}/len{k[1]}": v for k, v in sorted(table.items())}},
                      "dev": od, "test": ot,
                      "test_mix": {m: picks.count(m) / len(picks) for m in models}})
    cheap = [m for m in by_price if m != best]
    for a, b in itertools.combinations(cheap, 2):
        od, ed = agreement_cascade(data, dev, a, b, best)
        ot, et = agreement_cascade(data, test, a, b, best)
        cands.append({"family": "agreement_cascade", "name": f"agree({a.split('/')[-1]},{b.split('/')[-1]})->{best.split('/')[-1]}",
                      "params": {"a": a, "b": b, "strong": best},
                      "dev": od, "test": ot, "dev_escalation": ed, "test_escalation": et})
    for start in cheap:
        chain = [start, best]
        od, ed = format_cascade(data, dev, chain)
        ot, et = format_cascade(data, test, chain)
        cands.append({"family": "format_cascade", "name": f"format({start.split('/')[-1]})->{best.split('/')[-1]}",
                      "params": {"chain": chain}, "dev": od, "test": ot,
                      "dev_escalation": ed, "test_escalation": et})
    orc_test = oracle(data, test, by_price)

    # ---- selection on dev: per family, cheapest candidate meeting the target
    selected = {}
    for fam in ("rule_router", "agreement_cascade", "format_cascade"):
        ok = [c for c in cands if c["family"] == fam and c["dev"].correct.mean() >= target]
        if ok:
            selected[fam] = min(ok, key=lambda c: c["dev"].cost.mean())
        else:  # nothing meets the target: report the most accurate one, flagged
            fam_c = [c for c in cands if c["family"] == fam]
            if fam_c:
                selected[fam] = max(fam_c, key=lambda c: c["dev"].correct.mean()) | {"missed_target": True}

    best_test = next(c for c in cands if c["family"] == "single" and c["name"] == best)["test"]

    def describe(c):
        s = summarise(c["test"], rng)
        s["name"] = c["name"]
        s["family"] = c["family"]
        s["dev_accuracy"] = float(c["dev"].correct.mean())
        s["dev_cost_per_1k_usd"] = float(c["dev"].cost.mean() * 1000)
        for k in ("dev_escalation", "test_escalation", "test_mix", "missed_target"):
            if k in c:
                s[k] = c[k]
        s["params"] = c["params"]
        d, lo, hi = paired_diff_ci(c["test"].correct, best_test.correct, rng)
        s["acc_minus_best"] = [d, lo, hi]
        bc = best_test.cost.mean()
        s["cost_saving_vs_best"] = float(1 - c["test"].cost.mean() / bc) if bc > 0 else None
        return s

    # ---- reliability (all tasks, both splits)
    rel = {}
    all_ids = list(data.tasks)
    for m in models:
        rows = [_row(data, t, m) for t in all_ids]
        rel[m] = {"calls": len(rows),
                  "failed_after_retries": sum(not r["ok"] for r in rows),
                  "needed_retry": sum(r["attempts"] > 1 for r in rows),
                  "malformed": sum(r["ok"] and not r["well_formed"] for r in rows),
                  "truncated": sum((r.get("finish_reason") or "") == "length" for r in rows)}
    # availability with a one-step fallback chain best -> cheapest other model (replayed)
    fb = cheap[0] if cheap else None
    if fb:
        both_fail = sum((not _row(data, t, best)["ok"]) and (not _row(data, t, fb)["ok"]) for t in all_ids)
        rel["_fallback"] = {"chain": [best, fb],
                            "primary_failure_rate": rel[best]["failed_after_retries"] / len(all_ids),
                            "chain_failure_rate": both_fail / len(all_ids)}

    # ---- per-type x difficulty accuracy on test (analysis only)
    breakdown = {}
    for m in models:
        for ty in TASK_TYPES:
            for diff in ("easy", "medium", "hard"):
                ids = [t for t in test if data.tasks[t]["type"] == ty and data.tasks[t]["difficulty"] == diff]
                if ids:
                    breakdown.setdefault(m, {})[f"{ty}/{diff}"] = float(np.mean([_row(data, t, m)["correct"] for t in ids]))

    return {
        "mock": data.mock,
        "n_dev": len(dev), "n_test": len(test), "models": models,
        "prices_usd_per_m": data.prices,
        "best_single_on_dev": best, "target_dev_accuracy": float(target),
        "singles": [describe(c) for c in cands if c["family"] == "single"],
        "selected": {f: describe(c) for f, c in selected.items()},
        "all_candidates": [describe(c) for c in cands if c["family"] != "single"],
        "oracle_test": summarise(orc_test, rng),
        "reliability": rel,
        "breakdown_test": breakdown,
    }
