# RouteBench

A small, reproducible harness for one question an AI gateway has to answer on every request:
**which model should serve this prompt, and what happens when that model fails?**

RouteBench sends the same set of checkable tasks to several models behind one
OpenAI-compatible endpoint (OpenRouter, NexLLM or any other), records quality, cost,
latency and failures per call, and then replays routing strategies offline on those
recorded responses.

## What it measures

| Layer | What is implemented |
|---|---|
| Tasks | 400 generated tasks in 4 families (multi-step arithmetic, date arithmetic, invoice field extraction, ordering puzzles) x 3 difficulty levels. Every answer is computed by Python, so grading needs no LLM judge. Frozen, stratified dev/test split. |
| Gateway | Async client with per-call timeout, exponential backoff with jitter, `Retry-After` support, fail-fast on non-retryable 4xx, ordered fallback chain, SQLite response cache, token and USD accounting, hard budget stop. Runs are resumable. |
| Strategies | Single-model baselines; a rule router (detected task family x prompt-length tercile, cheapest model within epsilon of the best in each cell); an agreement cascade (two cheap models in parallel, escalate to the strong model when their answers disagree); a format cascade (escalate only on failure or malformed output); an oracle upper bound. |
| Protocol | Every strategy parameter is chosen on dev. Test is evaluated once. Accuracy differences against the best single model are reported with paired bootstrap 95% CIs. Routers see only the prompt text, never the generator's difficulty label. |
| Reliability | Failure-after-retry rate, retry rate, malformed and truncated outputs per model, and the failure rate of a replayed fallback chain. |

## Run it

```bash
pip install -r requirements.txt

# 1. dry run on the simulated backend (no key, no cost)
python -m routebench run --mock --out runs/mock
python -m routebench analyze --run runs/mock

# 2. pick current model ids and prices on your endpoint
export OPENROUTER_API_KEY=sk-or-...
python -m routebench models --contains qwen

# 3. edit config.yaml, then a 12-task smoke test, then the full run
python -m routebench run --out runs/smoke --n-per-type 3
python -m routebench run --out runs/main
python -m routebench analyze --run runs/main
```

`runs/main/report.md`, `summary.json` and `pareto.png` hold the results. The mock
backend exists only to test the pipeline. Its numbers are stamped SIMULATED and are not results.

## Tests

```bash
python -m pytest -q
```

Covers: every gold answer grades as correct, ordering puzzles have a unique solution
(brute force), invoice totals match the printed lines, the router feature recovers the
task family from the prompt alone, retry / Retry-After / fail-fast / fallback / cache /
budget behaviour of the gateway, resumable collection, and a check that corrupting
every test-split outcome leaves all dev-based choices unchanged.

## Limits

- Synthetic tasks with exact answers measure routing behaviour on checkable work, not
  general model quality. Open-ended tasks would need a judge and a different verifier.
- One sample per (task, model) at temperature 0. Provider-side nondeterminism is not measured.
- Router cells hold roughly 16 dev tasks each, so cell-level choices are noisy; the paired
  CI on test is the number to trust.
