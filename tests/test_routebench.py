import asyncio
import itertools
import json
import re
import shutil
from pathlib import Path

import pytest

from routebench import analyze
from routebench.client import Cache, ClientConfig, Gateway, BudgetExceeded, Price, Backend
from routebench.run import collect
from routebench.tasks import TASK_TYPES, canonical, generate, grade, parse_answer

TASKS = generate()


# ------------------------------------------------------------------ tasks
def test_generation_is_deterministic():
    again = generate()
    assert [t.to_dict() for t in again] == [t.to_dict() for t in TASKS]


def test_split_is_stratified_and_disjoint():
    for ty in TASK_TYPES:
        for diff in ("easy", "medium", "hard"):
            grp = [t for t in TASKS if t.type == ty and t.difficulty == diff]
            dev = sum(t.split == "dev" for t in grp)
            assert abs(dev - len(grp) / 2) <= 1
    assert len({t.id for t in TASKS}) == len(TASKS)


@pytest.mark.parametrize("task", TASKS, ids=lambda t: t.id)
def test_gold_answer_grades_correct(task):
    assert grade(task.type, task.answer, f"some reasoning\nANSWER: {task.answer}") == (True, True)


def test_order_puzzles_have_a_unique_answer():
    for t in (t for t in TASKS if t.type == "order"):
        facts = re.findall(r"(\w+) is (taller|shorter) than (\w+)\.", t.prompt)
        names = sorted({a for a, _, _ in facts} | {b for _, _, b in facts})
        k = {"tallest": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7}
        word = re.search(r"Who is the (\w+)", t.prompt).group(1)
        valid = []
        for perm in itertools.permutations(names):
            pos = {n: i for i, n in enumerate(perm)}  # 0 = tallest
            if all((pos[a] < pos[b]) if rel == "taller" else (pos[a] > pos[b]) for a, rel, b in facts):
                valid.append(perm)
        answers = {p[k[word] - 1] for p in valid}
        assert answers == {t.answer}, t.id


def test_hard_invoice_total_is_consistent_with_printed_lines():
    for t in (t for t in TASKS if t.type == "extract" and t.difficulty == "hard"):
        num = lambda label: float(re.search(label + r"[^:]*:\s*-?([\d,]+\.\d\d)", t.prompt).group(1).replace(",", ""))
        total = num("Subtotal") - num("Loyalty discount") + num("VAT")
        assert round(total, 2) == json.loads(t.answer)["total"], t.id


def test_parse_takes_last_answer_and_flags_malformed():
    assert parse_answer("ANSWER: 1\nwait\nANSWER: 2") == "2"
    assert grade("arith", "5", "no final line") == (False, False)
    assert grade("arith", "5", "ANSWER: 5 boxes") == (True, True)
    assert grade("arith", "1500", "ANSWER: 1,500") == (True, True)
    assert grade("order", "Bo", "ANSWER: **Bo**") == (True, True)
    assert grade("date", "2025-01-02", "ANSWER: 2025-01-03") == (False, True)
    gold = TASKS[[t.type for t in TASKS].index("extract")].answer
    g = json.loads(gold)
    assert grade("extract", gold, "ANSWER: " + json.dumps(dict(g, total=str(g["total"])))) == (True, True)
    assert grade("extract", gold, "ANSWER: {\"invoice_id\": 1}")[1] is False


def test_router_feature_detects_family_from_prompt_only():
    for t in TASKS:
        assert analyze.detect_type(t.prompt) == t.type, t.id


# ---------------------------------------------------------------- gateway
class Scripted(Backend):
    """Returns scripted (status, retry_after) per call for each model."""

    def __init__(self, script):
        self.script = {m: list(v) for m, v in script.items()}
        self.calls = []

    async def post(self, model, messages, params, timeout):
        self.calls.append(model)
        status, retry_after = self.script[model].pop(0)
        if status == 200:
            return 200, {"choices": [{"message": {"content": "ANSWER: 1"}, "finish_reason": "stop"}],
                         "usage": {"prompt_tokens": 1000, "completion_tokens": 1000}}, None, None
        return status, None, f"http_{status}", retry_after


def _gw(backend, tmp_path=None, budget=10.0, retries=3):
    sleeps = []

    async def fake_sleep(s):
        sleeps.append(s)

    cfg = ClientConfig(base_url="x", api_key="k", max_retries=retries, budget_usd=budget)
    gw = Gateway(cfg, backend, {"a": Price(1, 1), "b": Price(2, 2)},
                 cache=Cache(tmp_path / "c.sqlite") if tmp_path else None, sleep=fake_sleep)
    return gw, sleeps


MSG = [{"role": "user", "content": "q"}]


def test_retry_then_success_with_backoff():
    be = Scripted({"a": [(503, None), (429, "7"), (200, None)]})
    gw, sleeps = _gw(be)
    r = asyncio.run(gw.call("a", MSG))
    assert r.ok and r.attempts == 3
    assert sleeps[1] >= 7  # Retry-After honoured
    assert r.cost_usd == pytest.approx(0.002)


def test_non_retryable_error_fails_fast():
    be = Scripted({"a": [(400, None), (200, None)]})
    gw, _ = _gw(be)
    r = asyncio.run(gw.call("a", MSG))
    assert not r.ok and r.attempts == 1 and be.calls == ["a"]


def test_fallback_chain_uses_second_model():
    be = Scripted({"a": [(503, None)] * 4, "b": [(200, None)]})
    gw, _ = _gw(be)
    r = asyncio.run(gw.complete(["a", "b"], MSG))
    assert r.ok and r.served_by == "b" and r.fallback_used


def test_cache_avoids_second_call(tmp_path):
    be = Scripted({"a": [(200, None)]})
    gw, _ = _gw(be, tmp_path)
    r1 = asyncio.run(gw.call("a", MSG))
    r2 = asyncio.run(gw.call("a", MSG))
    assert r1.ok and r2.ok and r2.cached and len(be.calls) == 1


def test_budget_stops_new_calls():
    be = Scripted({"a": [(200, None)] * 5})
    gw, _ = _gw(be, budget=0.003)
    asyncio.run(gw.call("a", MSG))
    asyncio.run(gw.call("a", MSG))
    with pytest.raises(BudgetExceeded):
        asyncio.run(gw.call("a", MSG))


# ---------------------------------------------------------------- analysis
@pytest.fixture(scope="module")
def mock_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("run")
    cfg = Path(__file__).resolve().parents[1] / "config.yaml"
    asyncio.run(collect(str(cfg), str(out), mock=True))
    return out


def test_mock_run_is_complete_and_resumable(mock_run):
    rows = (mock_run / "results.jsonl").read_text().splitlines()
    assert len(rows) == 400 * 4
    asyncio.run(collect(str(Path(__file__).resolve().parents[1] / "config.yaml"), str(mock_run), mock=True))
    assert len((mock_run / "results.jsonl").read_text().splitlines()) == 1600  # nothing re-run


def test_selection_does_not_look_at_test(mock_run, tmp_path):
    ev1 = analyze.evaluate(mock_run)
    # corrupt every test-split outcome; dev-based choices must not change
    alt = tmp_path / "alt"
    shutil.copytree(mock_run, alt)
    lines = []
    for line in (alt / "results.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["split"] == "test":
            r["correct"] = not r["correct"]
            r["cost_usd"] *= 3
        lines.append(json.dumps(r))
    (alt / "results.jsonl").write_text("\n".join(lines) + "\n")
    ev2 = analyze.evaluate(alt)
    assert ev1["best_single_on_dev"] == ev2["best_single_on_dev"]
    for fam in ev1["selected"]:
        assert ev1["selected"][fam]["params"] == ev2["selected"][fam]["params"]
        assert ev1["selected"][fam]["dev_accuracy"] == ev2["selected"][fam]["dev_accuracy"]


def test_cascade_accounting(mock_run):
    data = analyze.load(mock_run)
    ids = data.ids("test")
    a, b, s = "mock/tiny", "mock/small", "mock/large"
    out, esc = analyze.agreement_cascade(data, ids, a, b, s)
    base = sum(data.R[(t, a)]["cost_usd"] + data.R[(t, b)]["cost_usd"] for t in ids)
    assert out.cost.sum() >= base
    assert 0 <= esc <= 1
    assert out.calls.mean() == pytest.approx(2 + esc)


def test_canonical_agreement_ignores_formatting():
    assert canonical("arith", "ANSWER: 1,200") == canonical("arith", "x\nANSWER: 1200")
