"""Procedurally generated tasks with deterministic ground truth.

Four task families, three difficulty levels each. Every task has an exact
answer computed by Python, so grading needs no LLM judge.

The difficulty label is stored for analysis only. Routers must not read it;
they may only use features computable from the prompt text (see router.py).
"""
from __future__ import annotations

import datetime as dt
import json
import random
import re
from dataclasses import asdict, dataclass

ANSWER_INSTRUCTION = (
    "Think step by step if needed, then end your reply with a single final line "
    "in the form `ANSWER: <answer>`."
)

DIFFICULTIES = ("easy", "medium", "hard")
TASK_TYPES = ("arith", "date", "extract", "order")


@dataclass
class Task:
    id: str
    type: str
    difficulty: str
    prompt: str
    answer: str
    split: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------- arithmetic
_ITEMS = [("warehouse", "boxes"), ("bakery", "loaves"), ("library", "books"),
          ("farm", "eggs"), ("depot", "tyres"), ("shop", "lamps")]
_DISTRACTORS = [
    ("The manager has worked there for {n} years.", 3, 30),
    ("The building is {n} metres from the station.", 50, 900),
    ("Last year the team ordered {n} new chairs, which is unrelated to stock.", 4, 40),
    ("A delivery truck can drive {n} km on a full tank.", 300, 900),
]


def _arith(rng: random.Random, difficulty: str) -> tuple[str, str]:
    steps = {"easy": 2, "medium": 4, "hard": 7}[difficulty]
    place, unit = rng.choice(_ITEMS)
    while True:
        value = rng.randrange(40, 400, 10)
        lines = [f"A {place} starts the day with {value} {unit}."]
        ok = True
        for _ in range(steps):
            op = rng.choice(["add", "sub", "ship", "pct", "mul"])
            if op == "add":
                n = rng.randint(5, 90)
                value += n
                lines.append(f"It receives {n} more {unit}.")
            elif op == "sub":
                n = rng.randint(5, 60)
                if n >= value:
                    ok = False
                    break
                value -= n
                lines.append(f"Customers take {n} {unit}.")
            elif op == "ship":
                k, n = rng.randint(2, 5), rng.randint(6, 40)
                value += k * n
                lines.append(f"Then {k} shipments arrive, each with {n} {unit}.")
            elif op == "pct":
                p = rng.choice([10, 20, 25, 50])
                if value * p % 100:
                    ok = False
                    break
                value -= value * p // 100
                lines.append(f"Then {p}% of the {unit} on hand are sent to another branch.")
            else:
                m = rng.choice([2, 3])
                value *= m
                lines.append(f"Then the stock is {'doubled' if m == 2 else 'tripled'} by a bulk order.")
        if not ok or value > 100000:
            continue
        if difficulty == "hard":
            for _ in range(2):
                pos = rng.randint(1, len(lines))
                tmpl, lo, hi = rng.choice(_DISTRACTORS)
                lines.insert(pos, tmpl.format(n=rng.randint(lo, hi)))
        q = f"How many {unit} does the {place} have at the end?"
        return " ".join(lines) + "\n" + q, str(value)


# ---------------------------------------------------------------------- date
_WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _date(rng: random.Random, difficulty: str) -> tuple[str, str]:
    start = dt.date(2023, 1, 1) + dt.timedelta(days=rng.randint(0, 1100))
    d = start
    ops = {"easy": 1, "medium": 2, "hard": 4}[difficulty]
    parts = [f"A project starts on {start.isoformat()}."]
    for i in range(ops):
        if difficulty == "easy":
            kind = "days"
        elif i == 0:  # the first step must create a milestone before it can be moved
            kind = rng.choice(["days", "weeks"])
        else:
            kind = rng.choice(["days", "weeks", "back", "next"])
        if kind == "days":
            n = rng.randint(3, 20) if difficulty == "easy" else rng.randint(20, 400)
            d += dt.timedelta(days=n)
            parts.append(f"The first milestone is {n} days after the project start." if i == 0
                         else f"The next milestone is {n} days after that.")
        elif kind == "weeks":
            n = rng.randint(2, 30)
            d += dt.timedelta(weeks=n)
            parts.append(f"The first milestone is {n} weeks after the project start." if i == 0
                         else f"The next milestone is {n} weeks after that.")
        elif kind == "back":
            n = rng.randint(2, 40)
            d -= dt.timedelta(days=n)
            parts.append(f"That milestone is then moved {n} days earlier.")
        else:
            wd = rng.randrange(7)
            delta = (wd - d.weekday()) % 7 or 7
            d += dt.timedelta(days=delta)
            parts.append(f"The milestone is pushed to the following {_WEEKDAYS[wd]} "
                         f"(strictly after the current milestone date).")
    q = "What is the date of the final milestone? Answer in YYYY-MM-DD format."
    return " ".join(parts) + "\n" + q, d.isoformat()


# ------------------------------------------------------------------- extract
_VENDORS = ["Nordlys Supplies ApS", "Harbour Logistics Ltd", "Kestrel Components GmbH",
            "Blue Fjord Trading", "Lotus Office Partners", "Granite Peak Tools",
            "Amber Lane Foods", "Copper Kettle Systems"]
_CUSTOMERS = ["Vesterbro Studio", "Aalto Robotics", "Meridian Health Clinic", "Saga Media"]


def _money(x: float) -> str:
    return f"{x:,.2f}"


def _extract(rng: random.Random, difficulty: str) -> tuple[str, str]:
    vendor = rng.choice(_VENDORS)
    customer = rng.choice(_CUSTOMERS)
    inv = f"INV-{rng.randint(10000, 99999)}"
    po = f"PO-{rng.randint(10000, 99999)}"
    issue = dt.date(2025, 1, 1) + dt.timedelta(days=rng.randint(0, 600))
    due = issue + dt.timedelta(days=rng.choice([14, 30, 45]))
    n_items = {"easy": 2, "medium": 3, "hard": 4}[difficulty]
    items = [(rng.choice(["Cable kit", "Desk lamp", "Sensor", "Paper (box)", "Adapter", "Filter"]),
              rng.randint(1, 12), rng.randint(500, 25000) / 100) for _ in range(n_items)]
    subtotal = round(sum(q * p for _, q, p in items), 2)
    lines = [f"From: {vendor}", f"Bill to: {customer}", f"Invoice number: {inv}",
             f"Issue date: {issue.isoformat()}"]
    if difficulty != "easy":
        lines.append(f"Customer reference: {po}")
    for name, q, p in items:
        lines.append(f"- {name} x{q} @ {_money(p)} = {_money(q * p)}")
    if difficulty == "easy":
        total = subtotal
        lines += [f"Due date: {due.isoformat()}", f"Total due: {_money(total)} EUR"]
    elif difficulty == "medium":
        tax = round(subtotal * 0.25, 2)
        total = round(subtotal + tax, 2)
        lines += [f"Subtotal: {_money(subtotal)}", f"VAT 25%: {_money(tax)}",
                  f"Total due: {_money(total)} EUR",
                  f"Payment terms: due by {due.isoformat()}"]
    else:
        # total is not printed; it must be computed and a paid balance ignored
        disc = round(subtotal * 0.05, 2)
        tax = round((subtotal - disc) * 0.25, 2)
        total = round(subtotal - disc + tax, 2)
        prev = rng.randint(10000, 90000) / 100
        lines += [f"Subtotal: {_money(subtotal)}",
                  f"Loyalty discount 5%: -{_money(disc)}",
                  f"VAT 25% on discounted amount: {_money(tax)}",
                  f"Previous balance {_money(prev)} (already paid, not included)",
                  f"Please pay within {(due - issue).days} days of the issue date."]
    text = "\n".join(lines)
    q = ("Extract the fields from this invoice. Return the answer as compact JSON with keys "
         '"invoice_id", "vendor", "due_date" (YYYY-MM-DD) and "total" (number, amount the '
         "customer must pay for this invoice).\n\n" + text)
    answer = json.dumps({"invoice_id": inv, "vendor": vendor, "due_date": due.isoformat(),
                         "total": round(total, 2)}, ensure_ascii=False)
    return q, answer


# --------------------------------------------------------------------- order
_NAMES = ["Asta", "Bo", "Carmen", "Dmitri", "Elif", "Freja", "Gustav", "Hana", "Ivo", "Jun"]
_ORDINAL = {1: "tallest", 2: "second tallest", 3: "third tallest", 4: "fourth tallest",
            5: "fifth tallest", 6: "sixth tallest", 7: "seventh tallest"}


def _order(rng: random.Random, difficulty: str) -> tuple[str, str]:
    n = {"easy": 3, "medium": 5, "hard": 7}[difficulty]
    people = rng.sample(_NAMES, n)  # people[0] is tallest
    facts = []
    for a, b in zip(people, people[1:]):  # adjacent pairs give a unique total order
        if difficulty == "hard" and rng.random() < 0.5:
            facts.append(f"{b} is shorter than {a}.")
        else:
            facts.append(f"{a} is taller than {b}.")
    rng.shuffle(facts)
    k = rng.randint(1, n)
    q = (f"There are {n} people. " + " ".join(facts) +
         f"\nWho is the {_ORDINAL[k]}? Answer with the name only.")
    return q, people[k - 1]


_GENERATORS = {"arith": _arith, "date": _date, "extract": _extract, "order": _order}


def generate(n_per_type: int = 100, seed: int = 20261008, dev_fraction: float = 0.5) -> list[Task]:
    """Generate tasks and a frozen stratified dev/test split."""
    rng = random.Random(seed)
    tasks: list[Task] = []
    for ttype in TASK_TYPES:
        for i in range(n_per_type):
            diff = DIFFICULTIES[i % 3]
            prompt, answer = _GENERATORS[ttype](rng, diff)
            tasks.append(Task(id=f"{ttype}-{i:03d}", type=ttype, difficulty=diff,
                              prompt=prompt + "\n\n" + ANSWER_INSTRUCTION, answer=answer))
    # stratified split by (type, difficulty)
    by_stratum: dict[tuple[str, str], list[Task]] = {}
    for t in tasks:
        by_stratum.setdefault((t.type, t.difficulty), []).append(t)
    split_rng = random.Random(seed + 1)
    for group in by_stratum.values():
        split_rng.shuffle(group)
        cut = round(len(group) * dev_fraction)
        for j, t in enumerate(group):
            t.split = "dev" if j < cut else "test"
    return tasks


# ------------------------------------------------------------------- grading
def parse_answer(text: str | None) -> str | None:
    if not text:
        return None
    m = None
    for m in re.finditer(r"ANSWER:\s*(.+)", text, re.IGNORECASE):
        pass
    if not m:
        return None
    ans = m.group(1).strip().strip("`").strip()
    return ans or None


def _num(s: str) -> float | None:
    s = s.replace(",", "").replace("EUR", "").strip().rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def normalise(task_type: str, raw: str | None):
    """Return a canonical form of an answer, or None if it is not well-formed."""
    if raw is None:
        return None
    if task_type == "arith":
        v = _num(raw.split()[0] if raw.split() else raw)
        return int(v) if v is not None and v == int(v) else None
    if task_type == "date":
        m = re.search(r"\d{4}-\d{2}-\d{2}", raw)
        return m.group(0) if m else None
    if task_type == "order":
        w = re.findall(r"[A-Za-z]+", raw)
        return w[0].lower() if w else None
    if task_type == "extract":
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
        keys = ("invoice_id", "vendor", "due_date", "total")
        if not all(k in obj for k in keys):
            return None
        total = obj["total"] if isinstance(obj["total"], (int, float)) else _num(str(obj["total"]))
        if total is None:
            return None
        return (str(obj["invoice_id"]).strip(), str(obj["vendor"]).strip().lower(),
                str(obj["due_date"]).strip(), round(float(total), 2))
    raise ValueError(task_type)


def grade(task_type: str, gold: str, output_text: str | None) -> tuple[bool, bool]:
    """Return (correct, well_formed) for a model output."""
    pred = normalise(task_type, parse_answer(output_text))
    if pred is None:
        return False, False
    gold_norm = normalise(task_type, gold)
    return pred == gold_norm, True


def canonical(task_type: str, output_text: str | None):
    """Canonical answer used to compare two models' outputs (for cascades)."""
    return normalise(task_type, parse_answer(output_text))
