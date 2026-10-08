"""Write report.md, summary.json and pareto.png from an evaluation dict."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
FAMILY_STYLE = {  # validated categorical slots 1-3 (all-pairs safe) + neutral for baselines
    "single": ("#8a8984", "o", "single model"),
    "rule_router": ("#2a78d6", "s", "rule router"),
    "agreement_cascade": ("#eb6834", "^", "agreement cascade"),
    "format_cascade": ("#1baf7a", "D", "format cascade"),
}


def _short(m: str) -> str:
    return m.split("/")[-1]


def _pct(x):
    return f"{100 * x:.1f}%"


def pareto_plot(ev: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    pts = [(s, "single") for s in ev["singles"]] + [(c, c["family"]) for c in ev["all_candidates"]]
    for fam, (color, marker, label) in FAMILY_STYLE.items():
        xs = [max(s["cost_per_1k_usd"], 1e-4) for s, f in pts if f == fam]
        ys = [100 * s["accuracy"] for s, f in pts if f == fam]
        if xs:
            ax.scatter(xs, ys, s=46, c=color, marker=marker, label=label,
                       edgecolors=SURFACE, linewidths=1.5, zorder=3)
    for s in ev["singles"]:
        ax.annotate(_short(s["name"]), (max(s["cost_per_1k_usd"], 1e-4), 100 * s["accuracy"]),
                    xytext=(6, -10), textcoords="offset points", fontsize=8, color=INK2)
    sel = list(ev["selected"].values())
    ax.scatter([max(s["cost_per_1k_usd"], 1e-4) for s in sel], [100 * s["accuracy"] for s in sel],
               s=170, facecolors="none", edgecolors=INK, linewidths=1.2, zorder=4,
               label="ring: chosen on dev")
    o = ev["oracle_test"]
    ax.scatter([max(o["cost_per_1k_usd"], 1e-4)], [100 * o["accuracy"]], marker="*", s=120,
               c=INK, label="oracle (upper bound)", zorder=3)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v:g}"))
    ax.set_xlabel("Cost per 1,000 tasks (USD, log scale)", color=INK2)
    ax.set_ylabel("Test accuracy (%)", color=INK2)
    title = "Accuracy vs cost on the frozen test split"
    if ev["mock"]:
        title += "  [SIMULATED DATA, NOT A RESULT]"
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.grid(True, color=GRID, linewidth=0.8)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.tick_params(colors=INK2)
    leg = ax.legend(frameon=False, fontsize=8, loc="lower right")
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def write(ev: dict, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    (out / "summary.json").write_text(json.dumps(ev, indent=2, ensure_ascii=False))
    pareto_plot(ev, out / "pareto.png")
    best = ev["best_single_on_dev"]
    L = []
    if ev["mock"]:
        L += ["> **SIMULATED RUN.** These numbers come from the mock backend and only show that the",
              "> pipeline works. They are not results and must not be reported anywhere.", ""]
    L += ["# Routing benchmark report", "",
          f"Tasks: {ev['n_dev']} dev / {ev['n_test']} test (frozen split). Models: "
          + ", ".join(f"`{m}`" for m in ev["models"]) + ".",
          f"Best single model on dev: `{best}`. Selection target: dev accuracy >= "
          f"{_pct(ev['target_dev_accuracy'])} (best minus 1pp), then lowest dev cost.", "",
          "## Single models (test)", "",
          "| model | accuracy [95% CI] | cost / 1k tasks | p50 / p95 latency |",
          "|---|---|---|---|"]
    for s in sorted(ev["singles"], key=lambda s: s["cost_per_1k_usd"]):
        L.append(f"| `{s['name']}` | {_pct(s['accuracy'])} [{_pct(s['acc_ci'][0])}, {_pct(s['acc_ci'][1])}] "
                 f"| ${s['cost_per_1k_usd']:.3f} | {s['p50_latency_s']:.1f}s / {s['p95_latency_s']:.1f}s |")
    L += ["", "## Strategies selected on dev, evaluated once on test", "",
          "| strategy | test accuracy | vs best model (paired 95% CI) | cost / 1k | saving vs best | p50 / p95 latency | escalation |",
          "|---|---|---|---|---|---|---|"]
    for fam, s in ev["selected"].items():
        d, lo, hi = s["acc_minus_best"]
        esc = _pct(s["test_escalation"]) if "test_escalation" in s else "-"
        flag = " (missed dev target)" if s.get("missed_target") else ""
        sav = _pct(s["cost_saving_vs_best"]) if s["cost_saving_vs_best"] is not None else "-"
        L.append(f"| {s['name']}{flag} | {_pct(s['accuracy'])} | {100 * d:+.1f}pp [{100 * lo:+.1f}, {100 * hi:+.1f}] "
                 f"| ${s['cost_per_1k_usd']:.3f} | {sav} | {s['p50_latency_s']:.1f}s / {s['p95_latency_s']:.1f}s | {esc} |")
    o = ev["oracle_test"]
    L += ["", f"Oracle router (cheapest model that was correct, upper bound): {_pct(o['accuracy'])} "
          f"at ${o['cost_per_1k_usd']:.3f} per 1k tasks.", "",
          "![accuracy vs cost](pareto.png)", "",
          "## Rule router table (selected)", ""]
    rr = ev["selected"].get("rule_router")
    if rr:
        L.append("Cell = detected task family / prompt-length tercile (fitted on dev).")
        L.append("")
        for k, v in rr["params"]["table"].items():
            L.append(f"- `{k}` -> `{v}`")
        if "test_mix" in rr:
            L.append("")
            L.append("Test traffic share: " + ", ".join(f"`{_short(m)}` {_pct(p)}" for m, p in rr["test_mix"].items()))
    L += ["", "## Reliability (all calls)", "",
          "| model | calls | failed after retries | needed a retry | malformed answer | truncated |",
          "|---|---|---|---|---|---|"]
    for m in ev["models"]:
        r = ev["reliability"][m]
        L.append(f"| `{m}` | {r['calls']} | {r['failed_after_retries']} | {r['needed_retry']} | {r['malformed']} | {r['truncated']} |")
    fb = ev["reliability"].get("_fallback")
    if fb:
        L.append("")
        L.append(f"Fallback `{_short(fb['chain'][0])}` -> `{_short(fb['chain'][1])}`: failure rate "
                 f"{_pct(fb['primary_failure_rate'])} -> {_pct(fb['chain_failure_rate'])} (replayed from recorded calls).")
    L += ["", "## Test accuracy by family and difficulty (analysis only, never used for routing)", ""]
    _rank = {"easy": 0, "medium": 1, "hard": 2}
    cols = sorted({k for m in ev["breakdown_test"].values() for k in m},
                  key=lambda c: (c.split("/")[0], _rank.get(c.split("/")[1], 9)))
    L.append("| model | " + " | ".join(cols) + " |")
    L.append("|---|" + "---|" * len(cols))
    for m, row in ev["breakdown_test"].items():
        L.append(f"| `{_short(m)}` | " + " | ".join(_pct(row.get(c, float('nan'))) for c in cols) + " |")
    L += ["", "## All candidate strategies (test)", "",
          "| strategy | dev acc | test acc | cost / 1k | escalation |", "|---|---|---|---|---|"]
    for c in ev["all_candidates"]:
        esc = _pct(c["test_escalation"]) if "test_escalation" in c else "-"
        L.append(f"| {c['name']} | {_pct(c['dev_accuracy'])} | {_pct(c['accuracy'])} | ${c['cost_per_1k_usd']:.3f} | {esc} |")
    L += ["", "## Method notes", "",
          "- One response per (task, model) at temperature 0; strategies are replayed offline from those responses.",
          "- Strategy parameters are chosen on dev only; test is evaluated once. Looking at test and then re-tuning turns it into a dev set.",
          "- A call that still fails after retries counts as wrong for that model.",
          "- Cascade latency assumes the two cheap calls run in parallel and escalation is sequential.",
          "- Tasks are synthetic with exact answers. They measure routing behaviour on checkable tasks, not general model quality.",
          "- Cost uses the provider-reported cost when present, otherwise listed per-token prices."]
    (out / "report.md").write_text("\n".join(L) + "\n")
    return out / "report.md"
