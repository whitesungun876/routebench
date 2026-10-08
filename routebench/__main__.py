"""CLI.

  python -m routebench models  --config config.yaml [--contains qwen]
  python -m routebench run     --config config.yaml --out runs/main [--yes] [--mock]
  python -m routebench analyze --run runs/main
"""
import argparse
import asyncio

from . import analyze, report, run


def main():
    p = argparse.ArgumentParser(prog="routebench")
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("models", help="list models and prices on the endpoint")
    m.add_argument("--config", default="config.yaml")
    m.add_argument("--contains")
    r = sub.add_parser("run", help="collect responses (resumable)")
    r.add_argument("--config", default="config.yaml")
    r.add_argument("--out", default="runs/main")
    r.add_argument("--mock", action="store_true", help="simulated backend, no API calls")
    r.add_argument("--yes", action="store_true", help="skip the cost confirmation")
    r.add_argument("--n-per-type", type=int)
    a = sub.add_parser("analyze", help="evaluate routing strategies and write the report")
    a.add_argument("--run", default="runs/main")
    args = p.parse_args()

    if args.cmd == "models":
        asyncio.run(run.list_models(args.config, args.contains))
    elif args.cmd == "run":
        out = asyncio.run(run.collect(args.config, args.out, mock=args.mock, yes=args.yes,
                                      n_per_type=args.n_per_type))
        print(f"done -> {out}. Next: python -m routebench analyze --run {out}")
    else:
        ev = analyze.evaluate(args.run)
        path = report.write(ev, args.run)
        print(f"report -> {path}")


if __name__ == "__main__":
    main()
