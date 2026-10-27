"""Command-line entry point; default execution never selects a provider."""

import argparse
import json
from pathlib import Path

from .cases import CASE_NAMES
from .runner import evaluate_case


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate a synthetic six-turn conversation with DeepEval.")
    parser.add_argument("--case", choices=CASE_NAMES, default="retained")
    parser.add_argument("--live", action="store_true", help="Call a paid provider using environment credentials")
    parser.add_argument("--report", type=Path, help="Also save the JSON result to this path")
    args = parser.parse_args(argv)
    try:
        report = evaluate_case(args.case, live=args.live)
    except ValueError as exc:
        parser.exit(2, f"Configuration error: {exc}\n")
    output = json.dumps(report, indent=2, ensure_ascii=False)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(output + "\n", encoding="utf-8")
    print(output)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
