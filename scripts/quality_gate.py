"""Fail when retrieval quality drops below a stored baseline.

Compares the newest report of a suite with its baseline in config/baselines/.
A metric fails when it is lower than baseline minus the tolerance.

Suites:
    scifact  experiments/beir_scifact_*.json  (scripts.beir_benchmark --dataset scifact)
    own      experiments/retrieval_eval_*.json (scripts.retrieval_eval on your documents)

Usage:
    python -m scripts.quality_gate --suite scifact
    python -m scripts.quality_gate --suite own --report experiments/retrieval_eval_x.json
    python -m scripts.quality_gate --suite own --update     # accept current numbers
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

SUITES = {"scifact": "beir_scifact_*.json", "own": "retrieval_eval_*.json"}


def lookup(report: dict, path: str):
    """Value at a dotted path; 'config.metric' also looks inside configs[...]['metrics']."""
    cfg, _, metric = path.partition(".")
    if cfg in report.get("configs", {}) and metric:
        node = report["configs"][cfg]
        return node.get("metrics", node).get(metric)
    node = report
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def latest_report(folder: Path, pattern: str) -> Path:
    files = sorted(folder.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"no report matching {folder / pattern}")
    return files[-1]


def check(report: dict, baseline: dict) -> tuple[list[dict], list[str]]:
    """Rows per metric, plus problems that make the report not comparable."""
    problems = [f"{key}: report has {lookup(report, key)!r}, baseline expects {want!r}"
                for key, want in baseline.get("match", {}).items()
                if lookup(report, key) != want]
    tol = baseline.get("tolerance", 0.0)
    rows = []
    for path, want in baseline["metrics"].items():
        got = lookup(report, path)
        ok = isinstance(got, (int, float)) and got >= want - tol
        rows.append({"metric": path, "baseline": want, "value": got,
                     "delta": None if got is None else round(got - want, 4), "ok": ok})
    return rows, problems


def updated(report: dict, baseline: dict, source: str) -> dict:
    new = dict(baseline)
    new["metrics"] = {k: lookup(report, k) for k in baseline["metrics"]}
    new["source"], new["date"] = source, date.today().isoformat()
    return new


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Retrieval quality gate")
    parser.add_argument("--suite", choices=sorted(SUITES), required=True)
    parser.add_argument("--report", help="report file (default: newest for the suite)")
    parser.add_argument("--reports-dir", default="experiments")
    parser.add_argument("--baselines-dir", default="config/baselines")
    parser.add_argument("--update", action="store_true",
                        help="write the report's values as the new baseline")
    args = parser.parse_args(argv)

    base_path = Path(args.baselines_dir) / f"{args.suite}.json"
    baseline = json.loads(base_path.read_text())
    report_path = Path(args.report) if args.report else latest_report(
        Path(args.reports_dir), SUITES[args.suite])
    report = json.loads(report_path.read_text())
    rows, problems = check(report, baseline)

    print(f"Quality gate: {args.suite}  report {report_path}  tolerance {baseline.get('tolerance', 0)}")
    for p in problems:
        print(f"  NOT COMPARABLE  {p}")
    for r in rows:
        delta = "missing" if r["delta"] is None else f"{r['delta']:+.4f}"
        print(f"  {'ok  ' if r['ok'] else 'FAIL'}  {r['metric']:<28} baseline {r['baseline']:<8} "
              f"now {r['value']!s:<8} {delta}")

    if args.update:
        if problems:
            print("Refusing to update: report not comparable with the baseline.")
            return 2
        base_path.write_text(json.dumps(updated(report, baseline, str(report_path)), indent=2) + "\n")
        print(f"Baseline updated: {base_path}")
        return 0
    failed = problems or [r for r in rows if not r["ok"]]
    print("RESULT:", "FAIL" if failed else "PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
