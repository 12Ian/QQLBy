"""Check reproduction progress against the numbers reported in the paper.

The paper's quantitative anchors are:

  * IV-A nine-class maneuver intention      ~88 %  test accuracy
  * IV-A eleven-level overload intention    ~94 %  test accuracy
  * IV-B trajectory extrapolation           ADE/FDE converging to ~102 m
  * Fig. 14 terminal errors (Table VI)      150 / 202 / 81 / 173 m

Point this script at the artefacts a reproduction run produced and it prints a
single table with the measured value, the paper target, and the relative gap.

Usage:
    python scripts/check_paper_targets.py \
        --intention  outputs/intention/test_metrics.json \
        --overload   outputs/overload/test_metrics.json \
        --forecast   outputs/forecast/validation_metrics.json \
        --figure14   outputs/figure14/figure14_summary.json
Every argument is optional; missing inputs are reported as "not provided".
"""
import argparse
import json
from pathlib import Path

# paper anchors -------------------------------------------------------------
TARGETS = {
    "intention_accuracy": 88.0,
    "overload_accuracy": 94.0,
    "forecast_ade_m": 102.0,
}
FIGURE14_PAPER = {"Left-Climb": 150., "Left-Dive": 202., "Right-Climb": 81., "Right-Dive": 173.}
# Reproduction is accepted as "aligned" within this relative band; the paper
# reports single-run statistics, so exact equality is not expected.
RELATIVE_TOLERANCE_PERCENT = 15.0


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def accuracy_of(path):
    data = load(path)
    value = data.get("intention_accuracy")
    if value is None:
        value = data.get("validation", {}).get("intention_accuracy")
    return None if value is None else 100. * float(value)


def forecast_ade_of(path):
    data = load(path)
    value = data.get("ade_m", data.get("validation", {}).get("ade_m"))
    return None if value is None else float(value)


def row(name, measured, target, unit, tolerance=RELATIVE_TOLERANCE_PERCENT):
    if measured is None:
        return dict(item=name, measured=None, target=target, unit=unit, gap=None, verdict="not provided")
    gap = measured - target
    relative = 100. * gap / target if target else None
    verdict = "ok" if relative is not None and abs(relative) <= tolerance else "off target"
    return dict(item=name, measured=measured, target=target, unit=unit,
                gap=gap, relative_percent=relative, verdict=verdict)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intention")
    parser.add_argument("--overload")
    parser.add_argument("--forecast")
    parser.add_argument("--figure14")
    parser.add_argument("--tolerance", type=float, default=RELATIVE_TOLERANCE_PERCENT)
    parser.add_argument("--output")
    args = parser.parse_args()

    rows = [
        row("IV-A nine-class accuracy", accuracy_of(args.intention) if args.intention else None,
            TARGETS["intention_accuracy"], "%", args.tolerance),
        row("IV-A eleven-level accuracy", accuracy_of(args.overload) if args.overload else None,
            TARGETS["overload_accuracy"], "%", args.tolerance),
        row("IV-B ADE", forecast_ade_of(args.forecast) if args.forecast else None,
            TARGETS["forecast_ade_m"], "m", args.tolerance),
    ]
    if args.figure14:
        summary = load(args.figure14)
        for entry in summary.get("rows", []):
            rows.append(row(f"Fig. 14 {entry['scenario']}", float(entry["terminal_error_m"]),
                            FIGURE14_PAPER.get(entry["scenario"]), "m", args.tolerance))
    else:
        for name, target in FIGURE14_PAPER.items():
            rows.append(row(f"Fig. 14 {name}", None, target, "m", args.tolerance))

    width = max(len(entry["item"]) for entry in rows)
    header = f"{'item'.ljust(width)}  {'measured':>10}  {'paper':>7}  {'gap':>9}  {'verdict':>12}"
    print(header)
    print("-" * len(header))
    for entry in rows:
        measured = "n/a" if entry["measured"] is None else f"{entry['measured']:.2f}"
        gap = "n/a" if entry.get("relative_percent") is None else f"{entry['relative_percent']:+.1f}%"
        print(f"{entry['item'].ljust(width)}  {measured:>10}  {entry['target']:7.1f}  {gap:>9}  {entry['verdict']:>12}")
    aligned = sum(1 for entry in rows if entry["verdict"] == "ok")
    print(f"\n{aligned}/{len(rows)} anchors within +/-{args.tolerance:.0f}%")
    if args.output:
        Path(args.output).write_text(json.dumps({"tolerance_percent": args.tolerance, "rows": rows},
                                                indent=2), encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
