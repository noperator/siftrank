#!/usr/bin/env python3
"""Experimental, offline test of whether close preferences help prioritize review."""

import argparse
import gzip
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def read_observations(captures, labels):
    cases = labels["cases"]
    counts, rows = Counter(), []
    opener = gzip.open if str(captures).endswith(".gz") else open
    with opener(captures, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            receipt = json.loads(line)
            counts["receipts"] += 1
            if receipt["status"] not in ("accepted", "ok"):
                counts["failed_receipts"] += 1
                continue
            case_id, run_id = receipt["case_id"], receipt["run_id"]
            grades = cases[case_id]["grades"]
            candidates = receipt["candidates"]
            by_id = {c["id"]: c for c in candidates}
            if len(by_id) != len(candidates):
                raise ValueError(f"duplicate candidate ids on line {line_number}")
            pairs = receipt.get("pairs", [])
            if len(pairs) != len(candidates) * (len(candidates) - 1) // 2:
                raise ValueError(f"incomplete comparison matrix on line {line_number}")
            seen = set()
            for pair in pairs:
                a, b = pair["first_id"], pair["second_id"]
                key = tuple(sorted((a, b)))
                p = pair["probability"]
                if (a == b or a not in by_id or b not in by_id or key in seen
                        or isinstance(p, bool) or not isinstance(p, (int, float))
                        or not math.isfinite(p) or not 0 <= p <= 1):
                    raise ValueError(f"invalid comparison on line {line_number}")
                seen.add(key)
                source_ids = [by_id[x].get("source_id", x) for x in (a, b)]
                values = [grades.get(x) for x in source_ids]
                judged = all(v is not None for v in values)
                if judged and any(isinstance(v, bool) or not isinstance(v, (int, float))
                                  or not math.isfinite(v) for v in values):
                    raise ValueError(f"invalid grade on line {line_number}")
                wrong = judged and values[0] != values[1] and p != 0.5 and (
                    (p > 0.5) != (values[0] > values[1]))
                counts["observations"] += 1
                counts["unjudged"] += not judged
                counts["equal_grade"] += judged and values[0] == values[1]
                counts["exact_model_ties"] += p == 0.5
                counts["known_errors"] += bool(wrong)
                counts["strong_known_errors"] += bool(wrong and (p <= 0.35 or p >= 0.65))
                rows.append({"case_id": case_id, "run_id": run_id,
                             "line": line_number, "first_id": a, "second_id": b,
                             "source_ids": source_ids, "probability": p,
                             # Round only the selection margin to make complementary
                             # decimal probabilities tie despite floating point noise.
                             "margin": round(abs(p - 0.5), 12), "wrong": bool(wrong)})
    if not rows:
        raise ValueError("no accepted pair observations")
    return rows, dict(counts)


def expected_errors(rows, budget):
    """Spend the full budget; average random ordering within equal-margin groups."""
    remaining, found = min(budget, len(rows)), 0.0
    # Labels do not participate in ordering or budget allocation. Equal-grade,
    # unjudged and exact-model-tie observations still consume review capacity.
    for _, group in itertools.groupby(sorted(rows, key=lambda r: r["margin"]),
                                      key=lambda r: r["margin"]):
        group = list(group)
        take = min(remaining, len(group))
        found += take * sum(r["wrong"] for r in group) / len(group)
        remaining -= take
        if remaining == 0:
            break
    return found


def point(runs, fraction=None, budget=None):
    total = sum(len(rows) for rows in runs.values())
    errors = sum(r["wrong"] for rows in runs.values() for r in rows)
    reviewed, found, random_found = 0, 0.0, 0.0
    for rows in runs.values():
        count = min(budget, len(rows)) if budget is not None else math.floor(len(rows) * fraction)
        reviewed += count
        found += expected_errors(rows, count)
        random_found += count * sum(r["wrong"] for r in rows) / len(rows)
    return {"reviewed": reviewed, "review_fraction": reviewed / total,
            "known_errors_found": found, "random_expected_errors": random_found,
            "error_recall": found / errors if errors else None,
            "random_error_recall": random_found / errors if errors else None}


def analyze(rows, counts):
    runs, cases = defaultdict(list), defaultdict(list)
    for row in rows:
        runs[(row["case_id"], row["run_id"])].append(row)
        cases[row["case_id"]].append(row)
    by_case = {}
    for case_id, case_rows in sorted(cases.items()):
        case_runs = {key: values for key, values in runs.items() if key[0] == case_id}
        by_case[case_id] = {"observations": len(case_rows),
                            "known_errors": sum(r["wrong"] for r in case_rows),
                            "at_25_percent": point(case_runs, fraction=0.25)}
    return {"schema_version": 1, "counts": counts, "runs": len(runs),
            "policy": "closest to 0.5 first; same budget per run; random tie-break expectation",
            "budget_unit": "pair observations, including equal-grade, unjudged and exact ties",
            "curve": [point(runs, fraction=i / 100) for i in range(101)],
            "at_25_percent": point(runs, fraction=0.25),
            "at_20_per_run": point(runs, budget=20), "by_case": by_case,
            "limitations": ["retrospective; labels score the review policy after selection",
                            "repeated pairs and seeds are correlated observations",
                            "pair errors are not distinct findings or final-ranking errors",
                            "fixed pair count does not establish equal human reading time",
                            "unknown labels remain in the budget; only known errors are scored"]}


def plot(report, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    if not report["counts"].get("known_errors"):
        raise ValueError("no known errors to plot")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, ax = plt.subplots(figsize=(9, 6), dpi=180)
    curve = report["curve"]
    ax.plot([p["review_fraction"] for p in curve], [p["error_recall"] for p in curve],
            color="#1767a6", linewidth=2.7, label="closest preferences first")
    ax.plot([p["review_fraction"] for p in curve], [p["random_error_recall"] for p in curve],
            color="#818994", linestyle="--", linewidth=1.8, label="random review (expected)")
    at = report["at_25_percent"]
    ax.scatter([at["review_fraction"]], [at["error_recall"]], color="#1767a6", zorder=3)
    ax.annotate(f"{at['review_fraction']:.0%} reviewed → {at['error_recall']:.1%} of errors\n"
                f"random: {at['random_error_recall']:.1%}",
                (at["review_fraction"], at["error_recall"]), xytext=(0.39, 0.19),
                textcoords="axes fraction", arrowprops={"arrowstyle": "-", "color": "#1767a6"})
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="comparison observations reviewed",
           ylabel="known wrong preferences found",
           title="experimental: do close calls help find mistakes?")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.14)
    ax.legend(loc="upper left", frameon=False)
    counts = report["counts"]
    fig.text(0.12, 0.025,
             f"{len(report['by_case'])} tasks · {report['runs']} runs · "
             f"{counts['observations']:,} pair observations · {counts['known_errors']:,} known errors\n"
             "retrospective synthetic benchmark; all pairs consume budget; repeated observations are correlated",
             fontsize=8.5, color="#555e69")
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    fig.savefig(destination, metadata={"Software": "matplotlib"})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    if args.output.exists() or (args.plot and args.plot.exists()):
        parser.error("use new output paths")
    rows, counts = read_observations(args.captures, json.loads(args.labels.read_text()))
    report = analyze(rows, counts)
    if args.plot:
        plot(report, args.plot)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"counts": counts, "at_25_percent": report["at_25_percent"],
                      "at_20_per_run": report["at_20_per_run"]}, indent=2))


if __name__ == "__main__":
    main()
