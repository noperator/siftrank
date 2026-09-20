#!/usr/bin/env python3
"""Plot the two recorded experiments separately; their review units differ."""

import argparse
import json
from pathlib import Path


def plot(network, synthetic, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})
    fig, axes = plt.subplots(1, 2, figsize=(15, 8), dpi=180)
    policies = {
        "low_confidence_mean": ("closest mean preference", "#1767a6"),
        "low_confidence_single": ("closest single observation", "#8860ad"),
        "conflict_first": ("strong reversals first", "#c77b30"),
        "random_expected": ("random review (expected)", "#858b95"),
    }
    left, right = axes
    for policy, (label, color) in policies.items():
        points = [p for p in network["curve"] if p["policy"] == policy and p["budget"] <= 50]
        left.plot([p["budget"] for p in points], [p["mean_errors_found"] for p in points],
                  label=label, color=color, linewidth=2.6 if policy == "low_confidence_mean" else 1.8,
                  linestyle="--" if policy == "random_expected" else "-")
    left.set(xlim=(0, 50), ylim=(0, 4.5), xlabel="unique pair reviews per run",
             ylabel="wrong final orderings found (mean per run)",
             title="captured traffic · 3 captures / 6 runs")
    network_at20 = network["at_20_per_run"]
    network_found = network_at20["low_confidence_mean"]
    left.scatter([20], [network_found], color="#1767a6", zorder=3)
    left.annotate(f"20 reviews: {network_found:.2f} errors\n"
                  f"random: {network_at20['random_expected']:.2f}",
                  (20, network_found), xytext=(26, 0.55),
                  fontsize=10.5, arrowprops={"arrowstyle": "-", "color": "#1767a6"})
    left.legend(loc="upper left", frameon=False, fontsize=9.5)

    curve = synthetic["curve"]
    right.plot([p["review_fraction"] for p in curve], [p["error_recall"] for p in curve],
               color="#1767a6", linewidth=2.6, label="closest preferences first")
    right.plot([p["review_fraction"] for p in curve], [p["random_error_recall"] for p in curve],
               color="#858b95", linewidth=1.8, linestyle="--", label="random review (expected)")
    right.set(xlim=(0, 1), ylim=(0, 1), xlabel="pair observations reviewed",
              ylabel="known wrong preferences found",
              title="synthetic scenarios · 6 tasks / 18 runs")
    right.xaxis.set_major_formatter(PercentFormatter(1))
    right.yaxis.set_major_formatter(PercentFormatter(1))
    point = synthetic["at_25_percent"]
    right.scatter([point["review_fraction"]], [point["error_recall"]], color="#1767a6", zorder=3)
    right.annotate(f"{point['review_fraction']:.1%} reviewed: {point['error_recall']:.1%} of errors\n"
                   f"random: {point['random_error_recall']:.1%}",
                   (point["review_fraction"], point["error_recall"]), xytext=(.40, .12),
                   fontsize=10.5, arrowprops={"arrowstyle": "-", "color": "#1767a6"})
    right.legend(loc="upper left", frameon=False, fontsize=10)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=.13)
    fig.suptitle("experimental blue team results\non a mix of captured traffic and synthetic traces",
                 y=.98, fontsize=20)
    fig.text(.10, .12, "320 selected host/time windows from controlled captures (2011).\n"
             "known normal/botnet cohort; 67 positive windows lost in selection.",
             fontsize=9.5, color="#555e69")
    synthetic_at20 = synthetic["at_20_per_run"]
    fig.text(.57, .12, "144 hand-written items; 7,020 repeated pair observations.\n"
             f"at 20 reviews/run: {synthetic_at20['error_recall']:.1%} of errors vs "
             f"{synthetic_at20['random_error_recall']:.1%} random.",
             fontsize=9.5, color="#555e69")
    fig.text(.5, .035, "different review units; results are not pooled. equal-grade pairs consume budget.\n"
             "exploratory retrospective checks; repeated runs are correlated; no measured analyst or causal-reasoning outcome.",
             ha="center", fontsize=9.5, color="#555e69")
    fig.subplots_adjust(left=.065, right=.985, top=.80, bottom=.24, wspace=.24)
    fig.savefig(output, metadata={"Software": "matplotlib"})
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", type=Path, default=Path(__file__).parent / "network/results.json")
    parser.add_argument("--synthetic", type=Path, default=Path(__file__).parent / "results.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("use a new output path")
    plot(json.loads(args.network.read_text()), json.loads(args.synthetic.read_text()), args.output)
