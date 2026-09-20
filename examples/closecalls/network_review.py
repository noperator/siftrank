#!/usr/bin/env python3
"""Replay unique-pair review of captured traffic, verifying bundled evidence first."""

import argparse
import gzip
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_lines(path):
    return [json.loads(line) for line in gzip.decompress(path.read_bytes()).splitlines()]


def verify_files(directory):
    provenance = json.loads((directory / "provenance.json").read_text())
    for name, metadata in provenance["files"].items():
        require(Path(name).name == name, "invalid artifact filename")
        data = (directory / name).read_bytes()
        require(len(data) == metadata["bytes"] and digest(data) == metadata["sha256"],
                f"artifact hash/size mismatch: {name}")
        if "uncompressed_sha256" in metadata:
            raw = gzip.decompress(data)
            require(digest(raw) == metadata["uncompressed_sha256"] and
                    len(raw) == metadata["uncompressed_bytes"] and
                    len(raw.splitlines()) == metadata["records"], f"gzip mismatch: {name}")
    return provenance


def verify_receipt(capture, wire, case):
    """Match every recorded preference to its original context and wire response."""
    require(all(capture[k] == wire[k] for k in ("case_id", "run_id", "receipt_index")),
            "receipt/wire identity mismatch")
    require(capture["status"] == "accepted" and wire["mode"] == "pairwise" and
            wire["sent"] and wire["status"] == 200 and not wire["error"], "failed/wrong arm")
    request = json.loads(wire["request_body"])
    response = json.loads(wire["response_body"])
    require(capture["model"] == wire["model"] == request["model"] == response["model"],
            "model mismatch")
    require(capture["criterion"] == case["criterion"], "criterion mismatch")
    candidates = capture["candidates"]
    require(len({c["id"] for c in candidates}) == len(candidates), "duplicate local ID")
    require(request["state"] == [{"id": c["id"], "value": c["value"]} for c in candidates],
            "request state differs from candidates or contains extra fields")
    context = json.dumps({"criterion": capture["criterion"], "candidates": candidates},
                         ensure_ascii=False, separators=(",", ":"))
    for char, escaped in (("<", r"\u003c"), (">", r"\u003e"), ("&", r"\u0026"),
                          ("\u2028", r"\u2028"), ("\u2029", r"\u2029")):
        context = context.replace(char, escaped)
    require(digest(context.encode()) == capture["context_sha256"], "context hash mismatch")
    for candidate in candidates:
        require(0 <= candidate["input_index"] < len(case["documents"]), "invalid input index")
        source = case["documents"][candidate["input_index"]]
        require(source == {"id": candidate["source_id"], "value": candidate["value"]},
                "candidate source identity/content mismatch")
    expected_count = len(candidates) * (len(candidates) - 1) // 2
    require(len(capture["pairs"]) == len(request["questions"]) ==
            len(response["answers"]) == expected_count, "incomplete pair matrix")
    wins = dict.fromkeys((c["id"] for c in candidates), 0.0)
    offset = 0
    for i, first in enumerate(candidates):
        for j in range(i + 1, len(candidates)):
            second = candidates[j]
            answer = response["answers"][f"pair_{i}_{j}"]
            p = answer["noul"]
            require(answer["type"] == "noul" and not isinstance(p, bool) and
                    isinstance(p, (int, float)) and math.isfinite(p) and 0 <= p <= 1,
                    "invalid probability")
            require(capture["pairs"][offset] == {"first_id": first["id"],
                    "second_id": second["id"], "probability": p}, "pair/wire mismatch")
            instructions = request["questions"][f"pair_{i}_{j}"]["instructions"]
            require(case["criterion"] in instructions and
                    f'candidate "{first["id"]}" rank ahead of candidate "{second["id"]}"'
                    in instructions, "question orientation/criterion mismatch")
            wins[first["id"]] += p
            wins[second["id"]] += 1 - p
            offset += 1
    require(capture["ordering"] == sorted(wins, key=lambda key: -wins[key]) ==
            json.loads(capture["response"])["docs"], "batch ordering mismatch")
    for field in ("input_tokens", "output_tokens"):
        original = {"input_tokens": "InputTokens", "output_tokens": "OutputTokens"}[field]
        require(capture["usage"][original] == wire[field] == response["usage"][field],
                "usage mismatch")


def load_verified(directory):
    provenance = verify_files(directory)
    cases = {c["name"]: c for c in json.loads((directory / "input.json").read_text())["cases"]}
    captures, wires = (read_lines(directory / name) for name in
                       ("captures.jsonl.gz", "wire.jsonl.gz"))
    require(len(captures) == len(wires) == provenance["counts"]["receipts"], "receipt count")
    by_run = defaultdict(list)
    for capture, wire in zip(captures, wires):
        require(capture["receipt_index"] == len(by_run[capture["run_id"]]) + 1,
                "receipt order mismatch")
        verify_receipt(capture, wire, cases[capture["case_id"]])
        by_run[capture["run_id"]].append(capture)
    rankings = json.loads((directory / "rankings.json").read_text())
    require(set(by_run) == {r["run_id"] for r in rankings} == set(provenance["runs"]),
            "run selection mismatch")
    require(len(rankings) == len(by_run), "duplicate ranking run")
    actual_counts = {"cases": len(cases), "runs": len(by_run), "receipts": len(captures),
                     "wire_calls": len(wires), "pair_observations": sum(len(c["pairs"]) for c in captures),
                     "source_candidates": sum(len(c["documents"]) for c in cases.values())}
    require(actual_counts == provenance["counts"], "provenance counts mismatch")
    for run in rankings:
        documents = cases[run["case"]]["documents"]
        require(len(documents) == len({d["id"] for d in documents}), "duplicate source ID")
        require(all(r["case_id"] == run["case"] for r in by_run[run["run_id"]]),
                "ranking/capture case mismatch")
        require(len(run["ranking"]) == len(documents) and
                {r["key"] for r in run["ranking"]} == {d["id"] for d in documents},
                "lost or duplicated ranked candidate")
        for index, row in enumerate(run["ranking"], 1):
            require(row["rank"] == index and documents[row["input_index"]] ==
                    {"id": row["key"], "value": row["value"]}, "ranked source mismatch")
    # Labels are opened only after the input/receipt/wire verification completes.
    labels = json.loads((directory / "labels.json").read_text())["cases"]
    for case_id, case in cases.items():
        require(set(labels[case_id]) == {d["id"] for d in case["documents"]}, "label identities")
        for label in labels[case_id].values():
            require(isinstance(label["known"], bool), "invalid known flag")
            grade = label["grade"]
            require(grade is None or (not isinstance(grade, bool) and isinstance(grade, (int, float))
                                     and math.isfinite(grade)), "invalid grade")
    return by_run, rankings, labels, provenance


def aggregate_pairs(receipts):
    """Collapse repeated observations without consulting labels or the final ordering."""
    pairs = {}
    for receipt in receipts:
        mapping = {c["id"]: c["source_id"] for c in receipt["candidates"]}
        for pair in receipt["pairs"]:
            a, b = mapping[pair["first_id"]], mapping[pair["second_id"]]
            p = pair["probability"]
            if a > b:
                a, b, p = b, a, 1 - p
            x = pairs.setdefault((a, b), {"first": a, "second": b, "n": 0, "sum": 0.0,
                                         "minimum": 1.0, "maximum": 0.0, "closest_margin": .5})
            x["n"] += 1
            x["sum"] += p
            x["minimum"], x["maximum"] = min(x["minimum"], p), max(x["maximum"], p)
            x["closest_margin"] = min(x["closest_margin"], abs(p - .5))
    for x in pairs.values():
        x["mean"] = x.pop("sum") / x["n"]
        x["mean_margin"] = abs(x["mean"] - .5)
        x["strong_conflict"] = x["minimum"] <= .35 and x["maximum"] >= .65
        x["conflict_strength"] = max(0, min(.5 - x["minimum"], x["maximum"] - .5))
    return list(pairs.values())


def review_queues(pairs):
    # Preserve the original hash ties and floating-point arithmetic for exact replay.
    def tie(x):
        return digest((x["first"] + "/" + x["second"]).encode())
    return {
        "low_confidence_mean": sorted(pairs, key=lambda x: (x["mean_margin"], tie(x))),
        "low_confidence_single": sorted(pairs, key=lambda x: (x["closest_margin"], tie(x))),
        "conflict_first": sorted(pairs, key=lambda x: (
            not x["strong_conflict"], -x["conflict_strength"] if x["strong_conflict"] else 0,
            x["mean_margin"], tie(x))),
    }


def score_queues(pairs, queues, order, labels, maximum_budget=100):
    ranks = {key: rank for rank, key in enumerate(order)}
    outcomes = {}
    for pair in pairs:
        a, b = pair["first"], pair["second"]
        ga, gb = labels.get(a, {}), labels.get(b, {})
        known = all(x.get("known", False) and x.get("grade") is not None for x in (ga, gb))
        wrong = bool(known and (ga["grade"] - gb["grade"]) * (ranks[a] - ranks[b]) > 0)
        outcomes[(a, b)] = (wrong, not known)
    curve = []
    for name, queue in queues.items():
        for budget in range(min(maximum_budget, len(queue)) + 1):
            values = [outcomes[(p["first"], p["second"])] for p in queue[:budget]]
            curve.append({"policy": name, "budget": budget,
                          "errors_found": sum(v[0] for v in values),
                          "unknown_reviews": sum(v[1] for v in values)})
    wrong = sum(v[0] for v in outcomes.values())
    unknown = sum(v[1] for v in outcomes.values())
    for budget in range(min(maximum_budget, len(pairs)) + 1):
        curve.append({"policy": "random_expected", "budget": budget,
                      "errors_found": budget * wrong / len(pairs) if pairs else 0,
                      "unknown_reviews": budget * unknown / len(pairs) if pairs else 0})
    return {"observed_pairs": len(pairs), "repeat_observed_pairs": sum(p["n"] > 1 for p in pairs),
            "strong_conflict_pairs": sum(p["strong_conflict"] for p in pairs),
            "wrong_ordered_observed_pairs": wrong, "curves": curve}


def analyze(by_run, rankings, labels, provenance):
    reports, points = [], defaultdict(list)
    for run in rankings:
        pairs = aggregate_pairs(by_run[run["run_id"]])
        queues = review_queues(pairs)
        scored = score_queues(pairs, queues, [r["key"] for r in run["ranking"]], labels[run["case"]])
        scored.update(run_id=run["run_id"], case_id=run["case"], seed=run["seed"])
        reports.append(scored)
        for point in scored["curves"]:
            points[point["policy"], point["budget"]].append(point["errors_found"])
    curve = [{"policy": policy, "budget": budget, "mean_errors_found": statistics.mean(values),
              "runs": len(values)} for (policy, budget), values in sorted(points.items())]
    return {"schema_version": 1, "verified": True, "counts": provenance["counts"],
            "selection": provenance["selection"], "curve": curve, "by_run": reports,
            "at_20_per_run": {p["policy"]: p["mean_errors_found"] for p in curve if p["budget"] == 20},
            "budget_unit": "unique observed pairs within each run, including equal-grade and unknown pairs",
            "error_definition": "final relative order contradicts unequal known publisher grades",
            "limitations": provenance["limitations"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).with_name("network"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = analyze(*load_verified(args.data))
    if args.output:
        with args.output.open("x") as output:
            output.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"verified": True, "counts": report["counts"],
                      "at_20_per_run": report["at_20_per_run"]}, indent=2))


if __name__ == "__main__":
    main()
