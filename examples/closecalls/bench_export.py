#!/usr/bin/env python3
"""Export one recorded engine benchmark run for the ranking review bench.

No provider calls, ranking changes, generated metrics, or evaluation labels.
"""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

MAX_SAFE_INTEGER = 2**53 - 1


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, minimum=0):
    return type(value) is int and minimum <= value <= MAX_SAFE_INTEGER


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def load(path):
    return json.loads(Path(path).read_text())


def receipt(path):
    data = Path(path).read_bytes()
    return {"name": Path(path).name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def select_run(results, fixture, case, seed, mode):
    require(isinstance(results, list), "results must be an engine benchmark run array")
    matches = [r for r in results if (r.get("case"), r.get("seed"), r.get("mode")) == (case, seed, mode)]
    require(len(matches) == 1, "case/seed/mode must identify exactly one recorded run")
    cases = [c for c in fixture["cases"] if c.get("name") == case]
    require(len(cases) == 1, "case must identify exactly one fixture case")
    documents = cases[0]["documents"]
    require(isinstance(documents, list) and documents, "fixture requires original documents")
    require(all(isinstance(d.get("id"), str) and d["id"] and isinstance(d.get("value"), str)
                for d in documents), "each fixture document requires its original id and value")
    require(len({d["id"] for d in documents}) == len(documents), "duplicate source ids")
    ranking = matches[0]["ranking"]
    require(isinstance(ranking, list), "recorded ranking must be an array")
    seen = set()
    for row in ranking:
        index = row.get("input_index")
        require(integer(index) and index < len(documents), "invalid recorded input_index")
        require(index not in seen, "duplicate ranked source position")
        seen.add(index)
        original = documents[index]
        require(row.get("key") == original["id"] and row.get("value") == original["value"],
                "ranked key/value do not match the original source position")
        require("document" in row and row["document"] is None,
                "this engine-fixture adapter requires recorded document=null")
        require(finite(row.get("score")), "recorded score is required")
        require(finite(row.get("exposure")) and 0 <= row["exposure"] <= 1,
                "recorded exposure must be between zero and one")
        require(integer(row.get("rank"), 1) and integer(row.get("rounds")),
                "recorded rank and rounds are required")
    return ranking, documents, cases[0].get("criterion"), matches[0].get("run_id")


def read_comparisons(path, documents, run_id=None):
    """Select original receipt lines by recorded run id; never turn them into snapshots."""
    selected = []
    opener = gzip.open if Path(path).suffix == ".gz" else open
    with opener(path, "rb") as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if run_id is not None and record.get("run_id") != run_id:
                continue
            require(isinstance(record.get("candidates"), list), f"receipt {number}: missing candidates")
            for candidate in record["candidates"]:
                index = candidate.get("input_index")
                require(integer(index) and index < len(documents), f"receipt {number}: invalid source index")
                original = documents[index]
                require(candidate.get("source_id") == original["id"] and candidate.get("value") == original["value"],
                        f"receipt {number}: source identity differs from fixture")
            selected.append(line)
    require(selected, "no comparison receipts match the selected run")
    return b"".join(selected), len(selected)


def check_trace(path, documents):
    sources = {d["id"]: d["value"] for d in documents}
    count = 0
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        snapshot = json.loads(line)
        for field in ("round", "trial", "trials_completed", "trials_remaining",
                      "total_input_tokens", "total_output_tokens"):
            require(integer(snapshot.get(field)), f"trace {number}: missing or invalid {field}")
        for field in ("elbow_position", "stable_trials_count"):
            require(field not in snapshot or integer(snapshot[field]), f"trace {number}: invalid {field}")
        require(isinstance(snapshot.get("rankings"), list), f"trace {number}: missing rankings")
        for row in snapshot["rankings"]:
            require(row.get("id") in sources and row.get("value") == sources[row["id"]]
                    and finite(row.get("score")), f"trace {number}: invalid ranked source or score")
        count += 1
    require(count > 0, "trial trace is empty")
    return count


def export_run(results_path, fixture_path, output, *, case, seed, mode, comparisons=None, trace=None):
    output = Path(output)
    require(not output.exists(), "output directory already exists; choose a new path")
    ranking, documents, criterion, run_id = select_run(load(results_path), load(fixture_path), case, seed, mode)
    sources = {"results": receipt(results_path), "fixture": receipt(fixture_path)}
    sidecars = {}
    if comparisons:
        content, count = read_comparisons(comparisons, documents, run_id)
        sidecars["comparisons.jsonl"] = content
        sources["comparisons"] = {**receipt(comparisons), "selected_records": count,
                                  "selected_sha256": hashlib.sha256(content).hexdigest(),
                                  "run_id": run_id}
    if trace:
        sidecars["trace.jsonl"] = Path(trace).read_bytes()
        sources["trace"] = {**receipt(trace), "snapshots": check_trace(trace, documents)}
    manifest = {
        "format": "siftrank.engine-review-export.v1",
        "selection": {"case": case, "seed": seed, "mode": mode, "run_id": run_id},
        "criterion": criterion,
        "sources": sources,
        "coverage": {
            "fixture_candidates": len(documents), "ranked_candidates": len(ranking),
            "unranked_input_indices": sorted(set(range(len(documents))) - {r["input_index"] for r in ranking}),
            "upstream_records_before_fixture_selection": None,
        },
        "identities": [{"input_index": i, "key": d["id"]} for i, d in enumerate(documents)],
        "semantics": {
            "results": "exact recorded ranking rows; score, exposure and rounds remain separate",
            "input": "original fixture value strings in source order; fixture relevance labels excluded",
            "source_match": "exporter verifies key/value/index; bench treats null document as a position match",
            "comparisons": "separate original receipts, not supported by the bench trial viewer",
            "trace": "recorded snapshots only; absent when unavailable",
            "sidecar_run_identity": "comparisons select recorded run_id when available; otherwise caller selects matching receipts. trace run attribution is caller supplied",
        },
    }
    payloads = {
        "results.json": ranking,
        "input.json": [d["value"] for d in documents],
        "manifest.json": manifest,
    }
    output.mkdir(parents=True)
    for name, value in payloads.items():
        (output / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    for name, data in sidecars.items():
        (output / name).write_bytes(data)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--comparisons", type=Path, help="recorded JSONL or JSONL.gz comparisons; select recorded run_id when present")
    parser.add_argument("--trace", type=Path, help="matching actual trial snapshots, never inferred from comparisons")
    args = parser.parse_args()
    try:
        manifest = export_run(args.results, args.fixture, args.output, case=args.case, seed=args.seed,
                              mode=args.mode, comparisons=args.comparisons, trace=args.trace)
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(1, f"bench export: {error}\n")
    print(json.dumps(manifest["coverage"], indent=2))


if __name__ == "__main__":
    main()
