#!/usr/bin/env python3
"""Verify the bundled historical captures against their wire responses, offline."""

import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_lines(path):
    return [json.loads(line) for line in gzip.decompress(path.read_bytes()).splitlines()]


def verify(directory):
    provenance = json.loads((directory / "provenance.json").read_text())
    for name, metadata in provenance["files"].items():
        data = (directory / name).read_bytes()
        require(len(data) == metadata["bytes"], f"size mismatch: {name}")
        require(hashlib.sha256(data).hexdigest() == metadata["sha256"], f"hash mismatch: {name}")
        if "uncompressed_sha256" in metadata:
            raw = gzip.decompress(data)
            require(len(raw) == metadata["uncompressed_bytes"], f"uncompressed size: {name}")
            require(hashlib.sha256(raw).hexdigest() == metadata["uncompressed_sha256"],
                    f"uncompressed hash: {name}")
            require(len(raw.splitlines()) == metadata["records"], f"record count: {name}")
    captures = read_lines(directory / "captures.jsonl.gz")
    wires = read_lines(directory / "wire.jsonl.gz")
    cases = json.loads((directory / "labels.json").read_text())["cases"]
    require(len(captures) == len(wires), "capture/wire count mismatch")
    counts, runs, identities, indices = Counter(), set(), {}, {}
    for capture, wire in zip(captures, wires):
        key = tuple(capture[k] for k in ("case_id", "run_id", "receipt_index"))
        require(key == tuple(wire[k] for k in ("case_id", "run_id", "receipt_index")),
                "capture/wire identity mismatch")
        require(capture["receipt_index"] == indices.get(key[:2], 0) + 1, "receipt order")
        indices[key[:2]] = capture["receipt_index"]
        request, response = json.loads(wire["request_body"]), json.loads(wire["response_body"])
        require(wire["sent"] and wire["status"] == 200 and not wire["error"], "failed wire call")
        require(capture["status"] == "accepted" and wire["mode"] == "pairwise", "wrong arm")
        require(capture["model"] == wire["model"] == response["model"] == provenance["model"],
                "model mismatch")
        case = cases[capture["case_id"]]
        require(capture["criterion"] == case["criterion"], "criterion mismatch")
        candidates = capture["candidates"]
        require(request["state"] == [{"id": c["id"], "value": c["value"]} for c in candidates],
                "request state differs from candidates or includes extra fields")
        require(len({c["id"] for c in candidates}) == len(candidates), "duplicate local ID")
        wins = {c["id"]: 0.0 for c in candidates}
        for candidate in candidates:
            source = capture["case_id"], candidate["source_id"]
            value = candidate["value"], candidate["input_index"]
            require(candidate["source_id"] in case["grades"], "unknown source ID")
            require(source not in identities or identities[source] == value, "source changed")
            identities[source] = value
        size = len(candidates) * (len(candidates) - 1) // 2
        require(len(capture["pairs"]) == len(request["questions"]) == len(response["answers"]) == size,
                "incomplete pair matrix")
        offset = 0
        for i, first in enumerate(candidates):
            for j in range(i + 1, len(candidates)):
                second, answer = candidates[j], response["answers"][f"pair_{i}_{j}"]
                p = answer["noul"]
                require(answer["type"] == "noul" and math.isfinite(p) and 0 <= p <= 1,
                        "invalid probability")
                require(capture["pairs"][offset] == {"first_id": first["id"],
                        "second_id": second["id"], "probability": p}, "pair/wire mismatch")
                require(case["criterion"] in request["questions"][f"pair_{i}_{j}"]["instructions"],
                        "question criterion mismatch")
                offset += 1
                wins[first["id"]] += p
                wins[second["id"]] += 1 - p
                a, b = (case["grades"][c["source_id"]] for c in (first, second))
                counts["pair_observations"] += 1
                counts["wrong_decisive_unequal_grade_preferences"] += a != b and p != .5 and ((p > .5) != (a > b))
                counts["exact_probability_ties"] += p == .5
                counts["equal_grade_observations"] += a == b
        require(capture["ordering"] == sorted(wins, key=lambda k: -wins[k]) ==
                json.loads(capture["response"])["docs"], "ordering mismatch")
        for field, token in (("InputTokens", "input_tokens"), ("OutputTokens", "output_tokens")):
            require(capture["usage"][field] == wire[token] == response["usage"][token], "usage mismatch")
        runs.add(key[:2])
    counts.update(cases=len(cases), runs=len(runs), receipts=len(captures), wire_calls=len(wires),
                  source_candidates=len(identities))
    require(all(value == provenance["counts"][key] for key, value in counts.items()), "count mismatch")
    require(Counter(c["status"] for c in captures) == provenance["counts"]["statuses"], "status mismatch")
    return {"verified": True, **counts}


if __name__ == "__main__":
    print(json.dumps(verify(Path(__file__).with_name("data")), indent=2))
