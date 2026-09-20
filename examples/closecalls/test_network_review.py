import copy
import json
import tempfile
import unittest
from pathlib import Path

import network_review as replay


def receipt(first, second, probability):
    return {"candidates": [{"id": "local-a", "source_id": first},
                           {"id": "local-b", "source_id": second}],
            "pairs": [{"first_id": "local-a", "second_id": "local-b",
                       "probability": probability}]}


class NetworkReviewTest(unittest.TestCase):
    def test_reversed_orientation_and_duplicate_aggregation(self):
        pairs = replay.aggregate_pairs([receipt("a", "b", .25), receipt("b", "a", .25),
                                        receipt("a", "b", .5)])
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["n"], 3)
        self.assertEqual(pairs[0]["mean"], .5)
        self.assertEqual((pairs[0]["minimum"], pairs[0]["maximum"]), (.25, .75))
        self.assertTrue(pairs[0]["strong_conflict"])

    def test_equal_grade_and_unknown_pairs_spend_budget_without_label_leakage(self):
        pairs = replay.aggregate_pairs([receipt("a", "b", .5), receipt("c", "d", .51),
                                        receipt("e", "f", .6)])
        queues = replay.review_queues(pairs)
        before = copy.deepcopy(queues)
        labels = {key: {"grade": 0, "known": True} for key in "abcdef"}
        labels["c"]["known"] = False
        labels["f"]["grade"] = 1
        scored = replay.score_queues(pairs, queues, list("abcdef"), labels, 3)
        curve = {p["budget"]: p for p in scored["curves"]
                 if p["policy"] == "low_confidence_mean"}
        self.assertEqual(curve[1]["errors_found"], 0)  # equal-grade consumes first review
        self.assertEqual(curve[2]["unknown_reviews"], 1)
        self.assertEqual(curve[2]["errors_found"], 0)  # unknown consumes second review
        self.assertEqual(curve[3]["errors_found"], 1)
        labels["b"]["grade"] = 10
        replay.score_queues(pairs, queues, list("abcdef"), labels, 3)
        self.assertEqual(queues, before)
        self.assertEqual(replay.review_queues(pairs), before)

    def test_error_is_final_order_not_individual_preference(self):
        pairs = replay.aggregate_pairs([receipt("a", "b", .9)])
        labels = {"a": {"grade": 1, "known": True}, "b": {"grade": 0, "known": True}}
        good = replay.score_queues(pairs, replay.review_queues(pairs), ["a", "b"], labels)
        bad = replay.score_queues(pairs, replay.review_queues(pairs), ["b", "a"], labels)
        self.assertEqual(good["wrong_ordered_observed_pairs"], 0)
        self.assertEqual(bad["wrong_ordered_observed_pairs"], 1)

    def test_artifact_tampering_fails_before_analysis(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory / "input.json").write_bytes(b"original")
            (directory / "provenance.json").write_text(json.dumps({"files": {
                "input.json": {"bytes": 8, "sha256": replay.digest(b"original")}}}))
            replay.verify_files(directory)
            (directory / "input.json").write_bytes(b"modified")
            with self.assertRaisesRegex(ValueError, "artifact hash/size mismatch"):
                replay.verify_files(directory)

    def test_wire_tampering_fails_even_when_receipt_is_unchanged(self):
        directory = Path(__file__).with_name("network")
        capture = replay.read_lines(directory / "captures.jsonl.gz")[0]
        wire = replay.read_lines(directory / "wire.jsonl.gz")[0]
        case = json.loads((directory / "input.json").read_text())["cases"][0]
        replay.verify_receipt(capture, wire, case)
        response = json.loads(wire["response_body"])
        response["answers"]["pair_0_1"]["noul"] = .123456
        wire["response_body"] = json.dumps(response)
        with self.assertRaisesRegex(ValueError, "pair/wire mismatch"):
            replay.verify_receipt(capture, wire, case)

    def test_complete_bundled_replay_matches_published_curve(self):
        directory = Path(__file__).with_name("network")
        report = replay.analyze(*replay.load_verified(directory))
        expected = json.loads((directory / "results.json").read_text())
        self.assertEqual(report, expected)
        self.assertEqual(report["counts"]["runs"], 6)
        self.assertEqual(report["counts"]["pair_observations"], 13062)
        self.assertAlmostEqual(report["at_20_per_run"]["low_confidence_mean"], 13 / 6)
        self.assertAlmostEqual(report["at_20_per_run"]["random_expected"], .7633129833318438)


if __name__ == "__main__":
    unittest.main()
