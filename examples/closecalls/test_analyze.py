import json
import tempfile
import unittest
from pathlib import Path

from analyze import analyze, expected_errors, read_observations


class ReviewTests(unittest.TestCase):
    def test_budget_is_spent_before_judgments(self):
        # Close exact ties and same-grade pairs consume the first two slots.
        rows = [{"case_id": "c", "run_id": "r", "margin": m, "wrong": w}
                for m, w in [(0, False), (0.01, False), (0.1, True), (0.4, True)]]
        self.assertEqual(expected_errors(rows, 2), 0)
        self.assertEqual(expected_errors(rows, 3), 1)
        report = analyze(rows, {"observations": 4, "known_errors": 2})
        self.assertEqual(report["curve"][50]["error_recall"], 0)
        self.assertEqual(report["curve"][50]["random_error_recall"], 0.5)
        self.assertEqual(report["curve"][100]["error_recall"], 1)

    def test_equal_margins_use_expected_random_tie_break(self):
        rows = [{"margin": 0.1, "wrong": wrong} for wrong in [True, False, False, True]]
        self.assertEqual(expected_errors(rows, 1), 0.5)
        self.assertEqual(expected_errors(list(reversed(rows)), 1), 0.5)

    def test_unequal_run_sizes_keep_separate_budgets(self):
        rows = [{"case_id": "c", "run_id": str(run), "margin": 0.1, "wrong": True}
                for run, count in [(1, 3), (2, 5)] for _ in range(count)]
        point = analyze(rows, {})["curve"][50]
        self.assertEqual(point["reviewed"], 3)  # floor(3/2) + floor(5/2)
        self.assertEqual(point["error_recall"], 3 / 8)

    def read_fixture(self, receipts, labels):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "captures.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in receipts))
            return read_observations(path, {"cases": {"c": {"grades": labels}}})

    def receipt(self, p=0.1):
        return {"status": "accepted", "run_id": "r", "case_id": "c",
                "candidates": [{"id": "temporary-a", "source_id": "original-a"},
                               {"id": "b"}],
                "pairs": [{"first_id": "temporary-a", "second_id": "b", "probability": p}]}

    def test_original_ids_orientation_and_unjudged_are_preserved(self):
        receipt = self.receipt()
        rows, counts = self.read_fixture([receipt], {"original-a": 2, "b": 1})
        self.assertTrue(rows[0]["wrong"])
        self.assertEqual(counts["strong_known_errors"], 1)
        rows, counts = self.read_fixture([receipt], {"original-a": 2})
        self.assertEqual(len(rows), 1)
        self.assertEqual(counts["unjudged"], 1)
        self.assertFalse(rows[0]["wrong"])

    def test_exact_and_grade_ties_are_charged_but_not_called_errors(self):
        for p, grades, counter in [(0.5, {"original-a": 2, "b": 1}, "exact_model_ties"),
                                   (0.9, {"original-a": 1, "b": 1}, "equal_grade")]:
            rows, counts = self.read_fixture([self.receipt(p)], grades)
            self.assertEqual(counts[counter], 1)
            self.assertEqual(counts["observations"], 1)
            self.assertFalse(rows[0]["wrong"])

    def test_bad_probabilities_and_partial_or_duplicate_pairs_fail(self):
        for p in [float("nan"), -0.1, 1.1, True, "0.5"]:
            with self.assertRaises(ValueError):
                self.read_fixture([self.receipt(p)], {})
        receipt = self.receipt()
        receipt["pairs"] = []
        with self.assertRaises(ValueError):
            self.read_fixture([receipt], {})

    def test_failures_count_separately_and_no_error_recall_is_undefined(self):
        failure = {"status": "error", "error": "timeout"}
        rows, counts = self.read_fixture([failure, self.receipt()], {"original-a": 0, "b": 1})
        self.assertEqual(counts["failed_receipts"], 1)
        self.assertIsNone(analyze(rows, counts)["at_25_percent"]["error_recall"])


if __name__ == "__main__":
    unittest.main()
