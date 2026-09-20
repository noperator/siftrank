import gzip
import json
import tempfile
import unittest
from pathlib import Path

from bench_export import check_trace, export_run, read_comparisons, select_run


class BenchExportTest(unittest.TestCase):
    def setUp(self):
        self.documents = [{"id": "a", "value": "first", "relevance": 0},
                          {"id": "b", "value": "second", "relevance": 3}]
        self.row = {"key": "b", "value": "second", "document": None, "score": 0.02,
                    "exposure": 0.5, "rank": 1, "rounds": 3, "input_index": 1}
        self.fixture = {"cases": [{"name": "case", "criterion": "review", "documents": self.documents}]}
        self.results = [{"case": "case", "seed": 1, "mode": "pairwise", "ranking": [self.row]}]

    def test_source_order_labels_metrics_and_missing_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "source-results.json").write_text(json.dumps(self.results))
            (root / "fixture.json").write_text(json.dumps(self.fixture))
            manifest = export_run(root / "source-results.json", root / "fixture.json", root / "export",
                                  case="case", seed=1, mode="pairwise")
            self.assertEqual(json.loads((root / "export/results.json").read_text()), [self.row])
            self.assertEqual(json.loads((root / "export/input.json").read_text()), ["first", "second"])
            self.assertEqual(manifest["coverage"]["unranked_input_indices"], [0])
            self.assertIsNone(manifest["coverage"]["upstream_records_before_fixture_selection"])
            self.assertFalse((root / "export/trace.jsonl").exists())
            self.assertNotIn("relevance", (root / "export/input.json").read_text())
            with self.assertRaisesRegex(ValueError, "already exists"):
                export_run(root / "source-results.json", root / "fixture.json", root / "export",
                           case="case", seed=1, mode="pairwise")

    def test_rejects_identity_mismatch_missing_metrics_and_ambiguous_run(self):
        for field, value in [("key", "wrong"), ("value", "changed"), ("input_index", 0),
                             ("exposure", None), ("rounds", None), ("score", float("nan"))]:
            with self.subTest(field=field):
                results = [{**self.results[0], "ranking": [{**self.row, field: value}]}]
                with self.assertRaises(ValueError):
                    select_run(results, self.fixture, "case", 1, "pairwise")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            select_run(self.results * 2, self.fixture, "case", 1, "pairwise")

    def test_comparison_receipts_preserve_bytes_and_check_source_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "results.json").write_text(json.dumps(self.results))
            (root / "fixture.json").write_text(json.dumps(self.fixture))
            raw = '{"candidates":[{"source_id":"b","input_index":1,"value":"second"}]}\n'
            (root / "pairs.jsonl").write_text(raw)
            export_run(root / "results.json", root / "fixture.json", root / "export",
                       case="case", seed=1, mode="pairwise", comparisons=root / "pairs.jsonl")
            self.assertEqual((root / "export/comparisons.jsonl").read_text(), raw)
            (root / "pairs.jsonl").write_text(raw.replace('"second"', '"different"'))
            with self.assertRaisesRegex(ValueError, "identity differs"):
                export_run(root / "results.json", root / "fixture.json", root / "bad-export",
                           case="case", seed=1, mode="pairwise", comparisons=root / "pairs.jsonl")
            self.assertFalse((root / "bad-export").exists())

    def test_shared_gzip_receipts_are_selected_by_recorded_run_id(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "captures.jsonl.gz"
            wrong = b'{"run_id":"other","candidates":[]}\n'
            right = b'{"run_id":"chosen","candidates":[{"source_id":"b","input_index":1,"value":"second"}]}\n'
            with gzip.open(path, "wb") as file:
                file.write(wrong + right)
            self.assertEqual(read_comparisons(path, self.documents, "chosen"), (right, 1))
            with self.assertRaisesRegex(ValueError, "no comparison"):
                read_comparisons(path, self.documents, "absent")

    def test_trial_snapshots_must_exist_and_are_not_pair_receipts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            snapshot = {"round": 1, "trial": 1, "trials_completed": 1, "trials_remaining": 2,
                        "total_input_tokens": 20, "total_output_tokens": 10,
                        "rankings": [{"id": "b", "value": "second", "score": 0.2}]}
            path.write_text(json.dumps(snapshot) + "\n")
            self.assertEqual(check_trace(path, self.documents), 1)
            path.write_text('{"candidates": []}\n')
            with self.assertRaisesRegex(ValueError, "round"):
                check_trace(path, self.documents)


if __name__ == "__main__":
    unittest.main()
