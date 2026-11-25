from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from propose_pairs import split_reviewed_pool


class SplitReviewedTests(unittest.TestCase):
    def setUp(self):
        self.entries = [{"row": row_id, "review": f"Approved row {row_id}"} for row_id in range(6)]
        self.counts = {"train": 3, "validation": 1, "test": 2}

    def test_keeps_groups_in_train_and_preserves_all_rows(self):
        baseline = split_reviewed_pool({"rows": self.entries}, self.counts, 577, set())
        grouped_rows = {baseline["train"][0]["row"], baseline["test"][0]["row"]}
        splits = split_reviewed_pool({"rows": self.entries}, self.counts, 577, grouped_rows)

        self.assertEqual({split: len(entries) for split, entries in splits.items()}, self.counts)
        self.assertTrue(grouped_rows <= {entry["row"] for entry in splits["train"]})
        self.assertCountEqual([entry for entries in splits.values() for entry in entries], self.entries)
        self.assertEqual(
            splits,
            split_reviewed_pool({"rows": self.entries[::-1]}, self.counts, 577, grouped_rows),
        )

    def test_rejects_more_grouped_rows_than_training_places(self):
        with self.assertRaisesRegex(ValueError, "Too many grouped rows"):
            split_reviewed_pool({"rows": self.entries}, self.counts, 577, {0, 1, 2, 3})
