from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import validate_pool


def row(instruction, data="", answer="answer"):
    return {"instruction": instruction, "input": data, "output": answer}


class PoolValidationTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            row("Pilot question"),
            row("Pilot carrier", "example"),
            row("What is the capital of Spain?", answer="Madrid"),
            row("  WHAT is the capital of SPAIN? ", answer="Madrid"),
            row("What is the capital of France?", answer="Paris"),
            row("Find the city", "I live in Boston.", "Boston"),
        ]
        self.pilot = {
            "alpaca_revision": "revision",
            "alpaca_sha256": "digest",
            "pairs": [{
                "source_row": 0,
                "carrier_row": 1,
                "direct_answer": "answer",
                "carrier_answer": "answer",
                "review": "The added question leaves the carrier answer intact.",
            }],
        }

    def pool(self, *row_ids):
        return {
            "alpaca_revision": "revision",
            "alpaca_sha256": "digest",
            "rows": [{"row": row_id, **self.rows[row_id], "review": "Checked."} for row_id in row_ids],
        }

    def validate(self, pool, count, role="source"):
        return validate_pool.validate_pool(self.rows, "digest", self.pilot, pool, role, count)

    def test_accepts_valid_source_and_carrier_pools(self):
        self.assertEqual(self.validate(self.pool(2, 4), 2), {0, 1})
        self.assertEqual(self.validate(self.pool(5), 1, role="carrier"), {0, 1})

    def test_rejects_wrong_count_and_provenance(self):
        with self.assertRaisesRegex(ValueError, "Expected 2"):
            self.validate(self.pool(2), 2)
        pool = self.pool(2)
        pool["alpaca_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "provenance"):
            self.validate(pool, 1)

    def test_rejects_reused_ids_and_normalized_duplicates(self):
        with self.assertRaisesRegex(ValueError, "Repeated Alpaca row ID 2"):
            self.validate(self.pool(2, 2), 2)
        with self.assertRaisesRegex(ValueError, "equivalent instruction and input"):
            self.validate(self.pool(2, 3), 2)
        with self.assertRaisesRegex(ValueError, "belongs to the pilot"):
            self.validate(self.pool(0), 1)

    def test_rejects_edited_answer_unreviewed_row_and_wrong_role(self):
        pool = self.pool(2)
        pool["rows"][0]["output"] = "Paris"
        with self.assertRaisesRegex(ValueError, "original answer changed"):
            self.validate(pool, 1)
        pool = self.pool(2)
        pool["rows"][0]["review"] = " "
        with self.assertRaisesRegex(ValueError, "needs an approval note"):
            self.validate(pool, 1)
        with self.assertRaisesRegex(ValueError, "empty input"):
            self.validate(self.pool(5), 1)

    def test_near_duplicate_report_flags_similar_rows(self):
        matches = list(validate_pool.near_duplicates(self.rows, {2, 4}, {0, 1}))
        pairs = {(match["first_row"], match["second_row"]) for match in matches}
        self.assertIn((2, 4), pairs)
        self.assertNotIn((0, 1), pairs)
