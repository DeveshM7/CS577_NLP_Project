"""Check an unsplit, reviewed Alpaca pool against the pinned source data."""

import argparse
from itertools import combinations
import json
from pathlib import Path
import re

from build_dataset import PILOT_SELECTION
from build_pilot import build_pairs, load_rows, select_row


FIELDS = ("instruction", "input", "output")
OVERLAP_THRESHOLD = 0.55


def normalized_text(row):
    return tuple(" ".join(row[field].casefold().split()) for field in FIELDS[:2])


def validate_pool(rows, digest, pilot, pool, role, count):
    if pool["alpaca_revision"] != pilot["alpaca_revision"] or pool["alpaca_sha256"] != digest:
        raise ValueError("Pool provenance does not match the pinned Alpaca file")
    pilot_ids = {
        pair[key]
        for pair in build_pairs(rows, pilot, digest)
        for key in ("source_row", "carrier_row")
    }
    entries = pool["rows"]
    if len(entries) != count:
        raise ValueError(f"Expected {count} reviewed rows, found {len(entries)}")

    seen_ids = set(pilot_ids)
    seen_text = {normalized_text(rows[row_id]): row_id for row_id in pilot_ids}
    for entry in entries:
        row_id = entry["row"]
        select_row(rows, row_id, needs_input=role == "carrier")
        if row_id in seen_ids:
            if row_id in pilot_ids:
                raise ValueError(f"Row {row_id} belongs to the pilot")
            raise ValueError(f"Repeated Alpaca row ID {row_id}")
        if any(entry[field] != rows[row_id][field] for field in FIELDS):
            raise ValueError(f"Text or original answer changed for Alpaca row {row_id}")
        if not isinstance(entry["review"], str) or not entry["review"].strip():
            raise ValueError(f"Row {row_id} needs an approval note")
        key = normalized_text(entry)
        if key in seen_text:
            raise ValueError(f"Rows {seen_text[key]} and {row_id} have equivalent instruction and input")
        seen_ids.add(row_id)
        seen_text[key] = row_id
    return pilot_ids


def overlap_score(first, second):
    shared = len(first & second)
    return shared / (len(first) + len(second) - shared) if shared else 0


def near_duplicates(rows, accepted_ids, pilot_ids):
    features = {}
    for row_id in sorted(accepted_ids | pilot_ids):
        task, data = normalized_text(rows[row_id])
        words = re.findall(r"\w+", f"{task} {data}")
        text = " ".join(words)
        features[row_id] = (set(words), {text[i:i + 3] for i in range(len(text) - 2)})
    for first_id, second_id in combinations(features, 2):
        if first_id in pilot_ids and second_id in pilot_ids:
            continue
        first_words, first_trigrams = features[first_id]
        second_words, second_trigrams = features[second_id]
        word_overlap = overlap_score(first_words, second_words)
        character_overlap = overlap_score(first_trigrams, second_trigrams)
        if max(word_overlap, character_overlap) >= OVERLAP_THRESHOLD:
            yield {
                "first_row": first_id,
                "second_row": second_id,
                "word_overlap": round(word_overlap, 3),
                "character_overlap": round(character_overlap, 3),
                "includes_pilot": first_id in pilot_ids or second_id in pilot_ids,
            }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("alpaca", type=Path)
    parser.add_argument("pool", type=Path)
    parser.add_argument("--role", required=True, choices=("source", "carrier"))
    parser.add_argument("--count", required=True, type=int)
    parser.add_argument("--near-duplicate-report", type=Path)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    try:
        rows, digest = load_rows(args.alpaca)
        pilot = json.loads(PILOT_SELECTION.read_text(encoding="utf-8"))
        pool = json.loads(args.pool.read_text(encoding="utf-8"))
        pilot_ids = validate_pool(rows, digest, pilot, pool, args.role, args.count)
        print(f"Validated {args.count} reviewed {args.role} rows")
        if args.near_duplicate_report:
            accepted_ids = {entry["row"] for entry in pool["rows"]}
            matches = list(near_duplicates(rows, accepted_ids, pilot_ids))
            with args.near_duplicate_report.open("x", encoding="utf-8") as stream:
                json.dump(matches, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            print(f"Flagged {len(matches)} possible near-duplicates for review")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Pool error: {exc}\n")


if __name__ == "__main__":
    main()
