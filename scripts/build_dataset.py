"""Export reviewed Alpaca splits and matched B/C training manifests."""

import argparse
from collections import Counter
import hashlib
from heapq import nlargest
import json
from pathlib import Path
import random
import re
from tempfile import TemporaryDirectory

from build_pilot import build_pair, build_pairs, load_rows, select_row


SPLITS = ("train", "validation", "test")
CONTEXTS = ("direct", "untrusted")
ROLES = ("source_row", "carrier_row")
PILOT_SELECTION = Path(__file__).resolve().parents[1] / "data" / "pilot_selection.json"


def split_membership(rows, splits, pilot_pairs):
    membership = {}
    normalized_rows = {}
    for split, pairs in (("pilot", pilot_pairs), *splits.items()):
        sources = set()
        for pair in pairs:
            source = pair["source_row"]
            if source in sources:
                raise ValueError(f"Source row {source} is repeated in {split}")
            sources.add(source)
            for role in ROLES:
                row_id = pair[role]
                instruction, data = select_row(rows, row_id, needs_input=role == "carrier_row")
                previous = membership.setdefault(row_id, split)
                if previous != split:
                    raise ValueError(f"Row {row_id} is assigned to both {previous} and {split}")
                key = tuple(" ".join(field.casefold().split()) for field in (instruction, data))
                other_split, other_row = normalized_rows.setdefault(key, (split, row_id))
                if other_split != split:
                    raise ValueError(f"Rows {other_row} ({other_split}) and {row_id} ({split}) are duplicates")
    return membership


def overlap_candidates(rows, membership):
    tokens = {}
    for row_id in membership:
        row = rows[row_id]
        text = f"{row['instruction']} {row['input']}".casefold()
        tokens[row_id] = set(re.findall(r"\w+", text))
    earlier = sorted(row_id for row_id, split in membership.items() if split in ("train", "validation"))
    for row_id in sorted(membership):
        if membership[row_id] != "test":
            continue
        matches = []
        for other_id in earlier:
            shared = len(tokens[row_id] & tokens[other_id])
            if shared:
                score = shared / (len(tokens[row_id]) + len(tokens[other_id]) - shared)
                matches.append({"row": other_id, "split": membership[other_id], "similarity": score})
        yield {"test_row": row_id, "matches": nlargest(3, matches, key=lambda match: match["similarity"])}


def training_arms(pairs, seed):
    rng = random.Random(seed)
    direct, untrusted = CONTEXTS
    contexts_b = [direct] * len(pairs) + [untrusted] * len(pairs)
    rng.shuffle(contexts_b)
    carriers = {pair["carrier_row"]: pair["clean_carrier"] for pair in pairs}
    clean_records = [
        {
            "id": f"train-clean-{carrier_row}",
            "pair_id": None,
            "split": "train",
            "source_row": None,
            "carrier_row": carrier_row,
            "context": "clean_carrier",
            **prompt,
        }
        for carrier_row, prompt in carriers.items()
    ]
    order = list(range(len(contexts_b) + len(clean_records)))
    rng.shuffle(order)
    for arm, contexts in (("B", contexts_b), ("C", CONTEXTS * len(pairs))):
        records = []
        for index in order:
            if index >= len(contexts):
                records.append(clean_records[index - len(contexts)])
                continue
            pair = pairs[index // 2]
            context = contexts[index]
            slot = index % 2 + 1
            records.append({
                "id": f"{pair['id']}-{slot}",
                "pair_id": pair["id"],
                "split": "train",
                "source_row": pair["source_row"],
                "carrier_row": pair["carrier_row"],
                "context": context,
                **pair[context],
            })
        yield arm, records


def summarize_arm(records):
    per_source = {}
    for record in records:
        if record["source_row"] is not None:
            per_source.setdefault(record["source_row"], set()).add(record["context"])
    both = sum(len(contexts) == 2 for contexts in per_source.values())
    return {
        "examples": len(records),
        "context_counts": dict(Counter(record["context"] for record in records)),
        "sources_in_both_contexts": both,
        "repeated_same_context_slots": len(per_source) - both,
    }


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def untrusted_positions(splits, seed):
    rng = random.Random(seed)
    positions = {}
    for split in SPLITS:
        count = len(splits[split])
        order = ["before"] * (count // 2) + ["after"] * (count - count // 2)
        rng.shuffle(order)
        positions[split] = order
    return positions


def export_dataset(rows, digest, selection_raw, output, seed):
    selection = json.loads(selection_raw)
    pilot = json.loads(PILOT_SELECTION.read_text(encoding="utf-8"))
    revision = selection["alpaca_revision"]
    if selection["alpaca_sha256"] != digest or revision != pilot["alpaca_revision"]:
        raise ValueError("Use the recorded Alpaca revision and checksum for the pilot and main splits")
    pilot_pairs = list(build_pairs(rows, pilot, digest))

    entries = selection["splits"]
    if not isinstance(entries, dict) or set(entries) != set(SPLITS):
        raise ValueError("Provide train, validation, and test splits")
    for split in SPLITS:
        if not isinstance(entries[split], list) or not entries[split]:
            raise ValueError(f"Provide a nonempty list of reviewed pairs for {split}")
    membership = split_membership(rows, entries, pilot_pairs)
    positions = untrusted_positions(entries, seed)
    splits = {
        split: [
            {**build_pair(rows, entry, split, revision, digest, position=position, require_review=False),
             "embedded_position": position}
            for entry, position in zip(entries[split], positions[split], strict=True)
        ]
        for split in SPLITS
    }
    manifest = {
        "alpaca_revision": revision,
        "alpaca_sha256": digest,
        "selection_sha256": hashlib.sha256(selection_raw).hexdigest(),
        "seed": seed,
        "splits": {
            split: {
                "pairs": len(pairs),
                "embedded_positions": dict(Counter(positions[split])),
                "carrier_uses": dict(sorted(Counter(pair["carrier_row"] for pair in pairs).items())),
            }
            for split, pairs in splits.items()
        },
        "arms": {},
    }
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temp:
        staging = Path(temp) / "dataset"
        staging.mkdir()
        for split, pairs in splits.items():
            write_jsonl(staging / f"{split}_pairs.jsonl", pairs)
        for arm, records in training_arms(splits["train"], seed):
            write_jsonl(staging / f"train_{arm}.jsonl", records)
            manifest["arms"][arm] = summarize_arm(records)
        write_jsonl(staging / "test_overlap_review.jsonl", overlap_candidates(rows, membership))
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        staging.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("alpaca", type=Path, help="Path to the pinned alpaca_data.json")
    parser.add_argument("--selection", type=Path, required=True, help="Reviewed split selection in JSON")
    parser.add_argument("--output", type=Path, required=True, help="New directory for the dataset manifests")
    parser.add_argument("--seed", type=int, default=577, help="Seed for embedded position, context assignment, and slot order")
    args = parser.parse_args()
    try:
        rows, digest = load_rows(args.alpaca)
        export_dataset(rows, digest, args.selection.read_bytes(), args.output, args.seed)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Dataset error: {exc}\n")
    print(f"Wrote train, validation, and test manifests to {args.output}")


if __name__ == "__main__":
    main()
