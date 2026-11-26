"""Export frozen Arm D training examples and embedded pairings."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory

from build_dataset import PILOT_SELECTION, training_arms, untrusted_positions, write_jsonl
from build_pilot import build_pair, load_rows, prompt, select_row
from validate_pool import near_duplicates, normalized_text


DATA = PILOT_SELECTION.parent


def read_json(path):
    return json.loads(path.read_bytes())


def verify_file(path, expected):
    if sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Checksum mismatch for {path}")


def paired_record(pair):
    return {
        "id": f"{pair['id']}-2",
        "pair_id": pair["id"],
        "split": "train",
        "source_row": pair["source_row"],
        "carrier_row": pair["carrier_row"],
        "context": "untrusted",
        **pair["untrusted"],
    }


def build_arm(rows, digest, selection, frozen, pilot, extension, seed):
    revision = frozen["alpaca_revision"]
    if any(item["alpaca_sha256"] != digest or item["alpaca_revision"] != revision
           for item in (selection, frozen, pilot, *extension)):
        raise ValueError("Use the pinned Alpaca revision for all data")
    splits = frozen["splits"]
    train = splits["train"]
    original = {entry["source_row"]: entry for entry in train}
    replacement_ids = {int(source): new for source, new in selection["embedded_replacements"].items()}
    new_direct = selection["new_direct_rows"]
    new_rows = set(new_direct) | set(replacement_ids.values())
    used = {
        pair[role]
        for group in (*splits.values(), pilot["pairs"], extension)
        for pair in group
        for role in ("source_row", "carrier_row")
    }
    if (len(original) != 1000 or len(new_direct) != 500 or len(replacement_ids) != 500
            or len(new_rows) != 1000 or new_rows & used or not replacement_ids.keys() <= original.keys()):
        raise ValueError("Expected 1,000 unused sources and 500 replacements of frozen training sources")
    occupied = {normalized_text(rows[row_id]) for row_id in used}
    for row_id in sorted(new_rows):
        select_row(rows, row_id, needs_input=False)
        key = normalized_text(rows[row_id])
        if key in occupied:
            raise ValueError(f"Source {row_id} duplicates a selected Alpaca row")
        occupied.add(key)
    duplicate = next(near_duplicates(rows, new_rows, used), None)
    if duplicate:
        raise ValueError(f"Resolve near-duplicate sources before export: {duplicate}")

    original_pairs, embedded_pairs = [], []
    position_counts = Counter()
    positions = untrusted_positions(splits, seed)["train"]
    for entry, position in zip(train, positions, strict=True):
        pair = build_pair(rows, entry, "train", revision, digest, position=position, require_review=False)
        original_pairs.append(pair)
        source = pair["source_row"]
        if source in replacement_ids:
            replacement = replacement_ids[source]
            pair = build_pair(
                rows, {**entry, "source_row": replacement, "direct_answer": rows[replacement]["output"], "review": ""},
                "train", revision, digest, position=position, require_review=False,
            )
            position_counts[position] += 1
        embedded_pairs.append({**pair, "embedded_position": position})

    if position_counts != {"before": 250, "after": 250}:
        raise ValueError("The new embedded requests must retain balanced placement")

    direct_records = []
    for source in [*replacement_ids, *new_direct]:
        task, data = select_row(rows, source, needs_input=False)
        answer = original[source]["direct_answer"] if source in original else rows[source]["output"]
        direct_records.append({
            "id": f"train-direct-{source}",
            "pair_id": None,
            "split": "train",
            "source_row": source,
            "carrier_row": None,
            "context": "direct",
            **prompt(task, data, answer),
        })
    _, arm_b = next(training_arms(original_pairs, seed))
    clean_records = [record for record in arm_b if record["context"] == "clean_carrier"]
    carrier_uses = Counter(pair["carrier_row"] for pair in embedded_pairs)
    if len(clean_records) != 200 or len(carrier_uses) != 200 or set(carrier_uses.values()) != {5}:
        raise ValueError("Arm D must reuse the 200 frozen training carriers five times each")

    records = [*direct_records, *(paired_record(pair) for pair in embedded_pairs), *clean_records]
    Random(seed).shuffle(records)
    return embedded_pairs, records


def export(alpaca, selection_path, output):
    frozen_path = DATA / "frozen" / "pairs.json"
    extension_path = DATA / "test_extension" / "test_pairs.jsonl"
    selection = read_json(selection_path)
    frozen_manifest = read_json(frozen_path.with_name("freeze_manifest.json"))
    extension_manifest = read_json(extension_path.with_name("freeze_manifest.json"))
    input_hashes = {
        "frozen_selection_sha256": (frozen_path, frozen_manifest["sha256"][frozen_path.name]),
        "pilot_selection_sha256": (PILOT_SELECTION, extension_manifest["pilot_selection_sha256"]),
        "test_extension_sha256": (extension_path, extension_manifest["sha256"][extension_path.name]),
    }
    for key, (path, checksum) in input_hashes.items():
        verify_file(path, checksum)
        if selection[key] != checksum:
            raise ValueError(f"Arm D was selected against a different version of {path}")
    seed = frozen_manifest["export_seed"]
    rows, digest = load_rows(alpaca)
    with extension_path.open(encoding="utf-8") as stream:
        extension = [json.loads(line) for line in stream]
    embedded_pairs, records = build_arm(
        rows, digest, selection, read_json(frozen_path), read_json(PILOT_SELECTION), extension, seed,
    )
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temporary:
        staging = Path(temporary) / "dataset"
        staging.mkdir()
        write_jsonl(staging / "train_D.jsonl", records)
        write_jsonl(staging / "train_D_embedded_pairs.jsonl", embedded_pairs)
        manifest = {
            "alpaca_sha256": digest,
            "selection_sha256": sha256(selection_path.read_bytes()).hexdigest(),
            "seed": seed,
            "context_counts": dict(Counter(record["context"] for record in records)),
            "embedded_positions": dict(Counter(pair["embedded_position"] for pair in embedded_pairs)),
            "distinct_sources": len({record["source_row"] for record in records if record["source_row"] is not None}),
            "sha256": {path.name: sha256(path.read_bytes()).hexdigest() for path in sorted(staging.iterdir())},
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        staging.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("alpaca", type=Path)
    parser.add_argument("--selection", type=Path, default=DATA / "diversity" / "selection.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        export(args.alpaca, args.selection, args.output)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Dataset error: {exc}\n")


if __name__ == "__main__":
    main()
