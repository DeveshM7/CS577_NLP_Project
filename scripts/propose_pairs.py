"""Select separate Alpaca pools, then pair their rows for review."""

import argparse
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory

from build_dataset import PILOT_SELECTION, SPLITS, split_membership
from build_pilot import build_pairs, load_rows
from validate_pool import normalized_text, validate_pool


PREFERRED_ANSWER_WORDS = 12


def candidate_pools(rows, excluded):
    sources, carriers = [], []
    for row_id, row in enumerate(rows):
        if row_id in excluded:
            continue
        task, data, answer = (row[field].strip() for field in ("instruction", "input", "output"))
        if task and answer:
            pool = carriers if data else sources
            pool.append(row_id)
    return sources, carriers


def select_pool(rows, candidates, counts, rng, metadata):
    if len(candidates) < sum(counts.values()):
        raise ValueError(
            f"Requested {sum(counts.values())} rows from {len(candidates)} candidates"
        )
    rng.shuffle(candidates)
    candidates.sort(key=lambda row_id: len(rows[row_id]["output"].split()) > PREFERRED_ANSWER_WORDS)
    splits = {}
    start = 0
    for split in SPLITS:
        end = start + counts[split]
        splits[split] = [{"row": row_id, **rows[row_id]} for row_id in candidates[start:end]]
        start = end
    return {**metadata, "candidate_count": len(candidates), "splits": splits}


def save_pools(rows, digest, pilot, pilot_pairs, counts, seed, output):
    excluded = split_membership(rows, {}, pilot_pairs)
    sources, carriers = candidate_pools(rows, excluded)
    metadata = {
        "alpaca_revision": pilot["alpaca_revision"],
        "alpaca_sha256": digest,
        "pool_seed": seed,
        "preferred_answer_words": PREFERRED_ANSWER_WORDS,
    }
    carrier_counts = {**counts, "train": (counts["train"] + 4) // 5}
    rng = random.Random(seed)
    pools = {
        "sources": select_pool(rows, sources, counts, rng, metadata),
        "carriers": select_pool(rows, carriers, carrier_counts, rng, metadata),
    }
    write_pools(output, pools)


def split_reviewed_pool(pool, counts, seed, train_rows):
    entries = sorted(pool["rows"], key=lambda entry: entry["row"])
    random.Random(seed).shuffle(entries)
    splits = {}
    start = 0
    for split in SPLITS:
        end = start + counts[split]
        splits[split] = entries[start:end]
        start = end

    train = splits["train"]
    available = [index for index in reversed(range(len(train))) if train[index]["row"] not in train_rows]
    for split in ("validation", "test"):
        for index, entry in enumerate(splits[split]):
            if entry["row"] in train_rows:
                if not available:
                    raise ValueError("Too many grouped rows for the training split")
                train_index = available.pop()
                train[train_index], splits[split][index] = entry, train[train_index]
    return splits


def save_reviewed_splits(rows, digest, pilot, sources, carriers, counts, seed, train_rows, output):
    role_counts = {"sources": counts, "carriers": {**counts, "train": (counts["train"] + 4) // 5}}
    validate_pool(rows, digest, pilot, sources, "source", sum(counts.values()))
    validate_pool(rows, digest, pilot, carriers, "carrier", sum(role_counts["carriers"].values()))
    entries = sources["rows"] + carriers["rows"]
    if len({entry["row"] for entry in entries}) != len(entries):
        raise ValueError("Reviewed pools contain a repeated Alpaca row ID")
    if len({normalized_text(entry) for entry in entries}) != len(entries):
        raise ValueError("Reviewed pools contain a repeated instruction and input")
    if not train_rows <= {entry["row"] for entry in entries}:
        raise ValueError("Training groups must contain only approved row IDs")

    metadata = {
        "alpaca_revision": pilot["alpaca_revision"],
        "alpaca_sha256": digest,
        "pool_seed": seed,
    }
    pools = {}
    for name, pool in (("sources", sources), ("carriers", carriers)):
        pools[name] = {
            **metadata,
            "candidate_count": len(pool["rows"]),
            "splits": split_reviewed_pool(pool, role_counts[name], seed, train_rows),
        }
    write_pools(output, pools)


def write_pools(output, pools):
    if output.exists():
        raise FileExistsError(f"Pool directory already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temp:
        staging = Path(temp) / "pools"
        staging.mkdir()
        for name, pool in pools.items():
            write_json(staging / f"{name}.json", pool)
        staging.rename(output)


def pair_pools(rows, digest, pilot, pilot_pairs, directory, seed):
    sources, carriers = [
        json.loads((directory / f"{name}.json").read_text(encoding="utf-8"))
        for name in ("sources", "carriers")
    ]
    for pool in (sources, carriers):
        if pool["alpaca_sha256"] != digest or pool["alpaca_revision"] != pilot["alpaca_revision"]:
            raise ValueError("Use pools from the recorded Alpaca revision and checksum")
        if set(pool["splits"]) != set(SPLITS):
            raise ValueError("Each pool must contain train, validation, and test splits")

    rng = random.Random(seed)
    splits = {}
    for split in SPLITS:
        split_sources = [item["row"] for item in sources["splits"][split]]
        split_carriers = [item["row"] for item in carriers["splits"][split]]
        if not 0 < len(split_carriers) <= len(split_sources):
            raise ValueError(f"{split} needs at least one carrier and a source for every carrier")
        if len(set(split_carriers)) != len(split_carriers):
            raise ValueError(f"{split} carrier IDs must be distinct")
        rng.shuffle(split_carriers)
        pairs = []
        for index, source_id in enumerate(split_sources):
            carrier_id = split_carriers[index % len(split_carriers)]
            source, carrier = rows[source_id], rows[carrier_id]
            pairs.append({
                "source_row": source_id,
                "carrier_row": carrier_id,
                "source_task": source["instruction"].strip(),
                "carrier_task": carrier["instruction"].strip(),
                "carrier_data": carrier["input"].strip(),
                "direct_answer": source["output"].strip(),
                "carrier_answer": carrier["output"].strip(),
                "review": "",
            })
        splits[split] = pairs
    split_membership(rows, splits, pilot_pairs)
    return {
        "alpaca_revision": pilot["alpaca_revision"],
        "alpaca_sha256": digest,
        "proposal_seed": seed,
        "pool_seeds": {"sources": sources["pool_seed"], "carriers": carriers["pool_seed"]},
        "candidate_pool_sizes": {
            "sources": sources["candidate_count"], "carriers": carriers["candidate_count"],
        },
        "splits": splits,
    }


def write_json(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("alpaca", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    select = commands.add_parser("pools", help="Select and save standalone and carrier rows")
    select.add_argument("--output", type=Path, required=True, help="New pool directory")
    select.add_argument("--train", type=int, default=30)
    select.add_argument("--validation", type=int, default=10)
    select.add_argument("--test", type=int, default=10)
    select.add_argument("--seed", type=int, default=577)
    reviewed = commands.add_parser("split-reviewed", help="Assign approved rows to train, validation, and test")
    reviewed.add_argument("--sources", type=Path, required=True, help="Reviewed standalone pool")
    reviewed.add_argument("--carriers", type=Path, required=True, help="Reviewed input-dependent pool")
    reviewed.add_argument("--train-groups", type=Path, help="JSON list of reviewed groups with rows to keep in train")
    reviewed.add_argument("--output", type=Path, required=True, help="New directory for the split pools")
    reviewed.add_argument("--train", type=int, default=1000)
    reviewed.add_argument("--validation", type=int, default=100)
    reviewed.add_argument("--test", type=int, default=200)
    reviewed.add_argument("--seed", type=int, default=577)
    pair = commands.add_parser("pair", help="Randomly match rows from the saved pools")
    pair.add_argument("--pools", type=Path, required=True, help="Directory containing the saved pools")
    pair.add_argument("--output", type=Path, required=True, help="New draft selection JSON file")
    pair.add_argument("--seed", type=int, default=577)
    args = parser.parse_args()

    try:
        rows, digest = load_rows(args.alpaca)
        pilot = json.loads(PILOT_SELECTION.read_text(encoding="utf-8"))
        if args.command in ("pools", "split-reviewed"):
            counts = {split: getattr(args, split) for split in SPLITS}
            if min(counts.values()) < 1:
                parser.error("Each split must have at least one source")
            if args.command == "pools":
                pilot_pairs = list(build_pairs(rows, pilot, digest))
                save_pools(rows, digest, pilot, pilot_pairs, counts, args.seed, args.output)
            else:
                sources = json.loads(args.sources.read_text(encoding="utf-8"))
                carriers = json.loads(args.carriers.read_text(encoding="utf-8"))
                groups = json.loads(args.train_groups.read_text(encoding="utf-8")) if args.train_groups else []
                train_rows = {row_id for group in groups for row_id in group["rows"]}
                if any(type(row_id) is not int for row_id in train_rows):
                    raise ValueError("Training groups must contain integer row IDs")
                save_reviewed_splits(rows, digest, pilot, sources, carriers, counts, args.seed, train_rows, args.output)
            print(f"Wrote sources.json and carriers.json to {args.output}")
        else:
            pilot_pairs = list(build_pairs(rows, pilot, digest))
            selection = pair_pools(rows, digest, pilot, pilot_pairs, args.pools, args.seed)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.output, selection)
            count = sum(len(pairs) for pairs in selection["splits"].values())
            print(f"Wrote {count} proposed pairs to {args.output}")
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        parser.exit(1, f"Dataset error: {exc}\n")


if __name__ == "__main__":
    main()
