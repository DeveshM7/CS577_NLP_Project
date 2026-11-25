"""Export an audited Alpaca selection as direct, clean, and paired prompts."""

import argparse
import hashlib
import json
from pathlib import Path


def load_rows(path):
    raw = path.read_bytes()
    rows = json.loads(raw)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Alpaca data must be a nonempty JSON array")
    return rows, hashlib.sha256(raw).hexdigest()


def select_row(rows, index, *, needs_input):
    if type(index) is not int or not 0 <= index < len(rows):
        raise ValueError(f"Invalid Alpaca row index: {index!r}")
    row = rows[index]
    fields = ("instruction", "input", "output")
    if not isinstance(row, dict) or any(not isinstance(row.get(key), str) for key in fields):
        raise ValueError(f"Row {index} needs string instruction, input, and output fields")
    instruction, data, answer = (row[key].strip() for key in fields)
    if not instruction or not answer or bool(data) != needs_input:
        expected = "nonempty" if needs_input else "empty"
        raise ValueError(f"Row {index} needs a task, an answer, and {expected} input")
    return instruction, data


def prompt(task, data, answer):
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Each reviewed reference answer must be a nonempty string")
    return {"task": task, "data": data, "expected_answer": answer.strip()}


def untrusted_prompt(task, data, source, answer, *, position=None):
    if position is None:
        return prompt(task, f"{data}\n\n{source}", answer)
    if position == "before":
        data = f"{source}\n{data}"
    elif position == "after":
        data = f"{data}\n{source}"
    else:
        raise ValueError(f"Unknown embedded instruction position: {position!r}")
    return prompt(task, data, answer)


def build_pair(rows, entry, split, revision, digest, *, position=None, require_review=True):
    source_id, carrier_id = entry["source_row"], entry["carrier_row"]
    source, _ = select_row(rows, source_id, needs_input=False)
    task, data = select_row(rows, carrier_id, needs_input=True)
    review = entry.get("review", "")
    if not isinstance(review, str):
        raise ValueError(f"Review note for row {carrier_id} must be text")
    if require_review and not review.strip():
        raise ValueError(f"Document why the carrier answer remains valid for row {carrier_id}")
    return {
        "id": f"{split}-{source_id}-{carrier_id}",
        "split": split,
        "alpaca_revision": revision,
        "alpaca_sha256": digest,
        "source_row": source_id,
        "carrier_row": carrier_id,
        "review": review.strip(),
        "direct": prompt(source, "", entry["direct_answer"]),
        "clean_carrier": prompt(task, data, entry["carrier_answer"]),
        "untrusted": untrusted_prompt(task, data, source, entry["carrier_answer"], position=position),
    }


def build_pairs(rows, selection, digest):
    if selection["alpaca_sha256"] != digest:
        raise ValueError("Selection was reviewed against a different Alpaca file")
    revision = selection["alpaca_revision"]
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("Record the upstream Alpaca revision")
    entries = selection["pairs"]
    if not entries:
        raise ValueError("Select at least one audited pair")
    sources = {entry["source_row"] for entry in entries}
    carriers = {entry["carrier_row"] for entry in entries}
    if len(sources) != len(entries) or len(carriers) != len(entries) or sources & carriers:
        raise ValueError("Use distinct source and carrier rows for each pilot pair")

    for entry in entries:
        yield build_pair(rows, entry, "pilot", revision, digest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("alpaca", type=Path, help="Path to Stanford Alpaca's alpaca_data.json")
    parser.add_argument("--selection", type=Path, help="Reviewed pair selection in JSON")
    parser.add_argument("--output", type=Path, help="Destination JSONL file")
    parser.add_argument("--list", choices=("source", "carrier"), help="List candidate rows")
    parser.add_argument("--start", type=int, default=0, help="First row index to inspect")
    parser.add_argument("--limit", type=int, default=30, help="Maximum candidates to list")
    args = parser.parse_args()
    if args.list:
        if args.selection or args.output or args.start < 0 or args.limit < 1:
            parser.error("--list requires a nonnegative start and positive limit, with no export options")
    elif not args.selection or not args.output:
        parser.error("provide either --list or both --selection and --output")

    try:
        rows, digest = load_rows(args.alpaca)
        if args.list:
            print(json.dumps({"alpaca_sha256": digest}))
            shown = 0
            for index in range(args.start, len(rows)):
                try:
                    select_row(rows, index, needs_input=args.list == "carrier")
                except ValueError:
                    continue
                print(json.dumps({"row": index, **rows[index]}, ensure_ascii=False))
                shown += 1
                if shown == args.limit:
                    break
            return

        selection = json.loads(args.selection.read_text(encoding="utf-8"))
        records = list(build_pairs(rows, selection, digest))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as output:
            for record in records:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Wrote {len(records)} reviewed pilot pairs to {args.output}")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Dataset error: {exc}\n")


if __name__ == "__main__":
    main()
