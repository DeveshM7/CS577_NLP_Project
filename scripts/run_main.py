"""Train and evaluate on the frozen dataset exports."""

import argparse
from collections import Counter
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, set_seed

from run_pilot import (
    CONTEXTS, MAX_NEW_TOKENS, MODEL, REVISION, SEED,
    encode_example, infer, train, training_batches,
)


EPOCHS = 2
EFFECTIVE_BATCH_SIZE = 32


def read_records(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def check_split(records, split):
    if not records or any(record["split"] != split for record in records):
        raise ValueError(f"Expected a nonempty {split} export")


def smoke_examples(records):
    check_split(records, "train")
    by_context = {context: [] for context in CONTEXTS}
    for record in records:
        by_context[record["context"]].append(record)
    if any(len(examples) < 2 for examples in by_context.values()):
        raise ValueError("Expected at least two records per context")
    return [example for group in zip(*(by_context[c][:2] for c in CONTEXTS), strict=True)
            for example in group]


def check_training(records, tokenizer):
    check_split(records, "train")
    context_counts = Counter(record["context"] for record in records)
    if context_counts != {"direct": 1000, "untrusted": 1000, "clean_carrier": 200}:
        raise ValueError(f"Unexpected frozen training counts: {context_counts}")
    if len({record["id"] for record in records}) != len(records):
        raise ValueError("Repeated training record ID")
    encoded = [encode_example(tokenizer, record) for record in records]
    longest = max(len(ids) for ids, _ in encoded)
    config = AutoConfig.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    if longest > config.max_position_embeddings:
        raise ValueError(f"Training sequence length {longest} exceeds the model limit")
    return encoded, {
        "context_counts": dict(context_counts),
        "max_sequence_tokens": longest,
        "input_tokens_per_epoch": sum(len(ids) for ids, _ in encoded),
        "supervised_tokens_per_epoch": sum(len(ids) - prompt_length for ids, prompt_length in encoded),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "train", "smoke", "reload", "test"))
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--arm", choices=("A", "B", "C"))
    parser.add_argument("--adapter", type=Path)
    args = parser.parse_args()
    if args.mode != "check" and args.output is None:
        parser.error("output is required for this mode")
    if args.mode == "test":
        if args.arm is None or ((args.adapter is None) != (args.arm == "A")):
            parser.error("test requires --arm and an adapter for B or C")
    elif args.arm is not None or args.adapter is not None:
        parser.error("--arm and --adapter are for test mode")
    records = read_records(args.dataset)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    if args.mode == "smoke":
        examples = smoke_examples(records)
    elif args.mode in ("check", "train"):
        encoded, training_data = check_training(records, tokenizer)
        if args.mode == "check":
            batches = list(training_batches(
                range(len(records)), EPOCHS, EFFECTIVE_BATCH_SIZE, shuffle=True, seed=args.seed,
            ))
            for epoch in range(1, EPOCHS + 1):
                indices = [index for batch_epoch, batch in batches if batch_epoch == epoch for index in batch]
                if sorted(indices) != list(range(len(records))):
                    raise RuntimeError(f"Epoch {epoch} does not cover every training record exactly once")
            print(json.dumps({
                **training_data,
                "seed": args.seed,
                "optimizer_updates": len(batches),
                "batch_sizes": dict(Counter(len(batch) for _, batch in batches)),
            }, indent=2))
            return
    elif args.mode == "reload":
        check_split(records, "validation")
    else:
        check_split(records, "test")
        if len({pair["id"] for pair in records}) != len(records):
            raise ValueError("Test pair IDs must be distinct")

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A CUDA GPU with BF16 support is required")
    set_seed(args.seed)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, dtype=torch.bfloat16, local_files_only=True,
    ).to("cuda")
    torch.cuda.reset_peak_memory_stats()
    summary = {
        "model": MODEL, "revision": REVISION, "seed": args.seed, "dtype": "bfloat16",
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
    }
    if args.mode == "smoke":
        summary.update(train(model, tokenizer, examples, args.output, seed=args.seed))
    elif args.mode == "train":
        summary.update(training_data)
        summary.update(train(
            model, tokenizer, records, args.output, encoded=encoded,
            epochs=EPOCHS, batch_size=EFFECTIVE_BATCH_SIZE, seed=args.seed,
        ))
    elif args.mode == "reload":
        adapter = PeftModel.from_pretrained(model, args.output / "adapter", is_trainable=False)
        predictions = args.output / "reload_predictions.jsonl"
        infer(adapter, tokenizer, records[:1], predictions)
        generated = read_records(predictions)
        if len(generated) != len(CONTEXTS) or any(not item["prediction"].strip() for item in generated):
            raise RuntimeError("Adapter reload did not yield a nonblank prediction for every context")
        summary["predictions"] = len(generated)
    else:
        if args.adapter is not None:
            model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
        predictions = args.output / "predictions.jsonl"
        infer(model, tokenizer, records, predictions)
        generated = read_records(predictions)
        expected = Counter((pair["id"], context) for pair in records for context in CONTEXTS)
        actual = Counter((record["id"], record["context"]) for record in generated)
        if actual != expected:
            raise RuntimeError("Prediction coverage does not match the test pairs and contexts")
        summary.update(
            arm=args.arm,
            adapter=str(args.adapter) if args.adapter is not None else None,
            predictions=len(generated),
            context_counts=dict(Counter(record["context"] for record in generated)),
            blank_predictions=sum(not record["prediction"].strip() for record in generated),
            predictions_at_token_limit=sum(record["generated_tokens"] == MAX_NEW_TOKENS for record in generated),
            do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS,
        )
    summary.update(
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
    )
    with (args.output / f"{args.mode}.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
