# CS 577 instruction boundary project

See the [spec](docs/PROJECT_SPEC.md), [implementation plan](docs/IMPLEMENTATION_PLAN.md), and [Scholar handoff](docs/HANDOFF.md).

We test whether seeing each training request in both roles, as an authorized instruction and as embedded data, improves paired accuracy on held-out requests compared with assigning those roles randomly at the same training-example budget. [Research question and hypothesis](docs/PROJECT_SPEC.md#research-question-and-hypothesis)

## Current status

The Scholar pilot and B/C training at both seeds are complete. A, B, and C were scored on the original 200 held-out pairs. C led B by 2.5 percentage points in paired accuracy at training seed 577 and 1.5 points at seed 578. Both arms followed embedded requests on 2 of 200 prompts at each seed. The proposed follow-up adds 400 fresh held-out pairs and uses the existing adapters. [Scholar handoff](docs/HANDOFF.md)

## Environment

Use Python 3.11 and uv. Run `./setup.sh` for data preparation or `./setup.sh gpu` for the Scholar GPU environment. Dataset prep uses only the Python standard library and also works with `python3.11` directly.

The Scholar pilot uses the Qwen2.5-3B-Instruct weights and tokenizer at revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`. Model files are outside Git.

## Alpaca pilot dataset

The initial pilot selection is in [data/pilot_selection.json](data/pilot_selection.json). It references Alpaca commit `761dc5bfbdeeffa89b8bff5d038781a4055f796a` and yields 20 pairs. Review mixed-language prompts and equivalent answer wording during the pilot.

Download [Stanford Alpaca's original JSON](https://github.com/tatsu-lab/stanford_alpaca#data-release) from a specific upstream commit into `data/raw/alpaca_data.json`. Record that commit with the curated selection. Alpaca data is available for noncommercial use under CC BY-NC 4.0.

List source candidates with empty inputs and carrier candidates with populated inputs:

```sh
python3.11 scripts/build_pilot.py data/raw/alpaca_data.json --list source
python3.11 scripts/build_pilot.py data/raw/alpaca_data.json --list carrier
```

Use `--start` and `--limit` to inspect additional rows. These commands screen by field shape. Reviewers still need to verify that the direct request has a clear answer, that the carrier depends on its input, and that the original carrier answer remains valid in the paired prompt.

Save the reviewed choices to `data/pilot_selection.json` using the following structure. Row numbers and values below are illustrative. Copy the SHA-256 printed by the listing command.

```json
{
  "alpaca_revision": "upstream commit",
  "alpaca_sha256": "digest from listing",
  "pairs": [
    {
      "source_row": 0,
      "carrier_row": 1,
      "direct_answer": "12",
      "carrier_answer": "Boston",
      "review": "The added arithmetic question names no city."
    }
  ]
}
```

Export the reviewed selection:

```sh
python3.11 scripts/build_pilot.py data/raw/alpaca_data.json \
  --selection data/pilot_selection.json --output artifacts/pilot.jsonl
```

Each JSONL row contains the direct prompt, the clean carrier, and the untrusted-context carrier, plus their expected answers and source indices. Keep pilot rows out of the final test. The target is 20–30 reviewed Alpaca pairs.

## Run the pilot on Scholar

From the repo root, build the pilot dataset as above, then set up the environment and cache the pinned model files. Put Python 3.11 and uv on your PATH first.

```sh
export UV_PROJECT_ENVIRONMENT="$SCRATCH/cs577/pilot-venv"
export HF_HOME="$SCRATCH/cs577/hf"
./setup.sh gpu
"$UV_PROJECT_ENVIRONMENT/bin/hf" download Qwen/Qwen2.5-3B-Instruct \
  --revision aa8e72537993ba99e69dfaafa59ed015b17504d1 \
  --include '*.json' '*.txt' '*.safetensors'
mkdir -p "$SCRATCH/cs577/runs"
sbatch --output="$SCRATCH/cs577/runs/slurm-%j.log" scripts/pilot.slurm
```

The job requests one A40 and runs greedy baseline inference on all three prompt types, followed by four BF16 LoRA updates with loss on the assistant answer and end-of-turn token. It saves predictions, logs, GPU memory and training throughput, input and script hashes, and the smoke-test adapter to `$SCRATCH/cs577/runs/<job-id>/`. The adapter is for verifying the training pipeline. The main experiment uses separate training and evaluation splits.

## Generate the frozen training files

Use the Round 5 selection and the pinned Alpaca JSON at `data/raw/alpaca_data.json`. Run from the repository root with Python 3.11. [Frozen selection](data/frozen/pairs.json) · [Freeze manifest](data/frozen/freeze_manifest.json) · [Scholar handoff](docs/HANDOFF.md)

```sh
python3.11 scripts/build_dataset.py data/raw/alpaca_data.json \
  --selection data/frozen/pairs.json --output artifacts/main --seed 577
```

The builder checks the Alpaca checksum, excludes pilot rows, and rejects reused IDs and exact duplicates across splits. It writes:

- `train_pairs.jsonl`, `validation_pairs.jsonl`, and `test_pairs.jsonl` with the three prompt variants per pair.
- `train_B.jsonl` and `train_C.jsonl` with two slots per source and one shared clean example per distinct training carrier. B randomly assigns source contexts and keeps repeated examples when both slots have the same context. C uses both contexts for every source.
- `manifest.json` with the seed, selection checksum, carrier usage, and actual context coverage.
- `test_overlap_review.jsonl` with up to three nearest training or validation rows for each test row, based on word overlap. Use these suggestions when auditing new selections.

Use a new output directory for each export you want to keep.

## Curate new dataset drafts

These commands document how the dataset was prepared. To run the experiment, use the frozen export above.

### Review the unsplit pools

Review candidates in batches, starting with standalone questions. Approve rows whose original Alpaca answer is sound and replace rejected rows with new candidates. Keep row IDs and review decisions so replacements can be checked against previous batches. Each approved-pool JSON contains the Alpaca revision and SHA-256 plus a `rows` list with each original row, its zero-based `row` ID, and a `review` note.

```sh
python3.11 scripts/validate_pool.py data/raw/alpaca_data.json \
  artifacts/source_audit/approved_through_0600.json --role source --count 200
python3.11 -m unittest discover -s tests
```

The validator checks the row count, pilot exclusion, ID and normalized text uniqueness, input requirements, and exact preservation of the original text and answer. Use `--near-duplicate-report` with a new JSON path to flag similar tasks by shared words or character trigrams. Review flagged comparisons before accepting a batch.

### Split the reviewed pools

```sh
python3.11 scripts/propose_pairs.py data/raw/alpaca_data.json split-reviewed \
  --sources artifacts/source_audit/approved_through_4046.json \
  --carriers artifacts/carrier_audit/rechecked_500.json \
  --train-groups artifacts/split_train_groups.json \
  --output artifacts/reviewed_splits --seed 577
```

This assigns all approved rows to train, validation, and test. The standalone counts are 1,000, 100, and 200, and the carrier counts are 200, 100, and 200. The optional group file records reviewed rows that share content. The splitter keeps those rows together in training by swapping each held-out group member with a training row of the same role.

### Draft commands

The following commands generate exploratory selections. For the main experiment, finish reviewing the unsplit pools before assigning splits or pairing rows.

```sh
python3.11 scripts/propose_pairs.py data/raw/alpaca_data.json pools \
  --train 1000 --validation 100 --test 200 --output artifacts/main_pools
```

This writes `sources.json` and `carriers.json` with row IDs, original Alpaca text, and train, validation, and test assignments. The script requires a task and an answer, excludes pilot rows, and prefers answers of up to 12 words. If you omit the size flags, it selects a smaller 30/10/10 draft.

Match the saved rows within each split using a fixed random seed. The training pool uses roughly one carrier for five sources, and validation and test use one carrier per source.

```sh
python3.11 scripts/propose_pairs.py data/raw/alpaca_data.json pair \
  --pools artifacts/main_pools --output artifacts/main_selection_draft_full.json
```

For new drafts, check the proposed answers and the scope of the carrier task. Correct or drop pairs when needed and record decisions in the optional `review` field. The exporter reads the selection's `splits` object. A carrier can serve multiple sources within one split.
