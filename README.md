# CS 577 instruction boundary project

See the [spec](docs/PROJECT_SPEC.md), [implementation plan](docs/IMPLEMENTATION_PLAN.md), and [Scholar handoff](docs/HANDOFF.md).

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
