# Mac and Scholar handoff

**Current phase** — The Scholar pilot and local dataset freeze are complete. Next, prepare and test the main B/C training jobs on Scholar.

Read the [spec](PROJECT_SPEC.md), [implementation plan](IMPLEMENTATION_PLAN.md), and [decision log](DECISIONS.md) before starting new work. Use this file for current work and run status. Record durable design changes in the decision log.

## Current state

| Area | State |
|---|---|
| Pilot data | The tracked selection is `data/pilot_selection.json`. Scholar regenerated `artifacts/pilot.jsonl` from the checksum-verified Alpaca revision `761dc5bfbdeeffa89b8bff5d038781a4055f796a`. The JSONL SHA-256 is `5fab92219a2e03543567c637e2e1dc4e02e31d8764c3081069b2b04e03ffe3e6`. |
| Main data | The frozen Round 5 selection contains 1,000 train, 100 validation, and 200 test pairs. The current training export gives B and C 2,200 records each, including the same 200 clean carriers. |
| Repository | `main` contains the pilot runner, pilot Slurm script, uv lockfile, frozen selection, and dataset exporter as of `1057604`. The old `scholar/pilot` branch was merged and deleted. |
| Scholar | The pilot used the locked environment at `$SCRATCH/cs577/pilot-venv`. A Slurm job verified PyTorch 2.9.1+cu128 on one A40. The pilot cached model and tokenizer revision `aa8e72537993ba99e69dfaafa59ed015b17504d1` at `$SCRATCH/cs577/hf`. |
| Next | Stage the main artifacts on Scholar. Set up the main training entry point, run a short LoRA job for each arm, and verify that each saved adapter reloads for inference. The scoring rubric is deferred. |

## Main dataset handoff

- From the Scholar checkout, switch to `main` and fast-forward with `git pull --ff-only origin main`. The export command uses the frozen selection, the same pinned Alpaca JSON used for the pilot, and seed 577. On Scholar, set `--output` to a new directory under `$SCRATCH/cs577/`. The Mac export is also available at `artifacts/reviewed_pairs_repaired_round_5/export_with_clean_carriers/`. [Export command](../README.md#generate-the-frozen-training-files)
- Use `train_B.jsonl` and `train_C.jsonl` for training. The three `*_pairs.jsonl` files provide the direct, clean carrier, and embedded-request prompts for each pair. Check their hashes against the corresponding filenames in the freeze manifest, which records them under `export/`.
- Use the same pinned model and tokenizer for both arms. Record the training file checksums, code revision, seed, and shared settings with each Slurm job. Save enough output from the short runs to verify adapter reload and inference before submitting the full jobs.

## Next Scholar task

Set up the main LoRA training entry point and Slurm job using the pilot as a starting point. Run a few updates for each arm on a small subset of its training file that covers direct, embedded-request, and clean carrier examples. Check that training computes loss on the expected answer tokens, the loss is finite, and each saved adapter reloads in a fresh process and generates a prediction. Record GPU memory and throughput to inform the full runs.

## Coordination

- Both sessions read the latest shared revision and this file before picking up work. Record the owner, branch or commit, and next action when handing off. During concurrent work, use separate branches and coordinate edits to shared files.
- The local session owns dataset preparation, source/config changes, and result review by default. The Scholar session owns Slurm submission, monitoring, and run artifact collection by default. Each run should record the exact code and data revision it used.
- Keep source, configuration, and lightweight run manifests in Git. Keep cached weights, adapters, raw run artifacts, and large datasets in appropriate storage and record the path or transfer location. Keep credentials out of the repo.
- On Scholar, submit Qwen computation through Slurm. Save logs and raw predictions with the run. Update the registry when jobs are queued, finish, or need follow-up. [Scholar GPU jobs](https://docs.rcac.purdue.edu/userguides/scholar/run_jobs/gpu/)

## Run registry

| Date | Owner | Run and status | Commit / data / config | Slurm job ID / GPU | Artifact location | Next step |
|---|---|---|---|---|---|---|
| 2026-11-23 | Scholar | Baseline inference completed with all 60 predictions. LoRA stopped before its first update on a chat-template tokenization error. | Base `b67f5d9` with script and lockfile hashes matching `f80abb4`. 20 Alpaca pilot pairs. | 494383 / A40 | `$SCRATCH/cs577/runs/494383/` | Corrected tokenization and verified assistant-answer alignment on 40 direct and untrusted prompts. |
| 2026-11-23 | Scholar | Completed in 52 seconds. Verified 60 unique predictions and four finite LoRA updates. Peak GPU allocation was 5.87 GiB for inference and 6.34 GiB for training. The smoke test processed 227 input tokens/s across four updates. | `6cadbe9`, same 20 pairs and model revision. Greedy decoding with a 128-token cap, BF16, rank 8 on query/value projections, batch size 1, seed 577. | 494390 / A40 | `$SCRATCH/cs577/runs/494390/` | Review `predictions.jsonl`; `smoke.json` records training measurements. |

The 60 [pilot predictions](../artifacts/pilot-run-494390/predictions.jsonl) are tracked for scoring review.

## Handoff log

| Date | Location | Update | Next action |
|---|---|---|---|
| 2026-11-23 | Local | Drafted the spec, implementation plan, and decision log. | Team review. |
| 2026-11-23 | Local | Added uv setup, README, and pilot dataset builder. Verified export with synthetic rows. | Obtain the Alpaca JSON and review real pairs. |
| 2026-11-23 | Local | Verified the pinned Alpaca source and exported 20 reviewed pairs with 60 prompts. | Prepare the first Qwen pilot job. |
| 2026-11-23 | Scholar | Implemented and verified the pilot on the former `scholar/pilot` branch. Stored predictions, logs, provenance, and the smoke-test adapter in Scholar scratch. | The pilot outputs are available for later scoring. |
