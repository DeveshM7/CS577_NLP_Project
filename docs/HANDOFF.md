# Mac and Scholar handoff

**Current phase:** Scholar pilot complete. Baseline predictions are ready for scoring review.

Read the [spec](PROJECT_SPEC.md), [implementation plan](IMPLEMENTATION_PLAN.md), and [decision log](DECISIONS.md) before starting new work. Use this file for current work and run status. Record durable design changes in the decision log.

## Current state

| Area | State |
|---|---|
| Data | The tracked selection is `data/pilot_selection.json`. Scholar regenerated `artifacts/pilot.jsonl` from the checksum-verified Alpaca revision `761dc5bfbdeeffa89b8bff5d038781a4055f796a`. The JSONL SHA-256 is `5fab92219a2e03543567c637e2e1dc4e02e31d8764c3081069b2b04e03ffe3e6`. |
| Scholar | `scholar/pilot` contains the runner, Slurm script, and uv lockfile. The locked environment is installed at `$SCRATCH/cs577/pilot-venv`. A Slurm job verified PyTorch 2.9.1+cu128 on one A40. Model and tokenizer revision `aa8e72537993ba99e69dfaafa59ed015b17504d1` is cached at `$SCRATCH/cs577/hf`. |
| Next | Review the 60 baseline predictions, double-score a subset, and establish the scoring rubric before the main experiment. |

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
| 2026-11-23 | Scholar | Implemented and verified the pilot on branch `scholar/pilot`. Stored predictions, logs, provenance, and the smoke-test adapter in Scholar scratch. | Use job 494390 for scoring review. |
