# Mac and Scholar handoff

**Current phase** — Both training seeds have completed held-out inference and first-pass scoring. The proposed next step is to curate 400 fresh held-out pairs locally.

Read the [spec](PROJECT_SPEC.md), [implementation plan](IMPLEMENTATION_PLAN.md), and [decision log](DECISIONS.md) before starting new work. Use this file for current work and run status. Record durable design changes in the decision log.

## Current state

| Area | State |
|---|---|
| Pilot data | The tracked selection is `data/pilot_selection.json`. Scholar regenerated `artifacts/pilot.jsonl` from the checksum-verified Alpaca revision `761dc5bfbdeeffa89b8bff5d038781a4055f796a`. The JSONL SHA-256 is `5fab92219a2e03543567c637e2e1dc4e02e31d8764c3081069b2b04e03ffe3e6`. |
| Main data | The frozen Round 5 selection contains 1,000 train, 100 validation, and 200 test pairs. The current training export gives B and C 2,200 records each, including the same 200 clean carriers. |
| Repository | The Scholar jobs used source snapshots based on `23c2263`. The frozen selection and exporter were published on `main`. |
| Scholar | The pilot used the locked environment at `$SCRATCH/cs577/pilot-venv`. A Slurm job verified PyTorch 2.9.1+cu128 on one A40. The pilot cached model and tokenizer revision `aa8e72537993ba99e69dfaafa59ed015b17504d1` at `$SCRATCH/cs577/hf`. |
| Held-out scoring | One AI-assisted reviewer scored all 1,800 predictions. The per-response labels, counts, and rubric are in `$SCRATCH/cs577/held-out-review.d6hysA/`. Embedded-request execution was A 106/200, B 2/200, C 2/200. Paired accuracy was A 42/200, B 133/200, C 138/200. The paired bootstrap interval for C minus B was −1.5 to +7.0 percentage points at this training seed. |
| Seed-578 scoring | The same rubric was applied to all 1,200 B/C responses. Labels were carried over for 1,041 identical responses to the same prompts, and 159 changed response records were reviewed. Embedded-request execution was 2/200 for each arm. Paired accuracy was B 135/200 and C 138/200. C minus B was +1.5 percentage points with a paired bootstrap interval of −1.5 to +5.0 points. Labels and counts are in `$SCRATCH/cs577/held-out-review-seed578.YciySk/`. |

## Main dataset handoff

- From the Scholar checkout, switch to `main` and fast-forward with `git pull --ff-only origin main`. The export command uses the frozen selection, the same pinned Alpaca JSON used for the pilot, and seed 577. On Scholar, set `--output` to a new directory under `$SCRATCH/cs577/`. The Mac export is also available at `artifacts/reviewed_pairs_repaired_round_5/export_with_clean_carriers/`. [Export command](../README.md#generate-the-frozen-training-files)
- Use `train_B.jsonl` and `train_C.jsonl` for training. The three `*_pairs.jsonl` files provide the direct, clean carrier, and embedded-request prompts for each pair. Check their hashes against the corresponding filenames in the freeze manifest, which records them under `export/`.
- Use the same pinned model and tokenizer across arms. The run registry lists the trained adapters, inference jobs, and source snapshots.

## Proposed test-set extension

- The local session curates 400 fresh pairs with 200 embedded requests before the carrier data and 200 after. Apply the existing answer review and overlap checks against the pilot and all frozen splits. Freeze the new pairs and their prompts with checksums. Preserve the original 200 pairs and their exact prompts so the saved predictions remain comparable.
- The Scholar session updates the inference runner's current 200-pair check and runs A plus the B/C adapters from both seeds on the new prompts with the same decoding settings. Score the new set with the existing rubric and report its result before the combined 600-pair result.

## Completed Scholar smoke tests

The smoke jobs used the pilot LoRA settings and two examples per training context in each arm. Answer-token alignment was verified on all sampled examples. The run registry records the finite updates, fresh-process reloads, memory, and throughput.

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
| 2026-11-24 | Scholar | Main B/C smoke jobs completed. Each arm recorded six finite updates, two per context, and a fresh-process adapter reload generated all three prompts for validation pair `validation-32353-2191`. Peak training allocation was 6.66 GiB for both. Measured throughput was 71 tokens/s for B and 265 for C. | Base `23c2263` with Scholar working-tree scripts hashed in job provenance. Round 5 export in `$SCRATCH/cs577/main-round5-seed577/`. Pilot training settings. | B 494898, C 494899 / one A40 each | `$SCRATCH/cs577/runs/{494898,494899}/` | Choose matched full-run settings. Use a longer benchmark for throughput estimates. |
| 2026-11-25 | Scholar | Full B/C training completed. Each arm processed 4,400 examples over 138 finite updates. Both saved adapters reloaded in fresh processes and produced predictions for all three contexts of validation pair `validation-32353-2191`. B took 11m 07s and peaked at 7.11 GiB allocated during training. C took 10m 21s and peaked at 7.13 GiB. | Base `23c2263`, source snapshot `$SCRATCH/cs577/full-training-source.6rBT7t/`, Round 5 export, BF16 rank-8 LoRA, AdamW at 1e-4, seed 577, two epochs, effective batch 32, microbatch 1. | B 494960, C 494961 / one A40 each | `$SCRATCH/cs577/runs/{494960,494961}/` | Generate held-out predictions for A, B, and C. |
| 2026-11-25 | Scholar | Held-out inference completed for A, B, and C. Verified 600 unique and nonblank predictions per arm, 200 of each context. Six A responses reached the 128-token limit. | Base `23c2263`, source snapshot `$SCRATCH/cs577/held-out-source.27LvT4/`, same 200 frozen test pairs for all arms. BF16, greedy decoding, 128 new tokens maximum. B and C use adapters from jobs 494960 and 494961. | A 495074, B 495075, C 495076 / one A40 each | `$SCRATCH/cs577/runs/{495074,495075,495076}/`, one `predictions.jsonl` per arm | Finalize the rubric before scoring. |
| 2026-11-25 | Scholar | Seed-578 B/C repeat completed. Both arms processed 4,400 records over 138 finite updates and generated three validation predictions after fresh-process adapter reload. B ran for 10m 02s and peaked at 7.10 GiB allocated. C ran for 9m 57s and peaked at 7.13 GiB. The frozen inputs and optimization settings match the seed-577 runs. | Base `23c2263`, source snapshot `$SCRATCH/cs577/seed578-training-source.WxDZbA/`, data seed 577, training seed 578. | B 495084, C 495085 / one A40 each | `$SCRATCH/cs577/runs/{495084,495085}/` | Generate seed-578 predictions on the same 200 test pairs, reusing A's baseline predictions. |
| 2026-11-25 | Scholar | Seed-578 held-out inference completed for B and C. Verified 600 unique and nonblank predictions per arm, 200 of each context. No responses reached the generation limit. | Original inference source snapshot `$SCRATCH/cs577/held-out-source.27LvT4/`, same 200 test pairs, BF16, greedy, 128 new tokens maximum. Adapters from jobs 495084 and 495085. A's predictions remain in job 495074. | B 495090, C 495091 / one A40 each | `$SCRATCH/cs577/runs/{495090,495091}/`, one `predictions.jsonl` per arm | Score with the established rubric and compare the two training seeds. |

The 60 [pilot predictions](../artifacts/pilot-run-494390/predictions.jsonl) are tracked for scoring review.

## Handoff log

| Date | Location | Update | Next action |
|---|---|---|---|
| 2026-11-23 | Local | Drafted the spec, implementation plan, and decision log. | Team review. |
| 2026-11-23 | Local | Added uv setup, README, and pilot dataset builder. Verified export with synthetic rows. | Obtain the Alpaca JSON and review real pairs. |
| 2026-11-23 | Local | Verified the pinned Alpaca source and exported 20 reviewed pairs with 60 prompts. | Prepare the first Qwen pilot job. |
| 2026-11-23 | Scholar | Implemented and verified the pilot on the former `scholar/pilot` branch. Stored predictions, logs, provenance, and the smoke-test adapter in Scholar scratch. | The pilot outputs are available for later scoring. |
