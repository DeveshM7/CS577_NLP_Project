# Mac and Scholar handoff

**Current phase:** Local pilot setup and dataset preparation. The first 20 reviewed Alpaca pilot pairs are exported. GPU runs are pending.

Read the [spec](PROJECT_SPEC.md), [implementation plan](IMPLEMENTATION_PLAN.md), and [decision log](DECISIONS.md) before starting new work. Use this file for current work and run status. Record durable design changes in the decision log.

## Current state

| Area | State |
|---|---|
| Local | Python 3.11 is available. The uv setup and pilot dataset builder are in place. The raw Alpaca download matches upstream commit 761dc5bfbdeeffa89b8bff5d038781a4055f796a byte for byte. The tracked selection is data/pilot_selection.json, and the generated output is artifacts/pilot.jsonl. uv and the GPU lockfile still need verification. The local checkout has no commits. |
| Scholar | User can log in from a terminal. The local Codex environment failed to resolve the Scholar hostname during a read-only SSH probe. Slurm submission and GPU allocation remain unverified in Codex. |
| Next | Add the Qwen inference and LoRA smoke-test entry points and a Slurm submission script. Resolve and verify the GPU environment on Scholar. |

## Coordination

- Both sessions read the latest shared revision and this file before picking up work. Record the owner, branch or commit, and next action when handing off. During concurrent work, use separate branches and coordinate edits to shared files.
- The local session owns dataset preparation, source/config changes, and result review by default. The Scholar session owns Slurm submission, monitoring, and run artifact collection by default. Each run should record the exact code and data revision it used.
- Keep source, configuration, and lightweight run manifests in Git. Keep cached weights, adapters, raw run artifacts, and large datasets in appropriate storage and record the path or transfer location. Keep credentials out of the repo.
- On Scholar, submit Qwen computation through Slurm. Save logs and raw predictions with the run. Update the registry when jobs are queued, finish, or need follow-up. [Scholar GPU jobs](https://docs.rcac.purdue.edu/userguides/scholar/run_jobs/gpu/)

## Run registry

| Date | Owner | Run and status | Commit / data / config | Slurm job ID / GPU | Artifact location | Next step |
|---|---|---|---|---|---|---|
| Pending | Unassigned | Pilot inference and LoRA smoke test | Pending | Pending | Pending | Await the inference and training entry points and Slurm script. |

## Handoff log

| Date | Location | Update | Next action |
|---|---|---|---|
| 2026-11-23 | Local | Drafted the spec, implementation plan, and decision log. | Team review. |
| 2026-11-23 | Local | Added uv setup, README, and pilot dataset builder. Verified export with synthetic rows. | Obtain the Alpaca JSON and review real pairs. |
| 2026-11-23 | Local | Verified the pinned Alpaca source and exported 20 reviewed pairs with 60 prompts. | Prepare the first Qwen pilot job. |
