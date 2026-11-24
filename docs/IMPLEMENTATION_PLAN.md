# Implementation plan

**Status:** Draft. Begin implementation after team review of the [specification](PROJECT_SPEC.md).

| Stage | Local work | Scholar work | Exit condition |
|---|---|---|---|
| 1. Set up the pilot | Pick and audit 20–30 source/carrier pairs. Define stable row IDs, split metadata, clean carrier prompts, and a scoring sheet. Prepare the Python environment, a dataset export, and a minimal training and evaluation entry point. | Confirm access and the Slurm allocation. Set up the environment and cache the pinned model. | The sample records and expected answers can be traced to their source rows, and the same repo revision is available on both machines. |
| 2. Check data, scoring, and compute | Review predictions with model labels hidden. Independently double-score a subset and resolve disagreements. Document any task filters or rubric changes. | Run the unmodified Qwen checkpoint on direct, clean carrier, and paired prompts. Run a few LoRA steps, saving logs for peak memory and throughput. | Clean tasks are reasonably answerable, paired cases have room for improvement, answers can be scored consistently, and training fits the allocation. |
| 3. Freeze the main run | Build splits, check exact overlaps, and review likely near-duplicate matches for held-out rows. Generate B and C from the same training source IDs. Record the count by context and per-source coverage. Freeze the rubric, test examples, and manifest before comparing results. | Train B and C from the same pinned checkpoint with matching settings. Evaluate A, B, and C on the same held-out prompts. | Saved predictions, adapter locations, and job logs are tied to the data and config revision. |
| 4. Analyze and extend | Score the predictions, calculate C minus B and supporting rates, and inspect disagreements. | Run the diversity control, benchmark slices, or another seed as time allows. Prioritize a repeat if the first seed gives a small or unstable-looking difference. | Results table, uncertainty reporting, failure analysis, and deviations from the proposal are ready for the report. |
| 5. Package for submission | Prepare the README and uv-based setup script. Assemble the report, code zip, and human-created video. | Check package reproducibility in the grading container when available. | Submission artifacts satisfy the handout and can be traced back to the runs. |

## Proposed ownership

Ownership remains open until teammates confirm availability. Devesh can lead Scholar runs and integration, Ishan can lead dataset creation and audits, and Hargun can lead scoring and failure analysis. Define the shared data format first so those tracks can proceed independently. If availability is uneven, keep A/B/C on the critical path and assign bounded review work. Record actual contributions for the report appendix.
