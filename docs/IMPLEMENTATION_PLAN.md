# Implementation plan

**Status** — Pilot and dataset preparation are complete. The next work is on Scholar. See the [handoff](HANDOFF.md) for the export location and pilot run details.

| Stage | Local work | Scholar work | Exit condition |
|---|---|---|---|
| 1. Pilot · complete | Audited 20 pairs and exported their three prompt variants. | Job 494390 produced 60 baseline predictions and completed four LoRA updates. | Pilot outputs, model revision, and compute measurements are recorded in the run registry. |
| 2. Main data · complete | Froze 1,300 pairs and verified regenerated prompt files against their checksums. Exported 2,200 training records per arm, including 200 shared clean carriers. | Stage the frozen artifacts for the main run. | The frozen pair files reproduce and both training inputs are ready. |
| 3. Main training and inference · next | Make the frozen artifacts available to Scholar. | Set up the main training entry point. Run a short LoRA job from each arm's export, confirm saved adapters reload for inference, and record memory and throughput. Then train both arms with matching settings and save raw predictions for A, B, and C on the same held-out prompts. | Code revision, data checksums, training configuration, adapters, predictions, and Slurm logs are recorded. |
| 4. Scoring and analysis · deferred | Finalize the rubric before scoring or comparing the main predictions. Calculate C minus B and supporting rates, and inspect disagreements. | Run the diversity control, benchmark slices, or another seed as time allows. | Results table, uncertainty reporting, failure analysis, and deviations from the proposal are ready for the report. |
| 5. Package for submission | Prepare the README and uv-based setup script. Assemble the report, code zip, and human-created video. | Check package reproducibility in the grading container when available. | Submission artifacts satisfy the handout and can be traced back to the runs. |

## Proposed ownership

Ownership remains open until teammates confirm availability. Devesh can lead Scholar runs and integration, Ishan can lead dataset creation and audits, and Hargun can lead scoring and failure analysis. Define the shared data format first so those tracks can proceed independently. If availability is uneven, keep A/B/C on the critical path and assign bounded review work. Record actual contributions for the report appendix.
