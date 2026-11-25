# Implementation plan

**Status** — Training, inference, and first-pass scoring are complete for seeds 577 and 578 on the original held-out set. See the [handoff](HANDOFF.md) for results and artifact paths.

| Stage | Local work | Scholar work | Exit condition |
|---|---|---|---|
| 1. Pilot · complete | Audited 20 pairs and exported their three prompt variants. | Job 494390 produced 60 baseline predictions and completed four LoRA updates. | Pilot outputs, model revision, and compute measurements are recorded in the run registry. |
| 2. Main data · complete | Froze 1,300 pairs and verified regenerated prompt files against their checksums. Exported 2,200 training records per arm, including 200 shared clean carriers. Published the frozen selection and exporter on `main`. | Generate or copy the main export during training setup. | The frozen pair files reproduce and both training inputs are available. |
| 3. Main training and inference · complete | Published the frozen selection and checksum manifest through `main`. | Trained B and C at seeds 577 and 578, verified adapter reloads, and generated predictions on the same 200 test pairs using the shared A baseline. | Code revision, data checksums, training configuration, adapters, predictions, and Slurm logs are recorded. |
| 4. Scoring and analysis · in progress | Blind-review pair disagreements and curate a proposed 400-pair test extension. | Completed first-pass scoring of both seeds. Update inference to accept the extension and run the existing adapters on it once frozen. | Report paired accuracy and supporting rates on the fresh pairs first, then the combined set. Record uncertainty and failure cases. |
| 5. Package for submission | Prepare the README and uv-based setup script. Assemble the report, code zip, and human-created video. | Check package reproducibility in the grading container when available. | Submission artifacts satisfy the handout and can be traced back to the runs. |

## Proposed ownership

Ownership remains open until teammates confirm availability. Devesh can lead Scholar runs and integration, Ishan can lead dataset creation and audits, and Hargun can lead scoring and failure analysis. Define the shared data format first so those tracks can proceed independently. If availability is uneven, keep A/B/C on the critical path and assign bounded review work. Record actual contributions for the report appendix.
