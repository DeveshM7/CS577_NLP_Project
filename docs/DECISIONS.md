# Project decisions

This log records the working design from the discussion. The documents are drafts for team review.

| Decision | Status | Rationale |
|---|---|---|
| Make B vs. C the primary comparison and use A as the baseline | Core design | Same starting checkpoint, source instructions, and training budget let us study guaranteed dual-context coverage. |
| Keep the 2,000-distinct-instruction design as a separate diversity control | Follow-up priority | It tests the tradeoff between repeating requests across contexts and covering a broader set. |
| Use Alpaca with a custom builder and explicit task/data fields | Working choice | Alpaca provides source requests and input-dependent tasks. StruQ supplies the structured-training precedent. The builder must track source IDs, splits, and context assignments. |
| Start with self-contained source requests and input-dependent carrier tasks | Working choice | Each reused request is complete by itself. The carrier requires the model to use its data to answer. |
| Freeze splits before building examples and track provenance | Working choice | Disjoint row IDs prevent reuse across splits. Exact-match checks and a targeted similarity review reduce obvious near-duplicate overlap. |
| Use 1,000 train, 100 validation, and 200 test pairs, with 20 pilot pairs held out | Frozen | The Round 5 checksums define the main data. The pilot stays outside the main splits. |
| Share the carrier mapping across B and C | Implemented | Training uses 200 carriers five times each. Validation and test use one distinct carrier per pair. |
| Use Qwen2.5-3B-Instruct with supervised LoRA for B and C | Pilot verified | The Scholar pilot completed four updates. Short training and adapter reload checks with the main exports are next. |
| Keep the same decoding and training settings across comparable arms | Working choice | Differences in settings would complicate interpretation. Log token counts and training steps. |
| Use paired accuracy as primary and report four supporting rates | Working choice | Separates direct correctness, carrier correctness, following the embedded request, and false refusals. |
| Prioritize BIPIA, then IHEval, with pairwise loss deferred | Proposed scope | The held-out Alpaca experiment answers the primary question. External slices test transfer. |
| Keep code and lightweight preparation local and run Qwen with Slurm on Scholar | Working choice | The Mac is the main editing workspace. Scholar supplies GPU compute. |

## Prompt formatting limitations

- **Blank-line separator.** In the reviewed draft, stopping at the first blank line recovered the original input exactly in 1,285 of 1,300 pairs (98.8%). The main exporter now joins the original input and embedded instruction with one newline.
- **Fixed instruction position.** The reviewed draft always placed the embedded instruction after the original input. The main exporter now shuffles an equal number of before and after placements within each split using the dataset seed. A pair uses the same placement across model variants.

## Frozen dataset pairs

Round 5 is the reference version for the 1,000 training, 100 validation, and 200 test pairs. The freeze manifest records the Alpaca revision, export seed, and SHA-256 checksums for the pair selection and prompt files. Its `export/` entries identify the generated pair files by filename. The `untrusted_prompts.jsonl` entry identifies the original local prompt preview. [Frozen selection](../data/frozen/pairs.json) · [Freeze manifest](../data/frozen/freeze_manifest.json)

Both training arms use the same 200 clean carrier examples, one per distinct training carrier. Each arm has 2,200 examples, including 1,000 direct and 1,000 embedded-request examples. The clean examples reduce the association between nonempty data and an embedded request.

## Open decisions

- Scoring rubric and implementation are deferred. A response can earn carrier credit by correctly applying the authorized task to embedded text as data.
- Confirm full-run LoRA settings, sequence length, precision, run sizes, and seeds after the main training smoke tests on Scholar.
- Benchmark slices, priority of the diversity control versus a seed repeat, and actual teammate ownership.
