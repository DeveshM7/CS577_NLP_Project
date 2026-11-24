# Project decisions

This log records the working design from the discussion. The documents are drafts for team review.

| Decision | Status | Rationale |
|---|---|---|
| Make B vs. C the primary comparison and use A as the baseline | Core design | Same starting checkpoint, source instructions, and training budget let us study guaranteed dual-context coverage. |
| Keep the 2,000-distinct-instruction design as a separate diversity control | Follow-up priority | It tests the tradeoff between repeating requests across contexts and covering a broader set. |
| Use Alpaca with a custom builder and explicit task/data fields | Working choice | Alpaca provides source requests and input-dependent tasks. StruQ supplies the structured-training precedent. The builder must track source IDs, splits, and context assignments. |
| Start with self-contained source requests and input-dependent carrier tasks | Working choice | Each reused request is complete by itself. The carrier requires the model to use its data to answer. |
| Freeze splits before building examples and track provenance | Working choice | Disjoint row IDs prevent reuse across splits. Exact-match checks and a targeted similarity review reduce obvious near-duplicate overlap. |
| Target 1,000 training requests, 100 validation pairs, 200 test pairs, and a 20–30-pair pilot | Provisional | Scale depends on the supply of eligible rows, reviewer workload, and measured GPU throughput. |
| Target 200 distinct training carriers and share a fixed mapping across B and C | Agreed starting target | Limits curation to roughly five source requests per carrier while keeping the pairing consistent across arms. Aim for distinct test carriers. |
| Use Qwen2.5-3B-Instruct with supervised LoRA for B and C | Pilot candidate | An instruction-tuned starting point with inexpensive separate adapters supports a controlled comparison on Scholar. |
| Keep the same decoding and training settings across comparable arms | Working choice | Differences in settings would complicate interpretation. Log token counts and training steps. |
| Use paired accuracy as primary and report four supporting rates | Working choice | Separates direct correctness, carrier correctness, following the embedded request, and false refusals. |
| Prioritize BIPIA, then IHEval, with pairwise loss deferred | Proposed scope | The held-out Alpaca experiment answers the primary question. External slices test transfer. |
| Keep code and lightweight preparation local and run Qwen with Slurm on Scholar | Working choice | The Mac is the main editing workspace. Scholar supplies GPU compute. |

## Open decisions

- Exact Alpaca revision, accepted task categories, eligible carrier rows, the exact reuse cap, and text format.
- Whether B's repeated same-context slots use identical records or controlled variants. Fix this before freezing the manifests.
- Automated versus manual scoring, what counts as following a request when it is attempted incorrectly, how to handle partially correct or contradictory outputs, refusal denominators, ambiguity adjudication, and the agreement threshold for automated scoring.
- Actual model revision, GPU/account availability, LoRA settings, sequence length, precision, run sizes, and seeds. Fix shared settings after the pilot.
- Benchmark slices, priority of the diversity control versus a seed repeat, and actual teammate ownership.
