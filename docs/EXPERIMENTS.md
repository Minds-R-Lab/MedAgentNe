# Experiments

All are driven by `medagentnet/run_r1.py` and implemented in
`simulation/experiments_v2.py`. Each writes a single `results.json`, and
`simulation/tables.py` generates the paper's LaTeX tables from it — no figure in
the paper is transcribed by hand.

```bash
python run_r1.py --provider mock --patients 200 --experiments e9 e1 e2 e3 e4 e5 e7 e8
```

| ID | Name | Varies | Reports |
|---|---|---|---|
| E1 | Disclosure tiers | tier 1/2/3 | detection, precision/recall/F1, field exposure, items disclosed, anonymity set, singling-out rate, latency |
| E2 | Consent restriction | revoked fraction 0–100%, 10 sampled graphs per level, plus targeted removal | observed denial rate, detection, F1 |
| E3 | Variance and generalisation | 5 seeds × {development, held-out} | pooled detection with CIs, per-scenario, per-negative-control, per-record-quality breakdowns |
| E4 | Ablation matrix | 10 configurations | detection, precision/recall/F1, queries per scenario, McNemar vs the full system |
| E5 | External baselines | 4 systems | precision/recall/F1, latency, McNemar |
| E6 | Backend matrix | model families and sizes | detection, false alarms, format-failure rate, token volume, latency |
| E7 | Scalability | patients × departments × concurrency | queries per scenario, throughput, latency percentiles |
| E8 | Adversarial | threats A1–A5 | each with and without its mitigation |
| E9 | Query-context audit | R0 vs R1 construction | disallowed fields, label terms, and the R0 pipeline scored end to end |

## Notes on individual experiments

**E2** samples directed department pairs uniformly at random per patient,
independently resampled for each graph. The *targeted* condition revokes exactly
the pairs carrying the relevant evidence, matched for the number of pairs
removed against a random selection, which isolates *which* pairs were removed
from *how many*. The observed denial rate is reported alongside utility so the
restriction can be verified to have taken effect — in R0 it did not.

**E4** includes `medagentnet_llm_only` and `medagentnet_grounded_only` so the
two arms of the assembly step can be attributed separately, and
`ablate_freetext_parser` specifically because the free-text fallback is itself a
keyword matcher and could otherwise be manufacturing detections.

**E5** does not include a single-agent RAG baseline: with a structured record of
this size retrieval is not the bottleneck, and the centralized single agent
already receives the entire record, which is the upper bound RAG approximates.
It does not run an external healthcare multi-agent framework either, because
those systems assume shared access to one record and cannot be run on a
federated cohort without changing the thing being compared.

**E6** requires each model to be pulled first. `create_llm_provider` raises
rather than substituting the rule-based backend, and `OllamaProvider` refuses to
substitute a different model, so a results file cannot be labelled with a model
that never ran. The backend identity returned by the server is recorded.

**E7** reports CPU/GPU/memory nowhere, because the runs are single-process on
one machine and those figures would characterise the hardware rather than the
architecture. No physical multi-node deployment has been tested.

**E9** runs the R0 pipeline end to end — legacy contexts *and* no
cross-departmental assembly — and scores it under both criteria, which is what
allows the change in the reported figures to be attributed rather than merely
noted.

## Runtime

| Backend | Full suite at 200 patients |
|---|---|
| Deterministic (`--provider mock`) | ~80 seconds |
| LLaMA 3.1 8B on one consumer GPU | ~6–9 hours; E2 and E4 dominate |

Reduce with `--patients 100`, or select individual experiments.
