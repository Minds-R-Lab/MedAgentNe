# MedAgentNet

Reference implementation and evaluation harness for **MedAgentNet**, an
inference-time federated multi-agent architecture for cross-departmental
clinical reasoning under per-patient consent.

Each hospital department runs an agent with access only to its own records. An
orchestrator issues purpose-limited structured queries, verifies patient consent
for each directed department pair, and assembles the tier-limited answers into a
shared clinical picture that it reasons over. **Records never leave their
department; only derived, tier-limited answers travel.**

> **Status.** The manuscript is under review at *Applied Soft Computing*
> (special issue on Agentic AI). This repository contains the revised (R1)
> implementation, the benchmark, and the results. The figures currently in
> `results/validation_mock/` come from the deterministic rule-based backend used
> to validate the harness end to end; the language-model runs are in progress.
> See [Reproducing the results](#reproducing-the-results).

---

## Why this repository looks the way it does

The first submission reported 100% detection with zero variance. Three reviewers
questioned that, and they were right to. Re-auditing the code against their
comments found four defects that invalidated the reported figures:

| # | Defect | Where | Now |
|---|---|---|---|
| 1 | The ground-truth answer was copied into the query context and rendered into every agent prompt at every disclosure tier | `simulation/runner.py` L160–167 | Queries are built from the patient's own record only; the agent holds a whitelist of permitted context fields. Locked by `tests/test_no_ground_truth_leakage.py` |
| 2 | Scoring counted *any* alert as a detection, with no check against the planted finding | `simulation/experiments.py` L736–781 | Four-level graded scorer requiring the reported finding to identify the planted one. Locked by `tests/test_scoring.py` |
| 3 | Partial consent revocation silently re-permitted the pair via the default policy | `protocol/consent.py` L51–54 | Denial is explicit and takes precedence. Locked by `tests/test_consent_enforcement.py` |
| 4 | The cross-departmental reasoning step described in the paper was not implemented | `agents/core.py` | Implemented as `OrchestratorAgent._assemble_evidence` + `_grounded_alerts` (Algorithm 2 in the paper) |

`docs/CHANGELOG_R1.md` records every change with file and line references, and
the original R0 results are preserved unmodified in `results/r0_archive/` so the
correction can be checked rather than taken on trust.

The four defects are now **regression tests**. They run in CI on every push.

---

## What the results say

From the harness self-test at 200 patients, three seeds
(`results/validation_mock/results.json`). Language-model figures will replace
these; the architectural comparisons are expected to hold, the absolute
detection rates are not.

**Federating the reasoning costs little accuracy.** Against a conventional
decision-support system applying the same knowledge base to a *fully aggregated*
record, on identical cases with identical scoring:

| System | Data centralised | F1 | vs MedAgentNet |
|---|---|---|---|
| MedAgentNet | no | 0.992 | — |
| Conventional CDSS, aggregated record | **yes** | 0.985 | *p* = 1.0 |
| Centralized single agent, full record | **yes** | 0.756 | *p* < 0.001 |
| Federated retrieval, no reasoning layer | no | 0.000 | *p* < 0.001 |

**The coordination layer is what does the work, not the language model.**

| Configuration | F1 | Conflict detection |
|---|---|---|
| MedAgentNet (full) | 0.992 | 100% |
| − cross-departmental assembly | 0.292 | 10.6% |
| − orchestration (single department) | 0.158 | — |
| − relevance routing (broadcast) | 0.992 | 100% |
| − tiered disclosure | 0.977 (precision 0.984 → 0.956) | 100% |
| model synthesis only, no knowledge base | 0.723 | — |

Relevance routing changes neither accuracy nor query volume at ten departments,
because a medication-conflict query already closes over every prescribing
department. We report the component as untested rather than validated.

**Disclosure has a measured price.** Exposure is the fraction of a responding
department's identifiable facts appearing verbatim in what it sent:

| | Tier 1 (flag only) | Tier 2 (clinical summary) | Tier 3 (full context) |
|---|---|---|---|
| Conflict detection | 13.3% | 100% | 100% |
| Field exposure | 0.0% | 69.0% | 69.1% |
| Mean anonymity set | n/a (nothing disclosed) | 16.4 of 200 | 16.8 of 200 |

Tier 2 is the operating point. Tier 3 adds no measurable utility and lowers
precision.

**Consent degrades utility roughly linearly, and *which* channels are revoked
matters more than how many.** Over ten independently sampled restriction graphs
per level: 100% → 80.5% → 62.3% → 46.3% → 25.7% → 10.0% as the revoked fraction
rises from 0 to 100%. Revoking exactly the pairs carrying the evidence, matched
for count against a random selection, gives **10.0% versus 56.7%**.

**Generalisation.** On a held-out scenario family authored after the prompts,
routing rules and knowledge base were frozen, conflict detection falls from
99.7% to 90.0%.

**Two mitigations that do not work,** reported rather than omitted: a per-pair
query budget cut traffic four-fold and left record reconstruction unchanged at
67% (the disclosure tier bounds it to 0%); and requiring corroboration removes
fabricated critical alerts from independently compromised agents but not from
colluding ones.

---

## Install

```bash
git clone https://github.com/Minds-R-Lab/MedAgentNe.git
cd MedAgentNet
pip install -r requirements.txt
```

Python 3.11+. The only hard dependencies are `pyyaml` and `requests`.

## Run

Everything runs without a model server, using the deterministic rule-based
backend. This is the fastest way to see the pipeline work:

```bash
cd medagentnet
python run_r1.py --provider mock --patients 200 --tag validation
```

About 80 seconds. Writes `data/results_r1/<tag>_<timestamp>/results.json` and
`tables.tex`.

With a language model:

```bash
ollama serve &
ollama pull llama3.1:8b
python run_r1.py --provider ollama --model llama3.1:8b --patients 200 \
                 --tag reported --experiments e9 e1 e2 e3 e4 e5 e7 e8
```

Backend matrix (pull each model first):

```bash
python run_r1.py --provider ollama --model llama3.1:8b \
                 --backend-matrix llama3.1:8b qwen2.5:7b mistral:7b phi3:mini \
                 --include-mock-in-matrix --experiments e6 --tag backends
```

The provider is never silently substituted: if a model is not reachable the run
stops rather than producing rule-based numbers under a model's name.

## Experiments

| ID | Question |
|---|---|
| E1 | Disclosure tier: utility against measured exposure |
| E2 | Consent restriction 0–100%, ten sampled graphs per level, plus targeted removal |
| E3 | Seed variance and held-out generalisation |
| E4 | Ablation matrix — which component does the work |
| E5 | Baselines: centralized LLM, conventional CDSS, direct retrieval |
| E6 | Backend matrix: families and sizes, format reliability, tokens, latency |
| E7 | Scalability: patients × departments × concurrency |
| E8 | Threat model A1–A5, each with and without its mitigation |
| E9 | Query-context audit: how much ground truth the R0 construction carried |

`docs/EXPERIMENTS.md` describes each in detail.

## Tests

```bash
cd medagentnet && python -m pytest tests -q
```

35 tests, under five seconds. They cover the four corrected defects plus the
architectural invariants — the orchestrator holding no patient store, Tier 1
disclosing nothing identifiable, an agent failure degrading rather than aborting,
and negative controls being genuinely negative.

CI runs these on every push, and additionally fails the build if a headline
result regresses — see `docs/CI.md`. The workflow ships as
`docs/ci-workflow.yml`; move it to `.github/workflows/ci.yml` to enable it.

## Layout

```
medagentnet/
  agents/core.py           department agents, orchestrator, routing, assembly
  protocol/models.py       data model; PHI inventory
  protocol/consent.py      per-pair consent, tiers, single-use tokens
  protocol/interactions.py grounded interaction knowledge base
  llm/                     backend abstraction (mock, Ollama, OpenAI-compatible, HF)
  data/generator.py        base synthetic cohort
  data/generator_hard.py   R1 cohorts, corruption operators, negative-control audit
  data/hard_cases.py       matched negatives, ambiguous cases, held-out family, notes
  simulation/scenarios.py  leak-free query construction (+ the R0 construction, for E9)
  simulation/evaluation.py graded scorer, Wilson intervals, McNemar
  simulation/privacy.py    exposure, cumulative reconstruction, anonymity set
  simulation/baselines.py  ablations and external comparison systems
  simulation/adversarial.py threat model A1–A5
  simulation/experiments_v2.py  E1–E9
  simulation/tables.py     LaTeX tables generated from results.json
  run_r1.py                CLI
docs/     changelog, threat model, benchmark and experiment documentation
results/  validation run, and the archived R0 results
tests/    regression tests for the corrected defects
```

Tables in the paper are generated from `results.json` by `simulation/tables.py`,
not transcribed, so any figure can be traced to the run that produced it.

## Scope and limitations

The evaluation is synthetic. It is substantially harder than its predecessor —
drug-matched negatives, ambiguity, missing fields, contradictory documentation,
stale prescriptions, free-text notes and a held-out family — but generated
records are still generated. The claims here are **architectural, not clinical**.
No clinician has adjudicated any alert, no prospective study has been run, and
nothing in this repository supports a claim of clinical benefit. It is research
software and is not suitable for use in patient care.

Further limitations, including idealised departmental partitioning and the fact
that most of the inference comes from the knowledge base rather than the model,
are set out in the manuscript and summarised in `docs/CHANGELOG_R1.md`.

## Citation

See `CITATION.cff`. The paper is under review; this file is updated when it
appears.

## License and status

MIT — see `LICENSE`.

Research software. Not a medical device, not validated for clinical use, and
not to be used in the care of patients. All records are synthetic. See
`NOTICE.md`.
