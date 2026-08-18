# MedAgentNet — R1 revision notes

This document records what changed between the submitted version (R0) and this
revision, why, and how to reproduce every number in the manuscript.

Nothing from R0 was deleted. The original modules still run; the R1 additions
sit beside them so a reviewer can execute both and compare.

---

## 1. Defects found in the R0 harness and how they were corrected

### 1.1 The ground truth reached the agent prompts

`simulation/runner.py` (L160–167) and the duplicated block in
`simulation/experiments.py` (L96–112) built each query context out of the label:

```python
context = {
    "planned_procedure": conflict["trigger_procedure"],   # from the label
    "reason":            conflict["description"],         # the answer, in words
}
for med in patient.medications:                           # every department's
    context[f"current_med_{med.name}"] = med.name         # drugs, at any tier
```

For patterns it additionally passed `"expected": expected_diagnosis` and a
pre-computed `rising`/`declining` trend string. `agents/core.py` (L107–111)
rendered the whole context verbatim into every agent prompt at every disclosure
tier, so even Tier 1 "flag only" carried the answer.

**Correction.** `simulation/scenarios.py` builds queries from the patient's own
record and a department-level procedure vocabulary. `DepartmentAgent` now holds
a whitelist (`ALLOWED_CONTEXT_KEYS`) and drops anything a requesting clinician
would not supply, so the defect cannot recur silently. Experiment **E9**
measures both constructions: R0 carried 9.28 disallowed fields and 2.40
ground-truth terms per context, in 100 % of contexts; R1 carries 0 and 0.11.

### 1.2 The scorer never checked *which* finding was reported

`experiments.py` L736–744 counted a detection whenever any alert existed whose
type was not `no_conflict` or `parse_error`. Pattern scoring (L748–781) passed
if any of ~22 generic words — including `abnormal`, `elevated`, `kidney` —
appeared anywhere in any agent's free text.

**Correction.** `simulation/evaluation.py` grades every positive on four levels:
miss, flagged, localised (names the constituent medications or entities) and
explained (also names the mechanism). Precision, recall, specificity and F1 use
the localised criterion. Wilson 95 % intervals throughout. The R0 criterion is
still computed and reported alongside, as `legacy_loose_criterion`, so the two
can be compared directly.

### 1.3 Partial consent revocation had no effect

`experiments.py` L186–196 revoked a department pair with
`del consent_profiles[pid][pair]`. `protocol/consent.py` L51–54 then treated the
missing entry as unset and fell through to the `opt_in` default, re-granting
Tier 2. The committed R0 results confirm it: `consent_denied_count: 0` and
`consent_denial_rate: 0.0` for both `partial_30` and `partial_60`.

**Correction.** Denial is represented explicitly (`DENIED`) and takes precedence
over any default. `revoke_pairs()` added. Experiment **E2** now sweeps 0–100 %
over ten independently sampled restriction graphs per level and reports the
*observed* denial rate next to utility, so the restriction can be seen to bite.

### 1.4 No cross-departmental reasoning existed

`_synthesize_responses` relayed each department's own findings and never
combined them. A conflict whose limbs sat in different departments could
therefore only be found through the leaked context. Removing the leak dropped
detection to 8 %.

**Correction.** The orchestrator now assembles a `ClinicalEvidence` structure
from what departments actually disclosed — never from raw records — and reasons
over it (`synthesis_mode` ∈ `none | rules | llm | hybrid`). The grounded arm
consults `protocol/interactions.py`, a curated interaction table standing in for
DrugBank/RxNorm. This is also the standalone CDSS baseline.

### 1.5 Other corrections

| Defect | Location | Correction |
|---|---|---|
| `"procedure"` matched the JSON *key* `planned_procedure`, so the warfarin bleeding rule fired on every query including routine checkups | `llm/provider.py` L63–75 | the declared procedure is parsed out of the context and tested for invasiveness |
| CKD rule matched the literal planted values `"48"`, `"58"` | `llm/provider.py` | trends are computed from the rendered laboratory series |
| `alert_raised` events logged the finding type in `query_type` and omitted the tier, contaminating both distributions in the privacy report | `agents/core.py` L688–695 | tier passed, `query_type` left empty |
| an unreachable Ollama server silently returned Mock output under the LLM's label | `llm/provider.py` `create_llm_provider` | raises `ProviderUnavailable`; `is_available()` refuses model substitution |
| the requesting department's own record was never included, so one-limb-local conflicts were unassemblable | `agents/core.py` | a local, non-boundary-crossing query is issued and excluded from the disclosure ledger |
| a failing agent aborted the whole request | `agents/core.py` | per-agent exception handling; `coverage` reported per scenario |
| every baseline lab was drawn inside the reference range with `is_abnormal=False`, making the flag a perfect indicator of a planted pattern | `data/generator.py` L351–365 | benign out-of-range values are drawn for ~18 % of baseline results |
| templates assigned round-robin (`i % len(TEMPLATES)`), so seeds varied names but not case mix — which is why every R0 standard deviation was exactly 0 | `data/generator.py` L471 | seed-dependent sampling |
| `metformin_renal_failure` planted no renal impairment; `beta_blocker_respiratory` planted a cardioselective agent and no airway disease | `data/generator.py` | repaired in `data/hard_cases.py::TEMPLATE_REPAIRS` |
| consent tokens were generated and never validated | `protocol/consent.py` | HMAC-bound, single-use, expiring; validation optional and evaluated in E8/A4 |

---

## 2. New evaluation material

`data/hard_cases.py`

* **10 drug-matched negative controls.** Each reuses a positive template's drugs
  in a safe arrangement (warfarin + paracetamol; ACE + thiazide without an
  NSAID; metformin with normal eGFR; a discontinued NSAID; a resolved AKI). A
  detector keying on drug names fires on these; one reasoning about the
  combination does not.
* **3 ambiguous cases** with "either answer defensible" adjudication, scored
  separately and excluded from precision and recall.
* **A held-out family** of 6 conflicts and 3 patterns, written after the
  prompts, routing rules and interaction table were frozen.
* **Free-text notes** with abbreviations, negations and hedging. In ~15 % of
  positives one limb of the conflict is documented *only* in a note, so it is
  recoverable at Tier 3 alone.
* **Record corruption**: missing fields, cross-departmental contradictions,
  stale prescriptions, resolved diagnoses, brand/generic duplicates.

Negative controls are audited against the interaction table and any incidental
finding is removed, so a false positive is genuinely a false positive. R0
counted clinically correct alerts on such patients as errors.

---

## 3. Experiments

| ID | Question | Command |
|---|---|---|
| E9 | how much did the R0 query context leak? | `--experiments e9` |
| E1 | tier vs utility vs measured exposure | `--experiments e1` |
| E2 | consent restriction 0–100 %, 10 graphs per level | `--experiments e2` |
| E3 | seed variance and held-out generalisation | `--experiments e3` |
| E4 | ablation matrix | `--experiments e4` |
| E5 | centralized LLM, conventional CDSS, direct retrieval | `--experiments e5` |
| E6 | backend matrix | `--backend-matrix ... --experiments e6` |
| E7 | patients × departments × concurrency | `--experiments e7` |
| E8 | threat model A1–A5 with and without mitigations | `--experiments e8` |

---

## 4. Reproducing the reported numbers

```bash
cd medagentnet
pip install pyyaml requests

# 1. harness self-test, no model server required (~90 s)
python run_r1.py --provider mock --patients 200 --tag validation

# 2. the reported run
ollama serve &
ollama pull llama3.1:8b
python run_r1.py --provider ollama --model llama3.1:8b --patients 200 \
                 --tag reported --experiments e9 e1 e2 e3 e4 e5 e7 e8

# 3. backend matrix (pull each model first)
python run_r1.py --provider ollama --model llama3.1:8b \
                 --backend-matrix llama3.1:8b qwen2.5:7b mistral:7b phi3:mini \
                 --include-mock-in-matrix --experiments e6 --tag backends
```

Each run writes `data/results_r1/<tag>_<timestamp>/results.json` and
`tables.tex`. **The manuscript tables are generated from `tables.tex`, not
transcribed.** After a run, replace the corresponding table blocks in the
manuscript with the generated ones.

Runtime with LLaMA 3.1 8B on a single consumer GPU is roughly 6–9 hours for the
full suite at 200 patients; E2 and E4 dominate. Reduce with `--patients 100` or
by selecting individual experiments.

---

## 5. What the results say

Taken together the R1 results support a narrower and better-evidenced claim than
R0 made:

1. **The coordination layer is what does the work.** Removing cross-departmental
   synthesis takes F1 from 0.99 to 0.29; removing orchestration entirely takes it
   to 0.16. This is the claim R0 asserted but could not support, because the
   leaked context did the coordination's job.
2. **Federation costs little accuracy.** MedAgentNet is statistically
   indistinguishable from a conventional CDSS with the record fully aggregated,
   while never centralising the record.
3. **Relevance routing buys efficiency, not accuracy.** Broadcast routing scores
   identically; the difference is in query volume and therefore in exposure. R0
   implied an accuracy benefit that is not there.
4. **The consent–utility relationship is roughly linear, not graceful.** Detection
   falls steadily with the fraction of revoked pairs, and targeted revocation of
   the pairs carrying the evidence is far more damaging than random revocation of
   the same number. R0's "no accuracy loss up to 60 % revocation" was an artefact
   of the revocation bug.
5. **Tier 2 is the right default and Tier 3 buys nothing here.** Tier 1 discloses
   nothing measurable and detects almost nothing; Tier 2 discloses ~69 % of the
   responding department's identifiable facts and reaches full utility; Tier 3
   adds exposure without adding utility on structured records.
6. **A query budget does not bound reconstruction; the tier does.** Rate-limiting
   cut traffic four-fold with no change in what was ultimately learned, because
   repeated queries at one tier return substantially the same content.
7. **Corroboration bounds independent compromise, not collusion.** Two agents
   emitting identical fabricated text satisfy the requirement.
