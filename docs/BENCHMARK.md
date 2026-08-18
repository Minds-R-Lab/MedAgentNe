# Benchmark design and scoring

The first version of this benchmark could not be failed. This document describes
how it was rebuilt so that it can be, and states plainly what each change is
for. Implementation: `data/generator_hard.py`, `data/hard_cases.py`,
`simulation/scenarios.py`, `simulation/evaluation.py`.

## What was wrong before

**The query context contained the answer.** Each conflict query carried a
`reason` field copied from the ground-truth template description — for example
*"ACE inhibitor + NSAID + Diuretic = Acute Kidney Injury risk"* — and each
pattern query carried an `expected` field holding the target diagnosis verbatim,
along with a pre-computed trend label. The context was rendered into every
agent's prompt at every disclosure tier, including Tier 1.

**The context also carried every department's medication list**, under a comment
reading *"so agents can detect them"*. Each agent could see the whole drug list
in its own prompt, so no cross-departmental exchange was required to detect a
cross-departmental interaction. This dissolved the task the architecture exists
to perform.

**Scoring accepted any alert.** A conflict counted as detected whenever any
alert existed whose type was not a sentinel value, with no check that it
corresponded to the planted conflict. Pattern scoring passed if any of roughly
twenty generic words appeared anywhere in any agent's free text, so an agent
reporting "elevated" on a chronic kidney disease case scored as having detected
it.

**Template assignment was round-robin by patient index**, so the random seed
varied names and baseline draws but left the multiset of planted scenarios
essentially fixed. That is why every reported standard deviation was exactly
zero.

Experiment E9 quantifies the first two: every R0 context carried ground-truth
text, with a mean of 9.28 fields no clinician would have supplied and 2.4 terms
drawn from the label. The R1 construction carries 0 and 0.11.

## Cohorts

Fractions are of patients. Clean controls are queried for both tasks and so
contribute two scenarios each; every other cohort contributes one.

| Cohort | Share | Purpose |
|---|---|---|
| Conflict | 30% | One of eight planted medication conflicts, limbs distributed across the departments it spans |
| Pattern | 20% | One of three multi-system diagnostic patterns, evidence distributed across departments |
| Matched negative | 25% | **The discriminating cohort** — see below |
| Ambiguous | 5% | Adjudicated defensible either way; excluded from precision and recall |
| Clean control | 20% | No planted finding, queried with an identical context schema |

Two R0 templates were repaired because the stated conflict did not exist in the
data: the metformin case asserted renal impairment without planting any, and the
beta-blocker case used a cardioselective agent and planted no airway disease.

### Matched negatives

Each reuses the medications of a positive template in an arrangement that is not
clinically actionable. A detector keying on drug names fires on all of these; one
reasoning about the combination fires on none.

| Control | Mirrors | Why it is safe |
|---|---|---|
| `warfarin_paracetamol_safe` | anticoagulant + NSAID | Paracetamol is the recommended analgesic; no NSAID present |
| `ace_diuretic_no_nsaid` | triple whammy | ACE + thiazide is a standard antihypertensive pairing; the third limb is absent |
| `metformin_normal_renal` | metformin + renal failure | eGFR normal and stable |
| `cardioselective_bb_no_airway_disease` | beta-blocker + airway disease | Metoprolol is cardioselective; no airway disease documented |
| `timolol_alone` | topical + systemic beta-blockade | No systemic agent |
| `methotrexate_with_folate` | methotrexate + NSAID | Standard regimen; no NSAID |
| `carbamazepine_alone` | carbamazepine + warfarin | Monotherapy; nothing to interact with |
| `warfarin_dental_checkup_no_procedure` | warfarin + extraction | Scale and polish; no invasive procedure planned |
| `discontinued_nsaid_on_warfarin` | anticoagulant + NSAID | The NSAID was stopped four months ago and is recorded inactive |
| `resolved_ckd_normal_function` | CKD progression | Historical AKI that fully resolved; eGFR recovered |

**Negative-control audit.** After generation, every negative is checked against
the same interaction knowledge base used for scoring, and any incidental
interaction arising from baseline prescribing is removed; a case that cannot be
made genuinely negative is dropped. Without this step a patient intended as a
control can acquire a real warfarin–NSAID interaction by chance, and a
clinically correct alert would be scored as a false positive. R0 had no such
step. `tests/test_architecture.py::test_negative_controls_are_genuinely_negative`
enforces it.

## Making the records hard to read

A benchmark of clean records measures something other than clinical reading.
Each record is subjected to corruption operators:

| Operator | Rate | Effect |
|---|---|---|
| Missing fields | 0.20 | A dose, frequency or date is simply absent |
| Contradiction | 0.15 | The same drug in two departments with different dose or conflicting active status |
| Stale prescription | 0.20 | An inactive drug that would trigger an alert if treated as current |
| Resolved diagnosis | 0.15 | A resolved condition that would change the assessment if treated as active |
| Note-only limb | 0.15 | One limb of a conflict documented **only** in free text — recoverable at Tier 3 and not below |
| Brand/generic duplicate | 0.15 | The same drug under two names |
| Benign out-of-range result | 0.18 | So an out-of-range flag no longer identifies a planted pattern, as it did in R0 where every baseline result was drawn inside the reference interval |

Free-text notes carry abbreviations, hedging and negation.

## Held-out scenario family

Six conflicts and three patterns authored **after** the prompts, routing rules
and interaction knowledge base were frozen, covering drugs and diseases that
appear in no prompt: SSRI + NSAID, simvastatin + macrolide, ACE inhibitor +
potassium-sparing diuretic, levothyroxine + calcium, allopurinol + azathioprine,
digoxin + amiodarone; occult gastrointestinal blood loss, drug-induced liver
injury, decompensating heart failure. Results on this family are reported
separately throughout. Conflict detection falls from 99.7% to 90.0%.

## Query construction

Three properties are enforced, and each is a regression test.

1. **No label content.** Queries are built from the patient's own record and a
   department-level procedure vocabulary. No field of the ground truth is read.
2. **Schema invariance.** Every cohort produces a context with an identical key
   set, so positives and negatives cannot be separated by prompt shape before any
   clinical content is read. In R0 a positive carried a `reason` key and a set of
   `current_med_*` keys that negatives lacked.
3. **No cross-department pre-aggregation.** The context never carries data from a
   department other than the one initiating the encounter.

The encounter — which department the patient is attending and for what — is part
of the clinical situation rather than the label: a warfarin/extraction conflict
only exists if an extraction is planned. Each matched negative is drawn with the
same encounter type as the positive it mirrors, and clean controls draw from the
same pooled distribution, so the encounter carries no discriminative
information.

As a structural safeguard the department agent holds a whitelist
(`DepartmentAgent.ALLOWED_CONTEXT_KEYS`) and discards anything else before
prompt construction, so the R0 defect cannot recur unnoticed.

## Scoring

For a positive case with planted finding *F*:

| Level | Name | Criterion |
|---|---|---|
| L0 | miss | No alert at or above the warning level |
| L1 | flagged | An alert was raised but does not identify *F* |
| L2 | localised | The alert names the medications, or for a pattern the clinical entities, constituting *F* |
| L3 | explained | The alert additionally names the mechanism or consequence of *F* |

**Detection rate means L2 or better.** A Tier 1 flag-only response can reach L1
by construction and can never reach L2, which is what makes the tier comparison
informative rather than circular.

Precision, recall, specificity and F1 use L2 for true positives and count any
actionable alert on a negative as a false positive. The false-positive
denominator is every negative in the cohort, not a fixed sample of twenty as in
R0 — where the reported "1.7% false positive rate with 2.9% standard deviation"
was a single false positive out of sixty routine scenarios, and the standard
deviation was that of three points {0, 0, 0.05}.

All proportions carry Wilson 95% intervals, which do not misbehave near zero and
one. Systems are compared on identical scenarios with McNemar's test.

**The R0 criterion is computed and reported alongside the strict one**, as
`legacy_loose_criterion`, so the two can be compared directly.
