# Notice

This is research software accompanying an academic manuscript. It is **not a
medical device**, has not been validated for clinical use, and must not be used
in the care of patients.

The evaluation in this repository is entirely synthetic. It is deliberately
harder than a naive benchmark — it includes drug-matched negative controls,
adjudicated ambiguous cases, missing fields, contradictory cross-departmental
documentation, stale prescriptions, resolved diagnoses, free-text notes, and a
held-out scenario family authored after the system was frozen — but generated
records are still generated, and the ways in which real clinical documentation
is difficult are not exhausted by the corruption operators implemented here.

The claims this repository supports are **architectural**, not clinical:

- that cross-departmental findings can be assembled without centralising records;
- that the coordination and disclosure machinery, rather than the language model,
  accounts for most of the resulting capability;
- that the information released at each disclosure tier can be measured.

No clinician has adjudicated any alert produced by this system. No prospective
study has been conducted. No claim of clinical benefit is made or supported.

The interaction knowledge base in `medagentnet/protocol/interactions.py` is a
small hand-curated table written to cover the mechanisms this benchmark
contains. It stands in for a maintained clinical source such as DrugBank or
RxNorm and is not suitable for any other purpose.

Patient names, identifiers, records and clinical narratives in this repository
are synthetic and refer to no real person.
