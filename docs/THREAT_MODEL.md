# Threat model

This is the threat model the architecture is assessed against. It is stated so
that the privacy claims can be checked rather than taken on trust, and so that
the boundaries of those claims are explicit. `medagentnet/simulation/adversarial.py`
implements it; experiment E8 evaluates each adversary with and without its
mitigation.

## Terminology

MedAgentNet provides **architectural privacy controls**, not formal privacy
guarantees. A formal guarantee is a statement about a mechanism's output
distribution against an adversary with stated capabilities (differential
privacy), or a statement resting on a cryptographic hardness assumption (secure
multi-party computation, homomorphic encryption). This system supplies neither.
It supplies application-layer controls whose effect is measurable on a given
workload, which is a weaker and different kind of claim.

## Assets

1. The records held in each departmental store.
2. The disclosure policy and the per-patient consent profile.
3. The audit trail.
4. **The integrity of the clinical alerts.** This is an asset in its own right:
   a fabricated alert can cause a harmful clinical action just as a disclosure
   can harm a patient's interests.

## Trust assumptions

- The transport layer is mutually authenticated and encrypted.
- The consent service and the audit store are trusted and correctly implemented.
- Department agents are authenticated but **not** assumed honest.
- The language-model backend is assumed manipulable through content it is asked
  to read. This is a documented property of retrieval-augmented systems.

## Adversaries

### A1 — Curious authorised department

Holds legitimate access for some queries and wants more of the record than any
single response yields. Issues many well-formed queries and differences the
answers.

**Measured.** Twelve queries per patient reconstructed 67% of the patient's
cross-departmental inventory on average, 92% in the worst case.

**Mitigation evaluated: a per-pair query budget — and it does not work.** It cut
traffic four-fold and left reconstruction unchanged, because repeated queries at
the same tier return substantially the same content. Rate limiting bounds cost,
not disclosure. What bounds this adversary is the **disclosure tier**:
restricting the requester to Tier 1 reduced reconstruction to zero.

### A2 — Compromised department agent

Returns fabricated findings, or inflates severity, to provoke a clinical action.

**Measured.** Two compromised departments affected every scenario they touched.

**Mitigation: corroboration.** A critical alert supported by exactly one
department is downgraded and marked single-source. This removed all fabricated
critical alerts under *independent* compromise.

**Where it fails.** Two agents that collude and emit identical text corroborate
one another, and all fabricated criticals survive. Corroboration bounds
independent compromise, not collusion. Provenance signing that binds a finding
to the record entries supporting it would be the missing defence; it is not
implemented.

### A3 — Content-level attacker

Places instructions in free text that will enter an agent's prompt — a scanned
referral letter, a patient-reported note, an external portal transcript —
attempting to override the disclosure policy.

**Mitigations evaluated:** schema-validated responses, so a reply that abandons
the JSON contract is discarded rather than rescued by keyword matching; and
withholding narrative below Tier 3, which removes the injection carrier from
most exchanges.

**Note on the reported figures.** A rule-based backend does not follow
instructions and is structurally immune, so the deterministic arm is a negative
control that establishes the harness detects nothing where nothing is there.
Only a generative backend can be meaningfully attacked here.

### A4 — Replay adversary

Captures a consent token from an authorised exchange and reuses it after consent
is withdrawn, or for a different patient, department pair or tier.

**Measured.** The submitted implementation generated tokens and never validated
them, so all four replay attempts succeeded at the token layer, including reuse
for a different patient.

**Mitigation.** Tokens are HMAC-bound to patient, directed department pair and
tier, single-use, and expire after 60 seconds. Only one attempt is then accepted
at the token layer — reuse before expiry by the legitimate holder — and the
independent consent check denies the exchange. Neither control alone is
sufficient; the layering is the point.

### A5 — Availability adversary

Renders the orchestrator or a subset of agents unreachable.

**Measured.** Utility and coverage fall together and the system returns a
partial assessment with an explicit statement of which departments did not
respond: F1 1.00 at full coverage, 0.57 with two of ten agents unavailable, 0.46
with four or six. Removing the orchestrator reduces the system to
single-department operation, F1 0.24.

**The safety-relevant property is the labelling.** An unlabelled partial answer
in a clinical setting is worse than no answer.

## Out of scope

- Compromise of the consent service or the audit store.
- Compromise of the underlying record systems.
- Side channels in the transport layer.
- Extraction of the model itself.

A deployment must address these by other means. The architecture does not bear
on them and no claim is made that it does.

## Human in the loop

Every alert is advisory. The system does not alter prescriptions, place orders
or modify records. A clinician acts on an alert or does not. This is a control
against A2 and A3 as well as against ordinary model error, and it bounds the
harm from any single failure to whatever a clinician would do with an incorrect
piece of advice presented alongside its stated evidence.

## Known gaps

- Corroboration does not bound colluding agents.
- A query budget does not bound reconstruction.
- Cryptographic log integrity is not implemented; the audit trail is append-only
  in memory and on disk, and contains sensitive information requiring the same
  protection as a clinical store.
- mTLS is specified but not exercised: the reported experiments run in a single
  process.
- Consent is modelled as a map from directed department pairs to a maximum tier.
  Real consent has temporal validity, purpose restrictions, delegated authority,
  capacity considerations and jurisdiction-specific withdrawal requirements.
