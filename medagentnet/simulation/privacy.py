"""
MedAgentNet - Quantitative information-leakage measurement (R1)
===============================================================

R0 defined "information leakage" in the evaluation-framework section and never
reported a number for it. This module measures it.

Definitions
-----------
For a patient *p*, a responding department *d* holds a set of identifiable
clinical facts, its **PHI inventory**

    I(p, d) = medication names ∪ dose strings ∪ condition names
              ∪ laboratory test names ∪ prescription dates

A response transmits some subset of those facts across the department boundary.
The **disclosed set** D(p, d, τ) is the subset of I(p, d) that appears verbatim
in the transmitted payload at tier τ. Two quantities are reported:

    field-level exposure     E_field(τ)  = |D| / |I|      averaged over responses
    item-count exposure      E_count(τ)  = |D|            averaged over responses

Neither depends on the language model's phrasing: an item counts as disclosed
only if the identifiable string itself crosses the boundary.

Beyond per-response exposure we report:

* **Cumulative reconstruction.** An adversary in one department who issues
  repeated queries accumulates disclosures. R(p, k) is the fraction of the
  patient's whole cross-departmental record reconstructable after *k* queries.
  This is the differencing attack the query budget is designed to bound.

* **Re-identification proxy.** The number of patients in the cohort consistent
  with the disclosed set, i.e. the anonymity-set size; k = 1 means the disclosed
  facts single the patient out.

* **Utility versus exposure.** F1 at each tier plotted against E_field at that
  tier, which is the privacy-utility curve the reviewers asked for.
"""
from __future__ import annotations

import re
from collections import defaultdict


def _norm(s) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def phi_inventory(patient, department: str = None) -> set:
    """The identifiable facts held for a patient (optionally in one department)."""
    items = set()
    for m in patient.medications:
        if department and m.department != department:
            continue
        items.add(("medication", _norm(m.name)))
        if m.dose:
            items.add(("dose", _norm(f"{m.name} {m.dose}")))
        if m.prescribed_date:
            items.add(("date", _norm(m.prescribed_date)))
    for c in patient.conditions:
        if department and c.department != department:
            continue
        items.add(("condition", _norm(c.name)))
    for l in patient.lab_results:
        if department and l.department != department:
            continue
        items.add(("lab_test", _norm(l.test_name)))
        items.add(("lab_value", _norm(f"{l.test_name} {l.value}")))
    return items


def _disclosed(items: set, blob: str) -> set:
    """Which inventory items appear verbatim in the transmitted payload."""
    out = set()
    for kind, value in items:
        if not value:
            continue
        if kind in ("dose", "lab_value"):
            # composite items: every component must be present
            parts = value.split()
            if all(p in blob for p in parts):
                out.add((kind, value))
        elif value in blob:
            out.add((kind, value))
    return out


def measure_disclosure(runner) -> dict:
    """Per-response and per-tier exposure over a completed run."""
    patients = runner.patient_index()
    records = runner.disclosure_records()

    per_tier = defaultdict(lambda: {
        "responses": 0, "disclosed_items": 0, "available_items": 0,
        "fractions": [], "by_kind": defaultdict(int),
    })
    per_response = []

    for rec in records:
        patient = patients.get(rec["patient_id"])
        if patient is None:
            continue
        inv = phi_inventory(patient, rec["responding_department"])
        if not inv:
            continue
        blob = _norm(rec["disclosed_text"])
        got = _disclosed(inv, blob)

        tier = rec["tier"]
        bucket = per_tier[tier]
        bucket["responses"] += 1
        bucket["disclosed_items"] += len(got)
        bucket["available_items"] += len(inv)
        bucket["fractions"].append(len(got) / len(inv))
        for kind, _ in got:
            bucket["by_kind"][kind] += 1

        per_response.append({
            "patient_id": rec["patient_id"],
            "from": rec["responding_department"],
            "to": rec["requesting_department"],
            "tier": tier,
            "available": len(inv),
            "disclosed": len(got),
            "fraction": round(len(got) / len(inv), 4),
        })

    summary = {}
    for tier, b in sorted(per_tier.items()):
        n = b["responses"]
        fr = b["fractions"]
        summary[f"tier_{tier}"] = {
            "responses": n,
            "mean_items_disclosed": round(b["disclosed_items"] / n, 3) if n else 0,
            "mean_items_available": round(b["available_items"] / n, 3) if n else 0,
            "mean_field_exposure": round(sum(fr) / len(fr), 4) if fr else 0.0,
            "max_field_exposure": round(max(fr), 4) if fr else 0.0,
            "zero_disclosure_responses": sum(1 for f in fr if f == 0.0),
            "by_field_kind": dict(b["by_kind"]),
        }
    return {"per_tier": summary, "n_responses": len(per_response),
            "per_response_sample": per_response[:200]}


def cumulative_reconstruction(runner, max_queries: int = 20) -> dict:
    """How much of a patient's record one department can accumulate.

    Groups disclosures by (patient, requesting department) and reports the
    fraction of the patient's full cross-departmental inventory recovered after
    each successive response.
    """
    patients = runner.patient_index()
    grouped = defaultdict(list)
    for rec in runner.disclosure_records():
        grouped[(rec["patient_id"], rec["requesting_department"])].append(rec)

    curves = defaultdict(list)
    finals = []
    for (pid, requester), recs in grouped.items():
        patient = patients.get(pid)
        if patient is None:
            continue
        full = phi_inventory(patient)
        if not full:
            continue
        acquired = set()
        trace = []
        for rec in recs[:max_queries]:
            acquired |= _disclosed(full, _norm(rec["disclosed_text"]))
            trace.append(len(acquired) / len(full))
        if not trace:
            continue
        # Carry the final value forward to the common horizon. Without this the
        # mean at large k is taken over only those pairs that happened to
        # receive many responses, which biases the curve upward and can make a
        # rate-limited arm appear to leak more than an unlimited one.
        last = trace[-1]
        for i in range(1, max_queries + 1):
            curves[i].append(trace[i - 1] if i <= len(trace) else last)
        finals.append(last)

    step = max(1, max_queries // 10)
    curve = {
        f"after_{k}_responses": round(sum(v) / len(v), 4)
        for k, v in sorted(curves.items())
        if v and (k % step == 0 or k == 1 or k == max_queries)
    }
    return {
        "reconstruction_curve": curve,
        "mean_final_reconstruction": round(sum(finals) / len(finals), 4)
        if finals else 0.0,
        "max_final_reconstruction": round(max(finals), 4) if finals else 0.0,
        "requester_patient_pairs": len(grouped),
    }


def reidentification_risk(runner) -> dict:
    """Anonymity-set size implied by what each response disclosed.

    For each response we take the disclosed medication and condition names and
    count how many patients in the cohort match all of them. An anonymity set of
    one means the disclosure singles the patient out within this cohort.
    """
    patients = list(runner.patient_index().values())
    if not patients:
        return {}

    profiles = []
    for p in patients:
        profiles.append({
            "meds": {_norm(m.name) for m in p.medications},
            "conds": {_norm(c.name) for c in p.conditions},
        })

    sizes = []
    for rec in runner.disclosure_records():
        patient = runner.patient_index().get(rec["patient_id"])
        if patient is None:
            continue
        blob = _norm(rec["disclosed_text"])
        meds = {_norm(m.name) for m in patient.medications
                if _norm(m.name) and _norm(m.name) in blob}
        conds = {_norm(c.name) for c in patient.conditions
                 if _norm(c.name) and _norm(c.name) in blob}
        if not meds and not conds:
            continue
        k = sum(1 for prof in profiles
                if meds <= prof["meds"] and conds <= prof["conds"])
        sizes.append(max(1, k))

    if not sizes:
        return {"responses_with_identifiable_content": 0}
    uniq = sum(1 for k in sizes if k == 1)
    return {
        "responses_with_identifiable_content": len(sizes),
        "mean_anonymity_set": round(sum(sizes) / len(sizes), 3),
        "min_anonymity_set": min(sizes),
        "singled_out_responses": uniq,
        "singled_out_rate": round(uniq / len(sizes), 4),
        "cohort_size": len(patients),
    }


def full_privacy_report(runner) -> dict:
    return {
        "disclosure": measure_disclosure(runner),
        "cumulative": cumulative_reconstruction(runner),
        "reidentification": reidentification_risk(runner),
    }
