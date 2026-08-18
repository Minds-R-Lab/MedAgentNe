"""
MedAgentNet - Comparison systems (R1)
=====================================

R0's only comparator was a rule-based provider swapped in behind the same
architecture, which therefore could not tell us whether the architecture did
anything. These baselines vary the architecture while holding the data, the
queries and the scorer fixed.

    centralized_llm     one agent with the entire cross-departmental record.
                        Upper bound on accuracy; the privacy configuration the
                        paper argues against.
    centralized_rules   a conventional CDSS: the same interaction knowledge base
                        applied to a fully aggregated record, no agents at all.
    single_department   the requesting department's agent alone. What a clinician
                        has today without any information exchange.
    broadcast_no_synth  every department is queried, responses are concatenated
                        and returned, with no cross-departmental reasoning.
                        This is the R0 pipeline once the leaked context is
                        removed.
    direct_retrieval    structured query against every department's store with
                        no reasoning layer at all: returns the union of records
                        and flags nothing. Measures how much of the task is
                        retrieval and how much is inference.
    medagentnet         the proposed system.

Every baseline consumes the same ``ScenarioSpec`` list and emits results in the
same shape, so ``simulation/evaluation.py`` scores them identically and
``paired_comparison`` can run McNemar tests on matched scenarios.
"""
from __future__ import annotations

import json
import time
from typing import Optional

from protocol.models import ConflictAlert, AlertLevel, DisclosureTier
from protocol.interactions import (
    ClinicalEvidence, evaluate_rules, evaluate_patterns,
)


def _result_shell(spec, alerts, responses, elapsed, extra=None):
    out = {
        "scenario_name": spec.scenario_name,
        "cohort": spec.cohort,
        "patient_id": spec.patient.patient_id,
        "requesting_department": spec.requesting_department,
        "query_type": spec.query_type,
        "is_emergency": spec.is_emergency,
        "clinical_context": spec.clinical_context,
        "ground_truth": spec.ground_truth,
        "record_quality_flags": list(spec.patient.record_quality_flags),
        "num_alerts": len(alerts),
        "alerts": [a.to_dict() for a in alerts],
        "num_responses": len(responses),
        "response_summaries": responses,
        "privacy_report": {},
        "elapsed_seconds": round(elapsed, 3),
    }
    if extra:
        out.update(extra)
    return out


def _evidence_from_record(patient, procedure: str,
                          departments: Optional[list] = None) -> ClinicalEvidence:
    """Evidence assembled directly from the raw record (centralized access)."""
    ev = ClinicalEvidence()
    ev.procedure = (procedure or "").lower()
    for m in patient.medications:
        if departments and m.department not in departments:
            continue
        if m.active:
            ev.add_drug(m.name, m.category, m.department)
    for c in patient.conditions:
        if departments and c.department not in departments:
            continue
        if c.active:
            ev.add_condition(c.name, c.department)
    for l in patient.lab_results:
        if departments and l.department not in departments:
            continue
        ev.add_lab(l.test_name, l.value, l.date, l.department)
    return ev


def _alerts_from_hits(patient_id, hits, is_emergency, alert_type):
    alerts = []
    for h in hits:
        level = (AlertLevel.CRITICAL.value if h["severity"] == "critical"
                 else AlertLevel.HIGH_RISK.value)
        alerts.append(ConflictAlert(
            patient_id=patient_id, alert_level=level, alert_type=alert_type,
            description=f"{h['label']}: {h['mechanism']}",
            involved_departments=h["involved_departments"],
            involved_medications=h["involved_medications"],
            recommendation="Review before proceeding.",
        ))
    return alerts


# ─────────────────────────────────────────────────────────────────────────────

def run_centralized_rules(specs, **_) -> list[dict]:
    """Conventional CDSS over a fully aggregated record."""
    out = []
    for spec in specs:
        t0 = time.time()
        ev = _evidence_from_record(
            spec.patient, spec.clinical_context.get("planned_procedure", ""))
        hits = evaluate_rules(ev)
        if spec.query_type == "LONG_PATTERN":
            hits += evaluate_patterns(ev)
        else:
            hits += [h for h in evaluate_patterns(ev) if h["severity"] == "critical"]
        alerts = _alerts_from_hits(spec.patient.patient_id, hits, spec.is_emergency,
                                   "centralized_rule")
        out.append(_result_shell(spec, alerts, [], time.time() - t0))
    return out


def run_centralized_llm(specs, llm, **_) -> list[dict]:
    """A single agent holding the entire cross-departmental record."""
    out = []
    system = (
        "You are a clinical decision support system with access to a patient's "
        "complete record across all hospital departments. Identify medication "
        "interactions, contraindications and multi-system diagnostic patterns. "
        "Treat DISCONTINUED medications and RESOLVED conditions as no longer in "
        "effect. "
        'Return JSON: {"alerts": [{"severity": "critical|high|moderate", '
        '"description": "...", "medications": [...], "departments": [...]}]}'
    )
    for spec in specs:
        t0 = time.time()
        p = spec.patient
        lines = [
            f"ENCOUNTER: {spec.clinical_context.get('planned_procedure','')}",
            f"REQUEST: {spec.clinical_context.get('query_reason','')}",
            "", "COMPLETE PATIENT RECORD:", "Medications:",
        ]
        for m in p.medications:
            lines.append(f"  - [{'ACTIVE' if m.active else 'DISCONTINUED'}] "
                         f"{m.name} ({m.category}) {m.dose} {m.frequency} "
                         f"[{m.department}]")
        lines.append("Conditions:")
        for c in p.conditions:
            lines.append(f"  - [{'ACTIVE' if c.active else 'RESOLVED'}] "
                         f"{c.name} ({c.code}) severity={c.severity} [{c.department}]")
        lines.append("Laboratory:")
        for l in sorted(p.lab_results, key=lambda x: x.date):
            lines.append(f"  - {l.date}: {l.test_name} = {l.value}{l.unit}")
        lines.append("Clinical notes:")
        for v in p.visits:
            if v.notes:
                lines.append(f"  - {v.date} [{v.department}]: {v.notes}")

        try:
            raw = llm.generate(system, "\n".join(lines))
        except Exception:
            raw = "{}"
        alerts = _parse_alert_json(raw, p.patient_id, "centralized_llm")
        out.append(_result_shell(spec, alerts, [], time.time() - t0))
    return out


def run_direct_retrieval(specs, **_) -> list[dict]:
    """Federated retrieval with no reasoning: return the records, flag nothing.

    Included to separate the retrieval half of the problem from the inference
    half. It should score near zero on detection by construction; the point is
    to show what an interoperability layer alone buys.
    """
    out = []
    for spec in specs:
        t0 = time.time()
        summaries = [
            {"department": d, "summary": "records returned", "risk_flags": [],
             "tier": int(DisclosureTier.CLINICAL_SUMMARY)}
            for d in spec.patient.departments
        ]
        out.append(_result_shell(spec, [], summaries, time.time() - t0))
    return out


def run_with_runner(specs, runner, force_tier=None, concurrency=1) -> list[dict]:
    """Run the given specs through a configured HardRunner (any ablation)."""
    runner.specs = specs
    return runner.run(force_tier=force_tier, concurrency=concurrency)


def _parse_alert_json(raw: str, patient_id: str, alert_type: str) -> list:
    data = None
    for attempt in (raw, raw[raw.find("{"):raw.rfind("}") + 1] if "{" in raw else ""):
        try:
            data = json.loads(attempt)
            break
        except Exception:
            continue
    if not isinstance(data, dict):
        return []

    # Accept either the orchestrator alert schema or the department-agent
    # finding schema, so that any backend able to answer one of the two can be
    # scored. Findings are mapped onto alerts.
    raw_alerts = data.get("alerts")
    if not raw_alerts:
        raw_alerts = []
        for f in data.get("findings", []) or []:
            if not isinstance(f, dict):
                continue
            if f.get("type") in ("no_conflict", "parse_error", "llm_error"):
                continue
            raw_alerts.append({
                "severity": f.get("severity", "moderate"),
                "description": f.get("description", ""),
                "medications": [
                    (m.get("name") if isinstance(m, dict) else str(m))
                    for m in data.get("medications_reported", []) or []
                ],
                "departments": [],
            })
        data = {"alerts": raw_alerts}

    alerts = []
    for a in data.get("alerts", []) or []:
        if not isinstance(a, dict):
            continue
        sev = str(a.get("severity", "moderate")).lower()
        level = {"critical": AlertLevel.CRITICAL.value,
                 "high": AlertLevel.HIGH_RISK.value,
                 "high_risk": AlertLevel.HIGH_RISK.value,
                 "moderate": AlertLevel.WARNING.value}.get(
                     sev, AlertLevel.WARNING.value)
        alerts.append(ConflictAlert(
            patient_id=patient_id, alert_level=level, alert_type=alert_type,
            description=str(a.get("description", ""))[:400],
            involved_medications=[str(m) for m in (a.get("medications") or [])],
            involved_departments=[str(d) for d in (a.get("departments") or [])],
            recommendation="Review before proceeding.",
        ))
    return alerts


# Configurations of the proposed system, expressed as HardRunner keyword sets.
ARCHITECTURE_VARIANTS = {
    "medagentnet": dict(routing_mode="relevance", synthesis_mode="hybrid",
                        enforce_consent=True, enforce_tiers=True,
                        structured_output=True, freetext_fallback=True),
    "medagentnet_grounded_only": dict(routing_mode="relevance", synthesis_mode="rules"),
    "medagentnet_llm_only": dict(routing_mode="relevance", synthesis_mode="llm"),
    "ablate_synthesis": dict(routing_mode="relevance", synthesis_mode="none"),
    "ablate_orchestration": dict(routing_mode="local", synthesis_mode="hybrid"),
    "ablate_relevance_routing": dict(routing_mode="broadcast", synthesis_mode="hybrid"),
    "ablate_tiers": dict(routing_mode="relevance", synthesis_mode="hybrid",
                         enforce_tiers=False),
    "ablate_consent": dict(routing_mode="relevance", synthesis_mode="hybrid",
                           enforce_consent=False),
    "ablate_structured_protocol": dict(routing_mode="relevance", synthesis_mode="hybrid",
                                       structured_output=False),
    "ablate_freetext_parser": dict(routing_mode="relevance", synthesis_mode="hybrid",
                                   freetext_fallback=False),
}
# Note: the R0 pipeline is NOT an entry here. Running it requires the R0 query
# construction as well as the missing synthesis step, and with R1 contexts
# strict_context=False is a no-op because there is nothing in the context to
# leak. Experiment E9 runs it correctly, with the legacy contexts.

EXTERNAL_BASELINES = {
    "centralized_rules": run_centralized_rules,
    "centralized_llm": run_centralized_llm,
    "direct_retrieval": run_direct_retrieval,
}
