#!/usr/bin/env python3
"""
Why does the grounded arm alert on negative controls?
=====================================================

The same knowledge base scores precision 1.000 on a complete record
(``baselines.run_centralized_rules``) and 0.594 on the assembled, tier-limited
evidence. The status fix did not move the matched-negative false-alarm rate at
all -- 0.333 before, 0.333 after -- so the cause is something other than
discontinued medications surviving the boundary, and guessing again is not
worth another hour of GPU time.

This runs only the negative controls, in grounded mode, and for each false
alarm records three things side by side:

  * which rule fired on the assembled evidence,
  * the evidence the orchestrator assembled from what departments disclosed,
  * the evidence the centralized comparator builds from the raw record,
    and which rules fire on that.

The difference between the last two is the answer. If a rule fires on the
complete record too, the benchmark planted a real interaction in a scenario
labelled negative, and the defect is in the generator rather than the system.

It also counts how often department agents actually emitted an ``active`` or
``status`` field, which says whether the schema change reached the model at all.

Run from the repository's medagentnet/ directory:
    PYTHONPATH=. python -u ../scripts/trace_false_alarms.py
"""
from __future__ import annotations

import os
import sys
import json
import logging
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.getcwd())

from llm.provider import create_llm_provider, ProviderUnavailable
from simulation.runner_v2 import HardRunner
from simulation.baselines import _evidence_from_record
from protocol.interactions import evaluate_rules, evaluate_patterns

# The generator labels cohorts "distractor" and "control"; the evaluation
# renames them "matched_negative" and "clean_control" for reporting. Import
# the constants so the two cannot drift apart again.
from data.generator_hard import COHORT_DISTRACTOR, COHORT_CONTROL

NEGATIVE_COHORTS = (COHORT_DISTRACTOR, COHORT_CONTROL)


def _ev_summary(ev):
    return {
        "drugs": sorted(ev.drugs),
        "categories": sorted(ev.categories),
        "conditions": sorted(c["name"] for c in ev.conditions),
        "labs": {k: [v for _, v in vs] for k, vs in sorted(ev.labs.items())},
        "procedure": ev.procedure,
    }


def _diff(fed, cen):
    out = {}
    for key in ("drugs", "categories", "conditions"):
        f, c = set(fed[key]), set(cen[key])
        if f - c:
            out[f"{key}_only_federated"] = sorted(f - c)
        if c - f:
            out[f"{key}_only_centralized"] = sorted(c - f)
    fl, cl = set(fed["labs"]), set(cen["labs"])
    if cl - fl:
        out["labs_missing_from_federated"] = sorted(cl - fl)
    if fl - cl:
        out["labs_only_federated"] = sorted(fl - cl)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="ollama")
    ap.add_argument("--model", default="llama3.1:8b")
    ap.add_argument("--patients", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--num-ctx", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--show", type=int, default=6)
    ap.add_argument("--out", default="data/results_r1/false_alarm_trace.json")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(name)-24s | %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)
    logging.getLogger("medagentnet.agents").setLevel(logging.WARNING)

    try:
        llm = create_llm_provider(
            {"provider": args.provider,
             args.provider: {"model": args.model, "num_ctx": args.num_ctx,
                             "temperature": args.temperature,
                             "request_timeout": 600}}, strict=True)
    except ProviderUnavailable as e:
        print(f"ERROR: {e}")
        return 2
    ok, detail = llm.preflight()
    if not ok:
        print(f"ERROR: backend did not answer: {detail}")
        return 2
    print(f"preflight ok : {detail}\n")

    r = HardRunner(config_dir="config", llm_provider=llm, seed=args.seed,
                   num_patients=args.patients, concurrency=args.concurrency,
                   routing_mode="relevance", synthesis_mode="rules")
    r.generate()
    specs = [s for s in r.build_scenarios() if s.cohort in NEGATIVE_COHORTS]
    if not specs:
        seen = sorted({s.cohort for s in r.specs})
        print(f"ERROR: no scenarios in cohorts {NEGATIVE_COHORTS}; "
              f"the generator emitted {seen}")
        return 2
    print(f"{len(specs)} negative controls\n")

    orch = r.orchestrator

    def one(spec):
        out = orch.process_request(
            requesting_dept=spec.requesting_department,
            patient_id=spec.patient.patient_id,
            clinical_context=spec.clinical_context,
            query_type=spec.query_type,
            is_emergency=spec.is_emergency)
        return spec, out

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        pairs = list(pool.map(one, specs))
    print("run complete\n")

    schema, traces = Counter(), []
    for spec, out in pairs:
        responses = out["responses"]
        for resp in responses:
            for item in resp.medications_reported:
                schema["med_with_status" if isinstance(item, dict) and
                       ("active" in item or "status" in item)
                       else "med_without_status"] += 1
            for item in resp.conditions_reported:
                schema["cond_with_status" if isinstance(item, dict) and
                       ("active" in item or "status" in item)
                       else "cond_without_status"] += 1

        if not out["alerts"]:
            continue

        fed = orch._assemble_evidence(spec.clinical_context, responses)
        cen = _evidence_from_record(
            spec.patient, spec.clinical_context.get("planned_procedure", ""))
        fed_s, cen_s = _ev_summary(fed), _ev_summary(cen)
        traces.append({
            "scenario": spec.scenario_name,
            "cohort": spec.cohort,
            "patient": spec.patient.patient_id,
            "query_type": spec.query_type,
            "alerts": [a.description[:160] for a in out["alerts"]],
            "rules_on_assembled": [h["id"] for h in evaluate_rules(fed)] +
                                  [h["id"] for h in evaluate_patterns(fed)],
            "rules_on_complete_record": [h["id"] for h in evaluate_rules(cen)] +
                                        [h["id"] for h in evaluate_patterns(cen)],
            "federated_evidence": fed_s,
            "centralized_evidence": cen_s,
            "diff": _diff(fed_s, cen_s),
            "record_quality_flags": list(spec.patient.record_quality_flags),
            "inactive_in_record": {
                "medications": [m.name for m in spec.patient.medications if not m.active],
                "conditions": [c.name for c in spec.patient.conditions if not c.active],
            },
            "raw_reported": [
                {"dept": resp.source_agent, "tier": int(resp.disclosure_tier),
                 "medications": resp.medications_reported,
                 "conditions": resp.conditions_reported}
                for resp in responses],
        })

    for t in traces[:args.show]:
        print("=" * 78)
        print(f"FALSE ALARM: {t['scenario']}  [{t['cohort']}]  {t['query_type']}")
        print("=" * 78)
        print(json.dumps(t, indent=2, default=str)[:4500])
        print()

    with open(args.out, "w") as f:
        json.dump({"schema_counts": dict(schema), "n_negatives": len(specs),
                   "n_false_alarms": len(traces), "traces": traces},
                  f, indent=2, default=str)

    print("=" * 78)
    print(f"negatives          : {len(specs)}")
    print(f"false alarms       : {len(traces)} "
          f"({len(traces)/max(1,len(specs)):.3f})")
    print(f"agent output shape : {dict(schema)}")
    print("  -> med_with_status == 0 means the model ignored the schema change")

    print("\nrules firing on the ASSEMBLED evidence:")
    for rid, n in Counter(x for t in traces for x in t["rules_on_assembled"]).most_common():
        print(f"  {rid:44} {n}")
    print("\nrules firing on the COMPLETE record (nonzero = the benchmark planted")
    print("a real interaction in a scenario it labelled negative):")
    c = Counter(x for t in traces for x in t["rules_on_complete_record"])
    for rid, n in c.most_common():
        print(f"  {rid:44} {n}")
    if not c:
        print("  (none)")

    print("\nmost common evidence differences:")
    d = Counter(k for t in traces for k in t["diff"])
    for k, n in d.most_common():
        print(f"  {k:44} {n}")
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
