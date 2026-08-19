#!/usr/bin/env python3
"""
Where do the false alarms come from?
====================================

Every negative control in the R1 run raises an alert: specificity is 0.000
across E1, E3 and E5. Both LLM-mediated arms (MedAgentNet hybrid, centralized
single agent) sit at precision ~0.38-0.40, while the grounded rule engine on
the aggregated record sits at precision 1.000. That points at the synthesis
layer rather than at the federation, but E4 is the experiment that would settle
it and E4 is many hours away.

This script settles it in about half an hour. It holds patients, queries, seed
and scorer fixed and varies only the orchestrator's synthesis mode:

    hybrid   grounded knowledge base + language model   (the configuration run)
    rules    grounded knowledge base only
    llm      language model only
    none     relay department findings, no synthesis    (the R0 behaviour)

If `rules` recovers specificity while `llm` does not, the false alarms are the
language model's and the paper's operating point should be the grounded
configuration. Run from the repository's medagentnet/ directory.
"""
from __future__ import annotations

import sys
import json
import time
import logging
import argparse

from llm.provider import create_llm_provider, ProviderUnavailable
from simulation.runner_v2 import HardRunner
from simulation.evaluation import evaluate_run

MODES = ("rules", "llm", "hybrid", "none")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="ollama")
    ap.add_argument("--model", default="llama3.1:8b")
    ap.add_argument("--patients", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--num-ctx", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0.0 for reported runs: the mode comparison is only "
                         "meaningful if sampling noise is not part of it")
    ap.add_argument("--request-timeout", type=int, default=600)
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    ap.add_argument("--out", default="data/results_r1/specificity_diagnosis.json")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(name)-28s | %(message)s",
        datefmt="%H:%M:%S", stream=sys.stdout)
    logging.getLogger("medagentnet.agents").setLevel(logging.WARNING)

    try:
        llm = create_llm_provider(
            {"provider": args.provider,
             args.provider: {"model": args.model,
                             "num_ctx": args.num_ctx,
                             "temperature": args.temperature,
                             "request_timeout": args.request_timeout}},
            strict=True)
    except ProviderUnavailable as e:
        print(f"ERROR: {e}")
        return 2

    ok, detail = llm.preflight()
    if not ok:
        print(f"ERROR: backend did not answer a test request: {detail}")
        return 2
    print(f"preflight ok : {detail}\n")

    rows, t0 = {}, time.time()
    for mode in args.modes:
        print("=" * 66)
        print(f"  synthesis_mode = {mode}")
        print("=" * 66)
        started = time.time()
        r = HardRunner(config_dir="config", llm_provider=llm,
                       seed=args.seed, num_patients=args.patients,
                       concurrency=args.concurrency,
                       routing_mode="relevance", synthesis_mode=mode)
        ev = evaluate_run(r.run())
        c = ev["classification"]
        rows[mode] = {
            "tp": c["tp"], "fp": c["fp"], "tn": c["tn"], "fn": c["fn"],
            "precision": c["precision"], "recall": c["recall"],
            "f1": c["f1"], "specificity": c["specificity"],
            "conflict_localised": ev["conflict_detection"]["localised_or_better"]["rate"],
            "pattern_localised": ev["pattern_detection"]["localised_or_better"]["rate"],
            "false_alarm_matched_negatives": ev["false_alarms"]["matched_negatives"]["rate"],
            "false_alarm_clean_controls": ev["false_alarms"]["clean_controls"]["rate"],
            "mean_alerts_per_scenario": round(
                sum(x["num_alerts"] for x in r.results) / max(1, len(r.results)), 2)
            if getattr(r, "results", None) else None,
            "wall_minutes": round((time.time() - started) / 60, 1),
        }
        print(json.dumps(rows[mode], indent=2), "\n")

    out = {"patients": args.patients, "seed": args.seed,
           "backend": llm.describe(),
           "total_minutes": round((time.time() - t0) / 60, 1),
           "rows": rows}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print("=" * 78)
    print(f"{'mode':10} {'prec':>7} {'recall':>7} {'F1':>7} {'spec':>7} "
          f"{'FA-neg':>7} {'conf':>7} {'patt':>7}")
    print("-" * 78)
    for m, v in rows.items():
        print(f"{m:10} {v['precision']:7.3f} {v['recall']:7.3f} {v['f1']:7.3f} "
              f"{v['specificity']:7.3f} {v['false_alarm_matched_negatives']:7.3f} "
              f"{v['conflict_localised']:7.3f} {v['pattern_localised']:7.3f}")
    print("=" * 78)
    print(f"written to {args.out}  ({out['total_minutes']} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
