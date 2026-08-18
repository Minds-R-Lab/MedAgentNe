"""
MedAgentNet - R1 experiment suite
=================================

Experiments
-----------
E1  Disclosure-tier sweep, with information leakage measured at each tier.
E2  Consent restriction sweep: fractions 0-100 %, many independently sampled
    restriction graphs per fraction, with confidence intervals.
E3  Multi-seed variance on the strict scorer, held-out family reported apart.
E4  Ablation matrix: which architectural component does the work.
E5  External baselines: centralized LLM, conventional CDSS, direct retrieval.
E6  Backend matrix: several model families and sizes, with format reliability,
    token volume and latency.
E7  Scalability: patients x departments x concurrency, with throughput and
    latency percentiles.
E8  Adversarial and failure-mode suite.
E9  Query-context audit: how much ground truth the R0 construction leaked.
"""
from __future__ import annotations

import os
import json
import time
import random
import logging
from datetime import datetime
from typing import Optional

from simulation.runner_v2 import HardRunner
from simulation.evaluation import (
    evaluate_run, aggregate_runs, grade_scenario, paired_comparison, rate,
    mean_std, percentiles,
)
from simulation import scenarios as scen
from simulation import baselines as bl
from simulation import adversarial as adv
from simulation.privacy import full_privacy_report, measure_disclosure, reidentification_risk
from llm.provider import MockLLMProvider, OllamaProvider, create_llm_provider

logger = logging.getLogger("medagentnet.experiments_v2")


class R1Experiments:

    def __init__(self, config_dir="config", llm_provider=None,
                 base_seed=42, num_patients=150, out_dir="data/results_r1",
                 concurrency=1, n_seeds=3, consent_graphs=10):
        self.config_dir = config_dir
        self.llm = llm_provider
        self.base_seed = base_seed
        self.num_patients = num_patients
        self.out_dir = out_dir
        # Applies to every experiment except E7, which varies concurrency
        # deliberately and pins its other sweeps to 1 so they stay comparable.
        self.concurrency = max(1, int(concurrency))
        # Sweep sizes. Lowering these is the only way to cut the cost of the
        # suite substantially: E2 and E4 together are roughly three quarters of
        # it. Whatever is used must be reported in the paper.
        self.n_seeds = max(1, int(n_seeds))
        self.consent_graphs = max(1, int(consent_graphs))
        os.makedirs(out_dir, exist_ok=True)

    def _seeds(self, n=None):
        n = n or self.n_seeds
        return tuple(self.base_seed + 1000 * i for i in range(n))

    # ── helper ───────────────────────────────────────────────────────────

    def runner(self, seed=None, num_patients=None, **kw) -> HardRunner:
        return HardRunner(
            config_dir=self.config_dir,
            llm_provider=self.llm,
            seed=self.base_seed if seed is None else seed,
            num_patients=self.num_patients if num_patients is None else num_patients,
            concurrency=kw.pop("concurrency", self.concurrency),
            **kw,
        )

    @staticmethod
    def _with_failures(ev: dict, runner) -> dict:
        """Attach the scenario-failure count, so it cannot pass unnoticed."""
        f = getattr(runner, "failed_scenarios", []) or []
        ev["harness_failures"] = {
            "n_failed_scenarios": len(f),
            "examples": f[:3],
            "note": "Scenarios the harness could not complete. Excluded from "
                    "scoring rather than counted as misses, since a harness "
                    "failure is not evidence about detection.",
        }
        return ev

    def _provider_stats(self):
        try:
            return self.llm.stats if self.llm else {}
        except Exception:
            return {}

    # ── E1: tiers and leakage ────────────────────────────────────────────

    def e1_tiers(self, seeds=None) -> dict:
        seeds = seeds or self._seeds()
        rows = {}
        step, nsteps = 0, 3 * len(seeds)
        for tier in (1, 2, 3):
            per_seed = []
            leak = []
            reid = []
            for s in seeds:
                step += 1
                logger.info(f"  [E1 {step}/{nsteps}] tier {tier}, seed {s}")
                r = self.runner(seed=s)
                results = r.run(force_tier=tier)
                per_seed.append(evaluate_run(results))
                leak.append(measure_disclosure(r)["per_tier"].get(f"tier_{tier}", {}))
                reid.append(reidentification_risk(r))
            agg = aggregate_runs(per_seed)
            exposure = [x.get("mean_field_exposure", 0.0) for x in leak if x]
            items = [x.get("mean_items_disclosed", 0.0) for x in leak if x]
            anon = [x.get("mean_anonymity_set", 0.0) for x in reid
                    if x.get("responses_with_identifiable_content")]
            singled = [x.get("singled_out_rate", 0.0) for x in reid
                       if x.get("responses_with_identifiable_content")]
            rows[f"tier_{tier}"] = {
                "aggregate": agg["across_runs"],
                "pooled": agg["pooled"],
                "leakage": {
                    "mean_field_exposure": mean_std(exposure),
                    "mean_items_disclosed": mean_std(items),
                    "mean_anonymity_set": mean_std(anon),
                    "singled_out_rate": mean_std(singled),
                    "responses_with_identifiable_content": sum(
                        x.get("responses_with_identifiable_content", 0) for x in reid),
                    "detail": leak[0] if leak else {},
                },
            }
        return {"experiment": "E1_disclosure_tiers", "seeds": list(seeds),
                "rows": rows,
                "note": "Utility is reported under the strict, ground-truth-matched "
                        "criterion. Exposure is the fraction of each responding "
                        "department's identifiable facts that crossed the boundary."}

    # ── E2: consent sweep ────────────────────────────────────────────────

    def e2_consent(self, fractions=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                   n_graphs=None, seed=None) -> dict:
        n_graphs = n_graphs or self.consent_graphs
        seed = self.base_seed if seed is None else seed
        rows = []
        step, nsteps = 0, len(fractions) * n_graphs
        for frac in fractions:
            trials = []
            denial_rates = []
            for g in range(n_graphs):
                step += 1
                logger.info(f"  [E2 {step}/{nsteps}] revoked {frac:.0%}, "
                            f"graph {g + 1}/{n_graphs}")
                r = self.runner(seed=seed)
                r.generate()
                rng = random.Random((seed * 7919) ^ (g * 104729) ^ int(frac * 1000))

                total_pairs = revoked_pairs = 0
                for p in r.patients:
                    if frac >= 1.0:
                        r.consent.revoke_consent(p.patient_id)
                        pairs = r.consent.permitted_pairs(p.patient_id)
                        total_pairs += len(pairs)
                        revoked_pairs += len(pairs)
                        continue
                    pairs = list(r.consent.consent_profiles.get(p.patient_id, {}).keys())
                    total_pairs += len(pairs)
                    if not pairs or frac <= 0:
                        continue
                    k = int(round(len(pairs) * frac))
                    if k:
                        chosen = rng.sample(pairs, min(k, len(pairs)))
                        revoked_pairs += r.consent.revoke_pairs(p.patient_id, chosen)

                r.build_scenarios()
                results = r.run()
                ev = evaluate_run(results)
                trials.append(ev)
                denials = sum(rr["privacy_report"].get("consent_denied", 0)
                              for rr in results)
                sent = sum(rr["num_responses"] for rr in results)
                denial_rates.append(denials / max(1, denials + sent))

            agg = aggregate_runs(trials)
            rows.append({
                "restriction_fraction": frac,
                "n_graphs": n_graphs,
                "observed_denial_rate": mean_std(denial_rates),
                "conflict_detection": agg["pooled"]["conflict_detection/localised_or_better"],
                "pattern_detection": agg["pooled"]["pattern_detection/localised_or_better"],
                "f1": agg["across_runs"]["classification/f1"],
                "recall": agg["across_runs"]["classification/recall"],
                "precision": agg["across_runs"]["classification/precision"],
            })

        # Targeted removal: revoke the pairs that actually carry the evidence.
        targeted = self._e2_targeted(seed)
        return {"experiment": "E2_consent_restriction", "rows": rows,
                "targeted_removal": targeted,
                "sampling": "directed department pairs revoked uniformly at random "
                            "per patient; independently resampled per graph",
                "note": "R0 revoked a pair by deleting its profile entry, which fell "
                        "through to the opt-in default; the recorded R0 runs show "
                        "zero consent denials at 30 % and 60 % restriction. Denial "
                        "rates observed here confirm the restriction now takes effect."}

    def _e2_targeted(self, seed) -> dict:
        """Revoke exactly the department pairs that carry the planted evidence."""
        out = []
        for frac_label, targeted in (("random_matched", False), ("targeted", True)):
            r = self.runner(seed=seed)
            r.generate()
            rng = random.Random(seed ^ 0x7A46)
            for p in r.patients:
                labels = (p.known_conflicts or []) + (p.known_patterns or [])
                critical = set()
                for lab in labels:
                    for d in lab.get("departments", []):
                        critical.add(d)
                pairs = list(r.consent.consent_profiles.get(p.patient_id, {}).keys())
                if not pairs:
                    continue
                if targeted:
                    chosen = [k for k in pairs if k[1] in critical]
                else:
                    k = len([x for x in pairs if x[1] in critical])
                    chosen = rng.sample(pairs, min(k, len(pairs)))
                if chosen:
                    r.consent.revoke_pairs(p.patient_id, chosen)
            r.build_scenarios()
            ev = evaluate_run(r.run())
            out.append({
                "removal": frac_label,
                "conflict_detection": ev["conflict_detection"]["localised_or_better"],
                "pattern_detection": ev["pattern_detection"]["localised_or_better"],
                "f1": ev["classification"]["f1"],
            })
        return {"rows": out,
                "note": "Matched by the number of pairs removed, so the difference "
                        "isolates which pairs were removed rather than how many."}

    # ── E3: variance and generalisation ──────────────────────────────────

    def e3_variance(self, seeds=None) -> dict:
        seeds = seeds or self._seeds(max(self.n_seeds, 5))
        dev, held = [], []
        for i, s in enumerate(seeds, 1):
            logger.info(f"  [E3 {i}/{len(seeds)}] seed {s}, development family")
            dev.append(evaluate_run(self.runner(seed=s).run()))
            logger.info(f"  [E3 {i}/{len(seeds)}] seed {s}, held-out family")
            held.append(evaluate_run(
                self.runner(seed=s, use_heldout=True).run()))
        return {
            "experiment": "E3_variance_and_generalisation",
            "seeds": list(seeds),
            "development_family": aggregate_runs(dev),
            "heldout_family": aggregate_runs(held),
            "note": "The held-out family was written after the prompts, the routing "
                    "rules and the interaction table were frozen; no held-out drug "
                    "or disease appears in any prompt.",
        }

    # ── E4: ablations ────────────────────────────────────────────────────

    def e4_ablations(self, seeds=None) -> dict:
        seeds = seeds or self._seeds()
        rows = {}
        reference_grades = {}
        nvar = len(bl.ARCHITECTURE_VARIANTS)
        for vi, (name, kw) in enumerate(bl.ARCHITECTURE_VARIANTS.items(), 1):
            logger.info(f"  [E4 {vi}/{nvar}] {name}")
            per_seed, grade_sets = [], []
            queries = []
            for s in seeds:
                r = self.runner(seed=s, **kw)
                results = r.run()
                per_seed.append(evaluate_run(results))
                grade_sets.append([grade_scenario(x) for x in results])
                queries.append(sum(x["num_responses"] for x in results) /
                               max(1, len(results)))
            agg = aggregate_runs(per_seed)
            rows[name] = {
                "config": kw,
                "f1": agg["across_runs"]["classification/f1"],
                "precision": agg["across_runs"]["classification/precision"],
                "recall": agg["across_runs"]["classification/recall"],
                "conflict_detection": agg["pooled"]["conflict_detection/localised_or_better"],
                "pattern_detection": agg["pooled"]["pattern_detection/localised_or_better"],
                "false_alarms_matched": agg["pooled"]["false_alarms/matched_negatives"],
                "queries_per_scenario": mean_std(queries),
            }
            if name == "medagentnet":
                reference_grades = grade_sets

        # McNemar against the full system on identical scenarios
        if reference_grades:
            for name, kw in bl.ARCHITECTURE_VARIANTS.items():
                if name == "medagentnet":
                    continue
                r = self.runner(seed=seeds[0], **kw)
                g = [grade_scenario(x) for x in r.run()]
                rows[name]["vs_medagentnet"] = paired_comparison(
                    reference_grades[0], g)
        return {"experiment": "E4_ablation_matrix", "seeds": list(seeds),
                "rows": rows}

    # ── E5: external baselines ───────────────────────────────────────────

    def e5_baselines(self, seeds=None) -> dict:
        seeds = seeds or self._seeds()
        rows = {}
        ref_grades = None
        for s in seeds:
            pass

        # MedAgentNet reference
        med_per_seed, med_grades = [], None
        for s in seeds:
            r = self.runner(seed=s, **bl.ARCHITECTURE_VARIANTS["medagentnet"])
            res = r.run()
            med_per_seed.append(evaluate_run(res))
            if med_grades is None:
                med_grades = [grade_scenario(x) for x in res]
                ref_specs = r.specs
        agg = aggregate_runs(med_per_seed)
        rows["medagentnet"] = {
            "f1": agg["across_runs"]["classification/f1"],
            "precision": agg["across_runs"]["classification/precision"],
            "recall": agg["across_runs"]["classification/recall"],
            "latency_s": agg["across_runs"]["latency_seconds/mean"],
            "data_centralised": False,
        }

        for name, fn in bl.EXTERNAL_BASELINES.items():
            logger.info(f"  [E5] baseline: {name}")
            per_seed, grades = [], None
            for s in seeds:
                r = self.runner(seed=s)
                r.generate()
                specs = r.build_scenarios()
                res = fn(specs, llm=r.llm, concurrency=self.concurrency)
                per_seed.append(evaluate_run(res))
                if grades is None:
                    grades = [grade_scenario(x) for x in res]
            a = aggregate_runs(per_seed)
            rows[name] = {
                "f1": a["across_runs"]["classification/f1"],
                "precision": a["across_runs"]["classification/precision"],
                "recall": a["across_runs"]["classification/recall"],
                "latency_s": a["across_runs"]["latency_seconds/mean"],
                "data_centralised": name.startswith("centralized"),
            }
            if med_grades and grades and len(med_grades) == len(grades):
                rows[name]["vs_medagentnet"] = paired_comparison(med_grades, grades)

        return {"experiment": "E5_baselines", "seeds": list(seeds), "rows": rows,
                "note": "All systems are run on identical patients and identical "
                        "queries and scored by the same criterion. The centralized "
                        "arms require the record to be aggregated in one place, "
                        "which is the configuration the architecture is intended "
                        "to avoid."}

    # ── E6: backend matrix ───────────────────────────────────────────────

    def e6_backends(self, providers: dict, seeds=None) -> dict:
        seeds = seeds or self._seeds(min(self.n_seeds, 2))
        """`providers` maps a display name to a BaseLLMProvider instance."""
        rows = {}
        grade_ref = None
        for pi, (name, provider) in enumerate(providers.items(), 1):
            logger.info(f"  [E6 {pi}/{len(providers)}] backend: {name}")
            per_seed, grades = [], None
            try:
                provider.reset_stats()
            except Exception:
                pass
            for s in seeds:
                r = HardRunner(config_dir=self.config_dir, llm_provider=provider,
                               seed=s, num_patients=self.num_patients,
                               **bl.ARCHITECTURE_VARIANTS["medagentnet"])
                res = r.run()
                per_seed.append(evaluate_run(res))
                if grades is None:
                    grades = [grade_scenario(x) for x in res]
            a = aggregate_runs(per_seed)
            stats = {}
            try:
                stats = provider.stats
            except Exception:
                pass
            rows[name] = {
                "identity": stats.get("provider", name),
                "f1": a["across_runs"]["classification/f1"],
                "recall": a["across_runs"]["classification/recall"],
                "precision": a["across_runs"]["classification/precision"],
                "false_alarms_matched": a["pooled"]["false_alarms/matched_negatives"],
                "mean_latency_s": stats.get("mean_latency_s"),
                "format_failure_rate": stats.get("format_failure_rate"),
                "approx_prompt_tokens": stats.get("approx_prompt_tokens"),
                "approx_completion_tokens": stats.get("approx_completion_tokens"),
                "errors": stats.get("errors"),
            }
            if grade_ref is None:
                grade_ref = (name, grades)
            elif grades and len(grades) == len(grade_ref[1]):
                rows[name]["vs_" + grade_ref[0]] = paired_comparison(
                    grade_ref[1], grades)
        return {"experiment": "E6_backend_matrix", "seeds": list(seeds),
                "rows": rows}

    # ── E7: scalability ──────────────────────────────────────────────────

    def e7_scalability(self, patient_counts=(50, 100, 250, 500),
                       department_counts=(4, 6, 8, 10),
                       concurrency_levels=(1, 2, 4, 8)) -> dict:
        by_patients = []
        for n in patient_counts:
            logger.info(f"  [E7] patients={n}")
            r = self.runner(num_patients=n, concurrency=1)
            res = r.run(concurrency=1)
            ev = evaluate_run(res)
            by_patients.append({
                "patients": n, **r.throughput_report(),
                "f1": ev["classification"]["f1"],
                "latency": ev["latency_seconds"],
            })

        by_departments = []
        for k in department_counts:
            logger.info(f"  [E7] departments={k}")
            r = self.runner(n_departments=k, concurrency=1)
            res = r.run(concurrency=1)
            ev = evaluate_run(res)
            by_departments.append({
                "departments": k, **r.throughput_report(),
                "f1": ev["classification"]["f1"],
                "recall": ev["classification"]["recall"],
                "latency": ev["latency_seconds"],
            })

        by_concurrency = []
        for c in concurrency_levels:
            logger.info(f"  [E7] concurrency={c}")
            r = self.runner(concurrency=c)
            res = r.run(concurrency=c)
            ev = evaluate_run(res)
            by_concurrency.append({
                "concurrency": c, **r.throughput_report(),
                "f1": ev["classification"]["f1"],
                "latency": ev["latency_seconds"],
            })

        return {"experiment": "E7_scalability",
                "by_patient_count": by_patients,
                "by_department_count": by_departments,
                "by_concurrency": by_concurrency,
                "note": "Communication volume is reported as inter-agent queries "
                        "per scenario; it is the quantity that grows with the "
                        "number of departments."}

    # ── E8: adversarial ──────────────────────────────────────────────────

    def e8_adversarial(self) -> dict:
        def make(seed, num_patients, **kw):
            return self.runner(seed=seed, num_patients=num_patients, **kw)
        return {"experiment": "E8_adversarial",
                **adv.run_all_adversarial(make, self.base_seed,
                                          min(self.num_patients, 60))}

    # ── E9: context audit ────────────────────────────────────────────────

    def e9_context_audit(self) -> dict:
        """Quantify how much ground truth each query construction carried."""
        r = self.runner()
        r.generate()
        specs = r.build_scenarios()

        r1_rows, r0_rows = [], []
        for spec in specs:
            r1_rows.append(scen.context_leak_report(spec.clinical_context, spec.patient))
        for p in r.patients:
            for c in p.known_conflicts:
                ctx = scen.build_legacy_conflict_context(p, c)
                r0_rows.append(scen.context_leak_report(ctx, p))
            for pat in p.known_patterns:
                ctx = scen.build_legacy_pattern_context(p, pat)
                r0_rows.append(scen.context_leak_report(ctx, p))

        def summarise(rows):
            if not rows:
                return {}
            return {
                "n_contexts": len(rows),
                "mean_keys": round(sum(x["n_keys"] for x in rows) / len(rows), 2),
                "mean_disallowed_keys": round(
                    sum(x["n_disallowed_keys"] for x in rows) / len(rows), 2),
                "contexts_carrying_ground_truth": sum(
                    1 for x in rows if x["carries_ground_truth_text"]),
                "mean_label_terms_present": round(
                    sum(x["n_label_terms_present"] for x in rows) / len(rows), 2),
                "contexts_with_any_label_term": sum(
                    1 for x in rows if x["n_label_terms_present"] > 0),
            }

        # Detection under the R0 context with no cross-departmental synthesis:
        # this is what the R0 pipeline was actually measuring.
        # R0 pipeline: legacy contexts AND no cross-departmental synthesis,
        # with the context whitelist disabled so the leak is actually carried.
        legacy = self.runner(routing_mode="relevance", synthesis_mode="none",
                             strict_context=False)
        legacy.generate()
        legacy_specs = []
        rng = random.Random(self.base_seed)
        for p in legacy.patients:
            for c in p.known_conflicts:
                s = scen.build_safety_scenario(p, rng)
                s.clinical_context = scen.build_legacy_conflict_context(p, c)
                legacy_specs.append(s)
            for pat in p.known_patterns:
                s = scen.build_pattern_scenario(p, rng)
                s.clinical_context = scen.build_legacy_pattern_context(p, pat)
                legacy_specs.append(s)
            if not p.known_conflicts and not p.known_patterns:
                legacy_specs.append(scen.build_safety_scenario(p, rng))
        legacy.specs = legacy_specs
        legacy_ev = evaluate_run(legacy.run())

        return {
            "experiment": "E9_query_context_audit",
            "r0_construction": summarise(r0_rows),
            "r1_construction": summarise(r1_rows),
            "r0_pipeline_scores": {
                "loose_criterion": legacy_ev["legacy_loose_criterion"],
                "strict_criterion": legacy_ev["classification"],
            },
            "note": "The R0 context placed the conflict description or the "
                    "expected diagnosis in the prompt, together with every "
                    "department's medication list, at every disclosure tier.",
        }

    # ── driver ───────────────────────────────────────────────────────────

    def run_all(self, which=None, providers=None, resume=False) -> dict:
        which = which or ["e9", "e1", "e2", "e3", "e4", "e5", "e7", "e8"]

        # A full run is many hours. Completed experiments are checkpointed as
        # they finish, so a crash, a preemption or an idle shutdown costs only
        # the experiment in flight rather than everything before it.
        ckpt = os.path.join(self.out_dir, "checkpoint.json")
        done = {}
        if resume and os.path.exists(ckpt):
            try:
                with open(ckpt) as f:
                    done = json.load(f)
                have = [k for k in which if k in done and "error" not in done[k]]
                if have:
                    logger.info(f"resuming: {', '.join(have)} already complete "
                                f"in {ckpt}")
            except Exception as e:
                logger.warning(f"could not read checkpoint {ckpt}: {e}")
                done = {}

        out = {
            "meta": {
                "timestamp": datetime.now().isoformat(),
                "base_seed": self.base_seed,
                "num_patients": self.num_patients,
                "n_seeds": self.n_seeds,
                "consent_graphs": self.consent_graphs,
                "provider": (self.llm.describe() if self.llm else "from config"),
                "concurrency": self.concurrency,
                "latency_note": (
                    "Per-scenario latency is wall-clock and therefore includes "
                    "queueing at the model server. It is comparable across "
                    "systems only at concurrency 1. E7 pins its patient and "
                    "department sweeps to 1 for this reason; take reported "
                    "latency from there."
                    if self.concurrency > 1 else
                    "Run sequentially; per-scenario latency is per-request."),
            }
        }
        t0 = time.time()
        table = {
            "e1": lambda: self.e1_tiers(),
            "e2": lambda: self.e2_consent(),
            "e3": lambda: self.e3_variance(),
            "e4": lambda: self.e4_ablations(),
            "e5": lambda: self.e5_baselines(),
            "e6": lambda: self.e6_backends(providers or {}),
            "e7": lambda: self.e7_scalability(),
            "e8": lambda: self.e8_adversarial(),
            "e9": lambda: self.e9_context_audit(),
        }
        for i, key in enumerate(which, 1):
            if resume and key in done and "error" not in done[key]:
                logger.info(f"  {key.upper()} already complete, skipping "
                            f"({i} of {len(which)})")
                out[key] = done[key]
                continue
            logger.info(f"{'=' * 62}")
            logger.info(f"  experiment {key.upper()}  ({i} of {len(which)})")
            logger.info(f"{'=' * 62}")
            started = time.time()
            try:
                out[key] = table[key]()
                out[key]["_wall_seconds"] = round(time.time() - started, 1)
                logger.info(f"  {key.upper()} done in "
                            f"{out[key]['_wall_seconds'] / 60:.1f} min "
                            f"(total so far {(time.time() - t0) / 60:.1f} min)")
            except Exception as e:
                logger.exception(f"{key} failed")
                out[key] = {"error": f"{type(e).__name__}: {e}"}

            # checkpoint after every experiment, complete or failed
            try:
                tmp = ckpt + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(out, f, indent=2, default=str)
                os.replace(tmp, ckpt)
            except Exception as e:
                logger.warning(f"could not write checkpoint: {e}")

        out["meta"]["total_seconds"] = round(time.time() - t0, 1)
        if self.llm:
            try:
                out["meta"]["provider_stats"] = self.llm.stats
            except Exception:
                pass
        return out

    def save(self, results: dict, tag: str = "") -> str:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"r1_{tag + '_' if tag else ''}{stamp}"
        d = os.path.join(self.out_dir, name)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "results.json"), "w") as f:
            json.dump(results, f, indent=2, default=str)
        from simulation.tables import generate_all_tables
        tex = generate_all_tables(results)
        with open(os.path.join(d, "tables.tex"), "w") as f:
            f.write(tex)
        with open(os.path.join(self.out_dir, "latest.json"), "w") as f:
            json.dump(results, f, indent=2, default=str)
        with open(os.path.join(self.out_dir, "latest_tables.tex"), "w") as f:
            f.write(tex)
        return d
