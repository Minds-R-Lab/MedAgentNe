"""
MedAgentNet - Revised simulation runner (R1)
============================================

Differences from ``simulation/runner.py``:

* queries are built by ``simulation/scenarios.py`` and carry no ground truth;
* every cohort (positive, matched negative, ambiguous, clean control) is run
  through the same query path with the same context schema;
* the false-positive denominator is every negative in the cohort, not a fixed
  sample of twenty;
* agents and orchestrator expose ablation switches;
* scenarios can be executed concurrently, so throughput and latency percentiles
  under load can be measured rather than inferred from a sequential loop;
* per-response disclosure is recorded for the leakage metric.
"""
from __future__ import annotations

import os
import time
import json
import random
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import yaml

from protocol.models import PatientRecord
from protocol.consent import ConsentManager, AuditTrail
from agents.core import DepartmentAgent, OrchestratorAgent
from llm.provider import BaseLLMProvider, create_llm_provider
from data.generator_hard import (
    HardCaseGenerator, COHORT_CONFLICT, COHORT_PATTERN, COHORT_DISTRACTOR,
    COHORT_AMBIGUOUS, COHORT_CONTROL,
)
from simulation import scenarios as scen

logger = logging.getLogger("medagentnet.runner_v2")


class SilentAudit(AuditTrail):
    """Audit trail that keeps entries in memory only.

    The R0 audit file grew to 63 MB over the reported experiments because every
    event was appended to disk. For sweeps we keep the same in-memory records
    (the privacy report is unchanged) without the I/O.
    """

    def __init__(self):
        self.log_file = None
        self.entries = []
        self._lock = threading.Lock()

    def log(self, **kwargs):
        from protocol.models import AuditEntry
        entry = AuditEntry(**{k: v for k, v in kwargs.items()
                              if k in AuditEntry.__dataclass_fields__})
        if kwargs.get("data_fields_shared") is not None:
            entry.data_fields_shared = kwargs["data_fields_shared"]
        with self._lock:
            self.entries.append(entry)
        return entry


class HardRunner:
    """Runs the R1 evaluation."""

    def __init__(self,
                 config_dir: str = "config",
                 llm_provider: Optional[BaseLLMProvider] = None,
                 seed: int = 42,
                 num_patients: int = 100,
                 use_heldout: bool = False,
                 cohort_mix: Optional[dict] = None,
                 noise_rates: Optional[dict] = None,
                 # architecture switches
                 routing_mode: str = "relevance",
                 synthesis_mode: str = "hybrid",
                 enforce_consent: bool = True,
                 enforce_tiers: bool = True,
                 structured_output: bool = True,
                 freetext_fallback: bool = True,
                 strict_context: bool = True,
                 validate_tokens: bool = False,
                 query_budget: int = 0,
                 corroborate_critical: bool = False,
                 n_departments: Optional[int] = None,
                 persist_audit: bool = False,
                 concurrency: int = 1):
        self.config_dir = config_dir
        self.seed = seed
        self.num_patients = num_patients
        self.use_heldout = use_heldout
        self.cohort_mix = cohort_mix
        self.noise_rates = noise_rates
        self.rng = random.Random(seed ^ 0x5EED)

        with open(os.path.join(config_dir, "settings.yaml")) as f:
            self.settings = yaml.safe_load(f)
        with open(os.path.join(config_dir, "departments.yaml")) as f:
            self.dept_config = yaml.safe_load(f)["departments"]

        # Optionally restrict the federation size, for the agent-count sweep.
        if n_departments:
            keep = ["general_practice", "laboratory"]
            others = [d for d in self.dept_config if d not in keep]
            keep = (keep + others)[:max(2, n_departments)]
            self.dept_config = {k: v for k, v in self.dept_config.items() if k in keep}

        self.llm = llm_provider or create_llm_provider(self.settings.get("llm", {}))

        self.consent = ConsentManager(
            default_policy=self.settings.get("privacy", {}).get("consent_default", "opt_in"),
            emergency_override=self.settings.get("privacy", {}).get(
                "emergency_override_enabled", True),
        )
        self.audit = (AuditTrail(log_file=self.settings.get("logging", {}).get(
            "audit_file", "data/audit_trail.jsonl")) if persist_audit else SilentAudit())

        self.department_agents = {
            dept_id: DepartmentAgent(
                department_id=dept_id, department_config=cfg,
                llm=self.llm, audit=self.audit,
                enforce_tiers=enforce_tiers,
                structured_output=structured_output,
                freetext_fallback=freetext_fallback,
                strict_context=strict_context,
            )
            for dept_id, cfg in self.dept_config.items()
        }

        self.orchestrator = OrchestratorAgent(
            department_agents=self.department_agents,
            llm=self.llm, consent_manager=self.consent, audit=self.audit,
            dept_config=self.dept_config,
            routing_mode=routing_mode,
            synthesis_mode=synthesis_mode,
            enforce_consent=enforce_consent,
            validate_tokens=validate_tokens,
            query_budget=query_budget,
            corroborate_critical=corroborate_critical,
        )

        # Default degree of parallelism for run(). Scenarios are independent,
        # so this is a throughput knob only: results are reassembled in spec
        # order, so a run at any concurrency is identical to a sequential one.
        # It matters when the backend is a model server that can serve several
        # requests at once (set OLLAMA_NUM_PARALLEL to match).
        self.concurrency = max(1, int(concurrency))

        self.patients: list[PatientRecord] = []
        self.specs: list[scen.ScenarioSpec] = []
        self.results: list[dict] = []
        self.failed_scenarios: list[dict] = []

    # ── data ─────────────────────────────────────────────────────────────

    def generate(self) -> list[PatientRecord]:
        gen = HardCaseGenerator(
            config_dir=self.config_dir, seed=self.seed,
            cohort_mix=self.cohort_mix, noise_rates=self.noise_rates,
            use_heldout=self.use_heldout,
        )
        # Keep the cohort inside the (possibly reduced) federation.
        gen.departments = list(self.dept_config.keys())
        gen.dept_config = self.dept_config
        self.patients = gen.generate(self.num_patients)

        for p in self.patients:
            p.departments = [d for d in p.departments if d in self.dept_config] \
                            or ["general_practice"]
            self.consent.register_patient(p.patient_id, p.departments)
            for dept in p.departments:
                if dept in self.department_agents:
                    self.department_agents[dept].load_patient_data(p)
        self.cohort_index = gen.save_patients(
            self.patients, output_dir=f"data/patients_hard/seed{self.seed}"
        )
        return self.patients

    def build_scenarios(self) -> list[scen.ScenarioSpec]:
        specs = []
        for p in self.patients:
            cohort = getattr(p, "cohort", COHORT_CONTROL)
            if cohort in (COHORT_CONFLICT, COHORT_DISTRACTOR, COHORT_AMBIGUOUS):
                specs.append(scen.build_safety_scenario(p, self.rng))
            elif cohort == COHORT_PATTERN:
                specs.append(scen.build_pattern_scenario(p, self.rng))
            else:
                # Controls provide negatives for BOTH tasks.
                specs.append(scen.build_safety_scenario(p, self.rng))
                specs.append(scen.build_pattern_scenario(p, self.rng))
        self.specs = specs
        return specs

    # ── execution ────────────────────────────────────────────────────────

    def _run_one(self, spec: scen.ScenarioSpec,
                 force_tier: Optional[int] = None) -> dict:
        t0 = time.time()
        out = self.orchestrator.process_request(
            requesting_dept=spec.requesting_department,
            patient_id=spec.patient.patient_id,
            clinical_context=spec.clinical_context,
            query_type=spec.query_type,
            is_emergency=spec.is_emergency,
            force_disclosure_tier=force_tier,
        )
        elapsed = time.time() - t0
        return {
            "scenario_name": spec.scenario_name,
            "cohort": spec.cohort,
            "patient_id": spec.patient.patient_id,
            "requesting_department": spec.requesting_department,
            "query_type": spec.query_type,
            "is_emergency": spec.is_emergency,
            "clinical_context": spec.clinical_context,
            "ground_truth": spec.ground_truth,
            "record_quality_flags": list(spec.patient.record_quality_flags),
            "num_alerts": len(out["alerts"]),
            "alerts": [a.to_dict() for a in out["alerts"]],
            "num_responses": len(out["responses"]),
            "response_summaries": [
                {"department": r.source_agent, "summary": r.summary,
                 "risk_flags": r.risk_flags, "tier": int(r.disclosure_tier)}
                for r in out["responses"]
            ],
            "privacy_report": out["privacy_report"],
            "elapsed_seconds": round(elapsed, 3),
        }

    def run(self, force_tier: Optional[int] = None,
            concurrency: Optional[int] = None) -> list[dict]:
        if not self.patients:
            self.generate()
        if not self.specs:
            self.build_scenarios()

        concurrency = self.concurrency if concurrency is None else max(1, int(concurrency))
        self.orchestrator.reset_counters()
        wall_start = time.time()

        total = len(self.specs)
        every = max(20, total // 8)
        done = [0]
        failed = []
        lock = threading.Lock()

        def _guarded(spec, tier):
            """Run one scenario; never let a single failure kill the run.

            A model backend can return a shape nothing else produces, and over a
            multi-hour sweep that will happen at least once. Losing an entire
            experiment to it is worse than losing one scenario, so failures are
            recorded and reported rather than raised. They are excluded from
            scoring: a harness failure is not evidence that the system missed a
            finding.
            """
            try:
                return self._run_one(spec, tier)
            except Exception as e:
                with lock:
                    failed.append({"scenario": spec.scenario_name,
                                   "patient": spec.patient.patient_id,
                                   "error": f"{type(e).__name__}: {e}"})
                logger.warning(f"    scenario failed ({spec.scenario_name}): "
                               f"{type(e).__name__}: {e}")
                return None

        def _tick():
            """Progress heartbeat, so a multi-hour run shows it is alive."""
            with lock:
                done[0] += 1
                n = done[0]
            if n % every and n != total:
                return
            el = time.time() - wall_start
            rate = n / el if el else 0
            eta = (total - n) / rate if rate else 0
            logger.info(f"    {n}/{total} scenarios  {rate:.1f}/s  "
                        f"elapsed {el/60:.1f}m  eta {eta/60:.1f}m")

        if concurrency <= 1:
            self.results = []
            for spec in self.specs:
                r = _guarded(spec, force_tier)
                if r is not None:
                    self.results.append(r)
                _tick()
        else:
            results = [None] * total
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = {pool.submit(_guarded, s, force_tier): i
                           for i, s in enumerate(self.specs)}
                for fut in as_completed(futures):
                    results[futures[fut]] = fut.result()
                    _tick()
            self.results = [r for r in results if r is not None]

        self.wall_seconds = time.time() - wall_start
        self.failed_scenarios = failed
        if failed:
            pct = 100.0 * len(failed) / max(1, total)
            level = logger.error if pct > 1.0 else logger.warning
            level(f"    {len(failed)}/{total} scenarios failed ({pct:.1f}%). "
                  f"They are excluded from scoring. First: {failed[0]['error']}")
        return self.results

    # ── reporting helpers ────────────────────────────────────────────────

    def disclosure_records(self) -> list[dict]:
        out = []
        for agent in self.department_agents.values():
            out.extend(agent.disclosure_log)
        return out

    def patient_index(self) -> dict:
        return {p.patient_id: p for p in self.patients}

    def throughput_report(self) -> dict:
        n = len(self.results)
        wall = getattr(self, "wall_seconds", 0.0) or 1e-9
        n_queries = sum(r["num_responses"] for r in self.results)
        return {
            "scenarios": n,
            "wall_seconds": round(wall, 2),
            "scenarios_per_second": round(n / wall, 4),
            "inter_agent_queries": n_queries,
            "queries_per_scenario": round(n_queries / n, 2) if n else 0,
            "queries_per_second": round(n_queries / wall, 4),
            "departments": len(self.department_agents),
        }

    def operational_report(self) -> dict:
        return {
            "rejected_tokens": self.orchestrator.rejected_tokens,
            "budget_blocks": self.orchestrator.budget_blocks,
            "suppressed_uncorroborated": self.orchestrator.suppressed_uncorroborated,
            "audit_events": len(self.audit.entries),
            "privacy_report": self.audit.get_privacy_report(),
        }
