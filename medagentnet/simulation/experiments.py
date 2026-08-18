"""
MedAgentNet - Comprehensive Experiments for Academic Paper
==========================================================
Runs structured experiments to generate results for the paper's
Results section: tier comparison, consent restriction impact,
multi-seed variance, and scalability analysis.
"""
import os
import json
import time
import math
import random
import logging
from datetime import datetime
from typing import Optional

import yaml

from simulation.runner import SimulationRunner
from data.generator import CONFLICT_TEMPLATES, PATTERN_TEMPLATES
from llm.provider import BaseLLMProvider, MockLLMProvider, create_llm_provider

logger = logging.getLogger("medagentnet.experiments")


def _mean(values: list) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    variance = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)


def _fmt(value: float, decimals: int = 1) -> str:
    """Format a percentage value for display."""
    return f"{value * 100:.{decimals}f}\\%"


def _fmt_ms(value: float) -> str:
    """Format seconds as milliseconds string."""
    return f"{value * 1000:.1f}"


class ExperimentRunner:
    """Orchestrates comprehensive experiments for academic paper Results section."""

    def __init__(self, config_dir: str = "config",
                 llm_provider: Optional[BaseLLMProvider] = None):
        self.config_dir = config_dir
        self.results_dir = "data/experiment_results"
        os.makedirs(self.results_dir, exist_ok=True)

        with open(os.path.join(config_dir, "settings.yaml")) as f:
            self.settings = yaml.safe_load(f)

        with open(os.path.join(config_dir, "departments.yaml")) as f:
            self.dept_config = yaml.safe_load(f)["departments"]

        self.base_seed = self.settings["simulation"]["random_seed"]
        self.base_num_patients = self.settings["simulation"]["num_patients"]

        # Optional LLM provider override — if set, all runners use this provider
        self.llm_provider = llm_provider

    # ────────────────────────────────────────────────────────────
    # Experiment 1: Disclosure Tier Comparison
    # ────────────────────────────────────────────────────────────
    def run_tier_comparison(self, num_patients: int = 100) -> dict:
        """Run conflict/pattern scenarios at each disclosure tier (1, 2, 3).

        Shows how detection accuracy changes with information disclosure level.
        """
        logger.info("=" * 60)
        logger.info("Experiment 1: Disclosure Tier Comparison")
        logger.info("=" * 60)

        results_by_tier = {}

        for tier in [1, 2, 3]:
            tier_name = {1: "Flag Only", 2: "Clinical Summary", 3: "Full Context"}[tier]
            logger.info(f"\n  Running Tier {tier} ({tier_name})...")

            # Create fresh runner for each tier
            self._set_config_patients(num_patients)
            runner = SimulationRunner(config_dir=self.config_dir, llm_provider=self.llm_provider)
            runner.generate_patients(num_patients)

            # Run conflict scenarios with forced tier
            conflict_results = []
            for patient in runner.patients:
                for conflict in patient.known_conflicts:
                    context = {
                        "planned_procedure": conflict.get("trigger_procedure", "evaluation"),
                        "relevant_categories": ["all"],
                        "reason": conflict.get("description", ""),
                    }
                    for med in patient.medications:
                        context[f"current_med_{med.name.lower().replace(' ', '_')}"] = med.name

                    result = runner.run_scenario(
                        patient=patient,
                        requesting_dept=conflict.get("trigger_department", "general_practice"),
                        clinical_context=context,
                        query_type="MED_CONFLICT",
                        is_emergency=conflict.get("alert_level") == "critical",
                        scenario_name=conflict.get("conflict_name", "unknown"),
                        force_disclosure_tier=tier,
                    )
                    conflict_results.append(result)

            # Run pattern scenarios with forced tier
            pattern_results = []
            for patient in runner.patients:
                for pattern in patient.known_patterns:
                    context = self._build_pattern_context(patient, pattern)
                    result = runner.run_scenario(
                        patient=patient,
                        requesting_dept="general_practice",
                        clinical_context=context,
                        query_type="LONG_PATTERN",
                        scenario_name=pattern.get("pattern_name", "unknown"),
                        force_disclosure_tier=tier,
                    )
                    pattern_results.append(result)

            # Evaluate
            eval_result = self._evaluate(runner, conflict_results, pattern_results)
            eval_result["tier"] = tier
            eval_result["tier_name"] = tier_name
            results_by_tier[tier] = eval_result

            logger.info(
                f"  Tier {tier}: conflicts={eval_result['conflict_detection']['detection_rate']:.2f}, "
                f"patterns={eval_result['pattern_detection']['detection_rate']:.2f}"
            )

        self._restore_config()

        return {
            "experiment": "tier_comparison",
            "num_patients": num_patients,
            "results_by_tier": results_by_tier,
        }

    # ────────────────────────────────────────────────────────────
    # Experiment 2: Consent Restriction Impact
    # ────────────────────────────────────────────────────────────
    def run_consent_restriction(self, num_patients: int = 100) -> dict:
        """Simulate three consent scenarios: full, partial, opt-out.

        Quantifies the privacy-utility tradeoff.
        """
        logger.info("=" * 60)
        logger.info("Experiment 2: Consent Restriction Impact")
        logger.info("=" * 60)

        scenarios = ["full", "partial_30", "partial_60", "opt_out"]
        results_by_scenario = {}

        for scenario in scenarios:
            logger.info(f"\n  Running consent scenario: {scenario}...")

            self._set_config_patients(num_patients)
            runner = SimulationRunner(config_dir=self.config_dir, llm_provider=self.llm_provider)
            runner.generate_patients(num_patients)

            # Apply consent restrictions
            if scenario != "full":
                rng = random.Random(self.base_seed)
                for patient in runner.patients:
                    if scenario == "opt_out":
                        runner.apply_consent_restriction(patient.patient_id, "opt_out")
                    elif scenario.startswith("partial"):
                        pct = int(scenario.split("_")[1]) / 100.0
                        # Get the actual consent profile pairs that exist
                        profile = runner.consent.consent_profiles.get(
                            patient.patient_id, {}
                        )
                        existing_pairs = list(profile.keys())
                        if not existing_pairs:
                            continue
                        # Revoke a fraction of them
                        n_revoke = max(1, int(len(existing_pairs) * pct))
                        revoke_pairs = rng.sample(
                            existing_pairs, min(n_revoke, len(existing_pairs))
                        )
                        # Directly remove from the profile
                        for pair in revoke_pairs:
                            if pair in runner.consent.consent_profiles.get(
                                patient.patient_id, {}
                            ):
                                del runner.consent.consent_profiles[patient.patient_id][pair]

            # Run conflict scenarios
            conflict_results = runner.run_all_conflict_scenarios()

            # Run pattern scenarios
            pattern_results = runner.run_all_pattern_scenarios()

            eval_result = self._evaluate(runner, conflict_results, pattern_results)
            eval_result["scenario"] = scenario
            results_by_scenario[scenario] = eval_result

            logger.info(
                f"  {scenario}: conflicts={eval_result['conflict_detection']['detection_rate']:.2f}, "
                f"patterns={eval_result['pattern_detection']['detection_rate']:.2f}"
            )

        self._restore_config()

        return {
            "experiment": "consent_restriction",
            "num_patients": num_patients,
            "results_by_scenario": results_by_scenario,
        }

    # ────────────────────────────────────────────────────────────
    # Experiment 3: Multi-Seed Variance
    # ────────────────────────────────────────────────────────────
    def run_multi_seed_variance(self, num_seeds: int = 5,
                                 num_patients: int = 100) -> dict:
        """Run full simulation with multiple random seeds.

        Reports mean ± std for all key metrics.
        """
        logger.info("=" * 60)
        logger.info(f"Experiment 3: Multi-Seed Variance ({num_seeds} seeds)")
        logger.info("=" * 60)

        runs = []

        for i in range(num_seeds):
            seed = self.base_seed + (i * 1000)
            logger.info(f"\n  Run {i + 1}/{num_seeds} (seed={seed})...")

            self._set_config_seed(seed, num_patients)
            runner = SimulationRunner(config_dir=self.config_dir, llm_provider=self.llm_provider)
            runner.generate_patients(num_patients)

            # Run all scenario types
            conflict_results = runner.run_all_conflict_scenarios()
            pattern_results = runner.run_all_pattern_scenarios()

            # Run routine queries on clean patients
            routine_results = []
            clean_patients = [
                p for p in runner.patients
                if not p.known_conflicts and not p.known_patterns
            ]
            for patient in clean_patients[:20]:
                dept = patient.departments[0] if patient.departments else "general_practice"
                result = runner.run_scenario(
                    patient=patient,
                    requesting_dept=dept,
                    clinical_context={
                        "planned_procedure": "routine_checkup",
                        "relevant_categories": ["all"],
                    },
                    query_type="MED_CONFLICT",
                    scenario_name="routine_check",
                )
                routine_results.append(result)

            eval_result = self._evaluate(
                runner, conflict_results, pattern_results, routine_results
            )

            # Collect per-conflict-type detection
            per_conflict = self._per_conflict_detection(conflict_results)
            per_pattern = self._per_pattern_detection(pattern_results)

            runs.append({
                "seed": seed,
                "conflict_detection_rate": eval_result["conflict_detection"]["detection_rate"],
                "pattern_detection_rate": eval_result["pattern_detection"]["detection_rate"],
                "false_positive_rate": eval_result["false_positives"]["false_positive_rate"],
                "avg_response_time_s": eval_result["performance"]["average_response_time_seconds"],
                "total_conflicts": eval_result["conflict_detection"]["total_conflict_scenarios"],
                "total_patterns": eval_result["pattern_detection"]["total_pattern_scenarios"],
                "per_conflict": per_conflict,
                "per_pattern": per_pattern,
                "privacy": eval_result["privacy_compliance"],
            })

            logger.info(
                f"  Seed {seed}: conflicts={eval_result['conflict_detection']['detection_rate']:.2f}, "
                f"patterns={eval_result['pattern_detection']['detection_rate']:.2f}, "
                f"fp={eval_result['false_positives']['false_positive_rate']:.2f}"
            )

        self._restore_config()

        # Compute aggregate statistics
        metrics_keys = [
            "conflict_detection_rate",
            "pattern_detection_rate",
            "false_positive_rate",
            "avg_response_time_s",
        ]
        aggregate = {}
        for key in metrics_keys:
            values = [r[key] for r in runs]
            aggregate[key] = {
                "mean": round(_mean(values), 4),
                "std": round(_std(values), 4),
                "min": round(min(values), 4),
                "max": round(max(values), 4),
            }

        # Per-conflict-type aggregate
        all_conflict_types = set()
        for run in runs:
            all_conflict_types.update(run["per_conflict"].keys())

        per_conflict_aggregate = {}
        for ctype in sorted(all_conflict_types):
            values = [r["per_conflict"].get(ctype, 0.0) for r in runs]
            per_conflict_aggregate[ctype] = {
                "mean": round(_mean(values), 4),
                "std": round(_std(values), 4),
            }

        # Per-pattern-type aggregate
        all_pattern_types = set()
        for run in runs:
            all_pattern_types.update(run["per_pattern"].keys())

        per_pattern_aggregate = {}
        for ptype in sorted(all_pattern_types):
            values = [r["per_pattern"].get(ptype, 0.0) for r in runs]
            per_pattern_aggregate[ptype] = {
                "mean": round(_mean(values), 4),
                "std": round(_std(values), 4),
            }

        return {
            "experiment": "multi_seed_variance",
            "num_seeds": num_seeds,
            "num_patients": num_patients,
            "runs": runs,
            "aggregate": aggregate,
            "per_conflict_aggregate": per_conflict_aggregate,
            "per_pattern_aggregate": per_pattern_aggregate,
        }

    # ────────────────────────────────────────────────────────────
    # Experiment 4: Scalability Analysis
    # ────────────────────────────────────────────────────────────
    def run_scalability(self, patient_counts: list = None) -> dict:
        """Run full simulation at different patient counts.

        Measures how detection rates and performance scale.
        """
        if patient_counts is None:
            patient_counts = [50, 100, 250, 500]

        logger.info("=" * 60)
        logger.info(f"Experiment 4: Scalability Analysis ({patient_counts})")
        logger.info("=" * 60)

        results_by_count = {}

        for count in patient_counts:
            logger.info(f"\n  Running with {count} patients...")

            self._set_config_patients(count)
            runner = SimulationRunner(config_dir=self.config_dir, llm_provider=self.llm_provider)

            start_total = time.time()
            runner.generate_patients(count)

            conflict_results = runner.run_all_conflict_scenarios()
            pattern_results = runner.run_all_pattern_scenarios()

            # Run routine on a sample of clean patients
            routine_results = []
            clean_patients = [
                p for p in runner.patients
                if not p.known_conflicts and not p.known_patterns
            ]
            for patient in clean_patients[:min(20, len(clean_patients))]:
                dept = patient.departments[0] if patient.departments else "general_practice"
                result = runner.run_scenario(
                    patient=patient,
                    requesting_dept=dept,
                    clinical_context={
                        "planned_procedure": "routine_checkup",
                        "relevant_categories": ["all"],
                    },
                    query_type="MED_CONFLICT",
                    scenario_name="routine_check",
                )
                routine_results.append(result)

            total_time = time.time() - start_total
            eval_result = self._evaluate(
                runner, conflict_results, pattern_results, routine_results
            )

            n_scenarios = (
                len(conflict_results) + len(pattern_results) + len(routine_results)
            )

            results_by_count[count] = {
                "patients": count,
                "conflict_scenarios": len(conflict_results),
                "pattern_scenarios": len(pattern_results),
                "routine_scenarios": len(routine_results),
                "total_scenarios": n_scenarios,
                "conflict_detection_rate": eval_result["conflict_detection"]["detection_rate"],
                "pattern_detection_rate": eval_result["pattern_detection"]["detection_rate"],
                "false_positive_rate": eval_result["false_positives"]["false_positive_rate"],
                "avg_response_time_ms": round(
                    eval_result["performance"]["average_response_time_seconds"] * 1000, 2
                ),
                "total_simulation_time_s": round(total_time, 2),
                "audit_events": eval_result["privacy_compliance"].get("total_events", 0),
            }

            logger.info(
                f"  {count} patients: {n_scenarios} scenarios in {total_time:.1f}s, "
                f"conflicts={eval_result['conflict_detection']['detection_rate']:.2f}, "
                f"patterns={eval_result['pattern_detection']['detection_rate']:.2f}"
            )

        self._restore_config()

        return {
            "experiment": "scalability",
            "results_by_count": results_by_count,
        }

    # ────────────────────────────────────────────────────────────
    # Experiment 5: LLM Provider Comparison
    # ────────────────────────────────────────────────────────────
    def run_provider_comparison(self, num_patients: int = 100,
                                 num_runs: int = 3) -> dict:
        """Run identical scenarios with Mock provider vs a real LLM provider.

        Compares detection accuracy, false positive rates, response times,
        and per-scenario agreement between the two providers.

        The real LLM provider is determined by settings.yaml (ollama,
        openai_compatible, or huggingface). The Mock provider is always
        used as the baseline.

        Args:
            num_patients: Number of synthetic patients per run.
            num_runs: Number of independent runs per provider (for variance).
        """
        logger.info("=" * 60)
        logger.info("Experiment 5: LLM Provider Comparison")
        logger.info("=" * 60)

        # ── Resolve the real LLM provider ──
        llm_config = self.settings.get("llm", {})
        real_provider_name = llm_config.get("provider", "mock")

        if real_provider_name == "mock":
            logger.warning(
                "settings.yaml has provider='mock'. "
                "Comparison requires a real provider (ollama, openai_compatible, huggingface). "
                "Set llm.provider in config/settings.yaml and try again."
            )
            return {
                "experiment": "provider_comparison",
                "error": "No real LLM provider configured. Set llm.provider in settings.yaml.",
            }

        real_llm = create_llm_provider(llm_config)
        if not real_llm.is_available():
            logger.error(f"Real LLM provider '{real_provider_name}' is not available.")
            return {
                "experiment": "provider_comparison",
                "error": f"Provider '{real_provider_name}' not available. Check connection.",
            }

        mock_llm = MockLLMProvider()
        provider_label = {
            "ollama": f"Ollama ({llm_config.get('ollama', {}).get('model', 'unknown')})",
            "openai_compatible": f"OpenAI API ({llm_config.get('openai_compatible', {}).get('model', 'unknown')})",
            "huggingface": f"HuggingFace ({llm_config.get('huggingface', {}).get('model_id', 'unknown')})",
        }.get(real_provider_name, real_provider_name)

        logger.info(f"  Baseline: MockLLM (rule-based)")
        logger.info(f"  Comparison: {provider_label}")
        logger.info(f"  Runs per provider: {num_runs}, Patients per run: {num_patients}")

        # ── Helper: run a single full evaluation with a given provider ──
        def _run_with_provider(provider: BaseLLMProvider, label: str,
                               seed: int) -> dict:
            self._set_config_seed(seed, num_patients)
            runner = SimulationRunner(
                config_dir=self.config_dir, llm_provider=provider
            )
            runner.generate_patients(num_patients)

            start = time.time()
            conflict_results = runner.run_all_conflict_scenarios()
            pattern_results = runner.run_all_pattern_scenarios()

            # Routine queries for FP measurement
            routine_results = []
            clean = [
                p for p in runner.patients
                if not p.known_conflicts and not p.known_patterns
            ]
            for patient in clean[:20]:
                dept = patient.departments[0] if patient.departments else "general_practice"
                result = runner.run_scenario(
                    patient=patient,
                    requesting_dept=dept,
                    clinical_context={
                        "planned_procedure": "routine_checkup",
                        "relevant_categories": ["all"],
                    },
                    query_type="MED_CONFLICT",
                    scenario_name="routine_check",
                )
                routine_results.append(result)

            elapsed = time.time() - start
            eval_result = self._evaluate(
                runner, conflict_results, pattern_results, routine_results
            )
            per_conflict = self._per_conflict_detection(conflict_results)
            per_pattern = self._per_pattern_detection(pattern_results)

            return {
                "provider": label,
                "seed": seed,
                "conflict_detection_rate": eval_result["conflict_detection"]["detection_rate"],
                "pattern_detection_rate": eval_result["pattern_detection"]["detection_rate"],
                "false_positive_rate": eval_result["false_positives"]["false_positive_rate"],
                "avg_response_time_s": eval_result["performance"]["average_response_time_seconds"],
                "total_time_s": round(elapsed, 2),
                "total_conflicts": eval_result["conflict_detection"]["total_conflict_scenarios"],
                "total_patterns": eval_result["pattern_detection"]["total_pattern_scenarios"],
                "per_conflict": per_conflict,
                "per_pattern": per_pattern,
            }

        # ── Run both providers across multiple seeds ──
        mock_runs = []
        real_runs = []

        for i in range(num_runs):
            seed = self.base_seed + (i * 1000)

            logger.info(f"\n  Run {i + 1}/{num_runs} (seed={seed})...")

            logger.info(f"    MockLLM...")
            mock_result = _run_with_provider(mock_llm, "MockLLM", seed)
            mock_runs.append(mock_result)
            logger.info(
                f"    MockLLM: conflicts={mock_result['conflict_detection_rate']:.2f}, "
                f"patterns={mock_result['pattern_detection_rate']:.2f}, "
                f"fp={mock_result['false_positive_rate']:.2f}, "
                f"time={mock_result['total_time_s']:.1f}s"
            )

            logger.info(f"    {provider_label}...")
            real_result = _run_with_provider(real_llm, provider_label, seed)
            real_runs.append(real_result)
            logger.info(
                f"    {provider_label}: conflicts={real_result['conflict_detection_rate']:.2f}, "
                f"patterns={real_result['pattern_detection_rate']:.2f}, "
                f"fp={real_result['false_positive_rate']:.2f}, "
                f"time={real_result['total_time_s']:.1f}s"
            )

        self._restore_config()

        # ── Compute aggregate statistics for each provider ──
        def _aggregate_runs(runs: list) -> dict:
            keys = [
                "conflict_detection_rate", "pattern_detection_rate",
                "false_positive_rate", "avg_response_time_s", "total_time_s",
            ]
            agg = {}
            for key in keys:
                values = [r[key] for r in runs]
                agg[key] = {
                    "mean": round(_mean(values), 4),
                    "std": round(_std(values), 4),
                    "min": round(min(values), 4),
                    "max": round(max(values), 4),
                }
            return agg

        mock_aggregate = _aggregate_runs(mock_runs)
        real_aggregate = _aggregate_runs(real_runs)

        # ── Per-scenario agreement: how often do both providers agree? ──
        agreement = {"total_scenarios": 0, "agreed": 0, "disagreed_scenarios": []}
        for m_run, r_run in zip(mock_runs, real_runs):
            for ctype in set(list(m_run["per_conflict"].keys()) +
                             list(r_run["per_conflict"].keys())):
                m_det = m_run["per_conflict"].get(ctype, 0.0) > 0
                r_det = r_run["per_conflict"].get(ctype, 0.0) > 0
                agreement["total_scenarios"] += 1
                if m_det == r_det:
                    agreement["agreed"] += 1
                else:
                    agreement["disagreed_scenarios"].append({
                        "seed": m_run["seed"], "scenario": ctype,
                        "mock_detected": m_det, "real_detected": r_det,
                    })

            for ptype in set(list(m_run["per_pattern"].keys()) +
                             list(r_run["per_pattern"].keys())):
                m_det = m_run["per_pattern"].get(ptype, 0.0) > 0
                r_det = r_run["per_pattern"].get(ptype, 0.0) > 0
                agreement["total_scenarios"] += 1
                if m_det == r_det:
                    agreement["agreed"] += 1
                else:
                    agreement["disagreed_scenarios"].append({
                        "seed": m_run["seed"], "scenario": ptype,
                        "mock_detected": m_det, "real_detected": r_det,
                    })

        agreement["agreement_rate"] = round(
            agreement["agreed"] / agreement["total_scenarios"], 4
        ) if agreement["total_scenarios"] else 0

        logger.info(f"\n  === Provider Comparison Summary ===")
        logger.info(f"  Mock   — Conflict: {mock_aggregate['conflict_detection_rate']['mean']:.2%} "
                     f"Pattern: {mock_aggregate['pattern_detection_rate']['mean']:.2%} "
                     f"FP: {mock_aggregate['false_positive_rate']['mean']:.2%}")
        logger.info(f"  {provider_label} — Conflict: {real_aggregate['conflict_detection_rate']['mean']:.2%} "
                     f"Pattern: {real_aggregate['pattern_detection_rate']['mean']:.2%} "
                     f"FP: {real_aggregate['false_positive_rate']['mean']:.2%}")
        logger.info(f"  Agreement rate: {agreement['agreement_rate']:.2%}")

        return {
            "experiment": "provider_comparison",
            "num_patients": num_patients,
            "num_runs": num_runs,
            "real_provider": provider_label,
            "real_provider_name": real_provider_name,
            "mock": {"runs": mock_runs, "aggregate": mock_aggregate},
            "real": {"runs": real_runs, "aggregate": real_aggregate},
            "agreement": agreement,
        }

    # ────────────────────────────────────────────────────────────
    # Run All Experiments
    # ────────────────────────────────────────────────────────────
    def run_all_experiments(self, num_patients: int = 100,
                            num_seeds: int = 5,
                            patient_counts: list = None,
                            include_provider_comparison: bool = False,
                            comparison_runs: int = 3) -> dict:
        """Run all experiments and save results.

        Args:
            num_patients: Base patient count for most experiments.
            num_seeds: Number of random seeds for variance experiment.
            patient_counts: List of patient counts for scalability experiment.
            include_provider_comparison: If True, run Mock vs real LLM comparison.
            comparison_runs: Number of runs per provider for comparison.
        """
        logger.info("\n" + "=" * 60)
        logger.info("MedAgentNet: Running All Experiments")
        logger.info("=" * 60)

        overall_start = time.time()

        all_results = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "base_num_patients": num_patients,
                "num_seeds": num_seeds,
                "patient_counts": patient_counts or [50, 100, 250, 500],
                "base_seed": self.base_seed,
            },
        }

        # Experiment 1
        all_results["tier_comparison"] = self.run_tier_comparison(num_patients)

        # Experiment 2
        all_results["consent_restriction"] = self.run_consent_restriction(num_patients)

        # Experiment 3
        all_results["multi_seed_variance"] = self.run_multi_seed_variance(
            num_seeds, num_patients
        )

        # Experiment 4
        all_results["scalability"] = self.run_scalability(patient_counts)

        # Experiment 5 (optional — requires real LLM provider)
        if include_provider_comparison:
            all_results["provider_comparison"] = self.run_provider_comparison(
                num_patients, comparison_runs
            )

        total_time = time.time() - overall_start
        all_results["total_experiment_time_s"] = round(total_time, 2)

        # Save JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(self.results_dir, f"experiments_{timestamp}.json")
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        logger.info(f"\nJSON results saved to: {json_path}")

        # Generate and save LaTeX tables
        latex = self.generate_latex_tables(all_results)
        latex_path = os.path.join(self.results_dir, f"tables_{timestamp}.tex")
        with open(latex_path, "w") as f:
            f.write(latex)
        logger.info(f"LaTeX tables saved to: {latex_path}")

        return all_results

    # ────────────────────────────────────────────────────────────
    # Evaluation Helpers
    # ────────────────────────────────────────────────────────────
    def _evaluate(self, runner: SimulationRunner,
                  conflict_results: list, pattern_results: list,
                  routine_results: list = None) -> dict:
        """Evaluate simulation results. Mirrors SimulationRunner.evaluate()."""
        if routine_results is None:
            routine_results = []

        # Conflict detection
        conflicts_detected = 0
        conflicts_total = len(conflict_results)
        for result in conflict_results:
            if result["num_alerts"] > 0:
                has_real = any(
                    (a.get("alert_type") or "unknown") not in ("no_conflict", "parse_error")
                    for a in result["alerts"]
                )
                if has_real:
                    conflicts_detected += 1

        # Pattern detection
        patterns_detected = 0
        patterns_total = len(pattern_results)
        for result in pattern_results:
            if result["num_alerts"] > 0:
                has_meaningful = any(
                    (a.get("alert_type") or "") not in ("no_conflict", "parse_error", "llm_error")
                    and (a.get("severity") or a.get("alert_level") or "")
                    in ("moderate", "high", "high_risk", "critical")
                    for a in result["alerts"]
                )
                has_keywords = any(
                    any(
                        kw in ((a.get("description") or "") + " " + (a.get("alert_type") or "")).lower()
                        for kw in [
                            "pattern", "trend", "rising", "declining", "progressive",
                            "diabetes", "ckd", "kidney", "thyroid", "retinopathy",
                            "neuropathy", "glucose", "hba1c", "egfr", "creatinine",
                            "connection", "suggestive", "consistent", "progression",
                            "worsening", "deteriorat", "abnormal", "elevated",
                        ]
                    )
                    for a in result["alerts"]
                )
                has_summary = any(
                    any(
                        kw in (rs.get("summary") or "").lower()
                        for kw in [
                            "pattern", "trend", "rising", "declining", "diabetes",
                            "kidney", "thyroid", "retinopathy", "neuropathy",
                            "glucose", "abnormal", "elevated", "progression",
                        ]
                    )
                    for rs in result.get("response_summaries", [])
                )
                if has_meaningful or has_keywords or has_summary:
                    patterns_detected += 1

        # False positives
        false_positives = 0
        routine_total = len(routine_results)
        for result in routine_results:
            has_fp = any(
                a.get("alert_level") in ("high_risk", "critical")
                for a in result["alerts"]
            )
            if has_fp:
                false_positives += 1

        # Performance
        all_results = conflict_results + pattern_results + routine_results
        times = [r["elapsed_seconds"] for r in all_results if r["elapsed_seconds"] > 0]
        avg_time = _mean(times) if times else 0

        # Privacy
        privacy = runner.audit.get_privacy_report()

        return {
            "conflict_detection": {
                "total_conflict_scenarios": conflicts_total,
                "conflicts_detected": conflicts_detected,
                "detection_rate": round(
                    conflicts_detected / conflicts_total, 4
                ) if conflicts_total else 0,
                "missed": conflicts_total - conflicts_detected,
            },
            "pattern_detection": {
                "total_pattern_scenarios": patterns_total,
                "patterns_detected": patterns_detected,
                "detection_rate": round(
                    patterns_detected / patterns_total, 4
                ) if patterns_total else 0,
                "missed": patterns_total - patterns_detected,
            },
            "false_positives": {
                "routine_scenarios_checked": routine_total,
                "false_positives": false_positives,
                "false_positive_rate": round(
                    false_positives / routine_total, 4
                ) if routine_total else 0,
            },
            "performance": {
                "average_response_time_seconds": round(avg_time, 6),
                "max_response_time_seconds": round(max(times), 6) if times else 0,
                "min_response_time_seconds": round(min(times), 6) if times else 0,
            },
            "privacy_compliance": privacy,
        }

    def _per_conflict_detection(self, conflict_results: list) -> dict:
        """Return per-conflict-type detection rate (1.0 or 0.0 per scenario)."""
        by_type = {}
        for r in conflict_results:
            name = r.get("scenario_name", "unknown")
            detected = r["num_alerts"] > 0 and any(
                (a.get("alert_type") or "unknown") not in ("no_conflict", "parse_error")
                for a in r["alerts"]
            )
            if name not in by_type:
                by_type[name] = []
            by_type[name].append(1.0 if detected else 0.0)

        return {name: _mean(vals) for name, vals in by_type.items()}

    def _per_pattern_detection(self, pattern_results: list) -> dict:
        """Return per-pattern-type detection rate."""
        by_type = {}
        for r in pattern_results:
            name = r.get("scenario_name", "unknown")
            detected = r["num_alerts"] > 0 and any(
                (a.get("alert_type") or "unknown") not in ("no_conflict", "parse_error", "llm_error")
                for a in r["alerts"]
            )
            if name not in by_type:
                by_type[name] = []
            by_type[name].append(1.0 if detected else 0.0)

        return {name: _mean(vals) for name, vals in by_type.items()}

    def _build_pattern_context(self, patient, pattern: dict) -> dict:
        """Build clinical context for a pattern scenario."""
        context = {
            "pattern_category": pattern.get("pattern_type", "diagnostic"),
            "departments_involved": pattern.get("departments", []),
            "expected": pattern.get("expected_diagnosis", ""),
            "query_reason": (
                f"Cross-departmental pattern analysis requested. "
                f"Looking for connections between findings across "
                f"{', '.join(pattern.get('departments', []))}. "
                f"Check for trends, rising/declining values, and "
                f"multi-system disease patterns."
            ),
        }

        # Add lab trends
        lab_by_code = {}
        for lab in patient.lab_results:
            code = lab.test_code
            if code not in lab_by_code:
                lab_by_code[code] = []
            lab_by_code[code].append(lab)

        for code, labs in lab_by_code.items():
            sorted_labs = sorted(labs, key=lambda x: x.date)
            values = [l.value for l in sorted_labs]
            abnormals = [l for l in sorted_labs if l.is_abnormal]
            if len(values) > 1:
                trend = "rising" if values[-1] > values[0] else "declining"
                context[f"lab_trend_{code.lower()}"] = (
                    f"{sorted_labs[0].test_name}: {' -> '.join(str(v) for v in values)} "
                    f"({trend})"
                )
            if abnormals:
                context[f"lab_abnormal_{code.lower()}"] = (
                    f"{abnormals[-1].test_name}: {abnormals[-1].value} "
                    f"(normal: {abnormals[-1].normal_range[0]}-{abnormals[-1].normal_range[1]})"
                )

        # Add conditions
        for cond in patient.conditions:
            context[f"condition_{cond.department}_{cond.code}"] = (
                f"{cond.name} (severity: {cond.severity}, dept: {cond.department})"
            )

        # Add medications
        for med in patient.medications:
            context[f"medication_{med.department}_{med.name.lower().replace(' ', '_')}"] = (
                f"{med.name} ({med.category}), prescribed by {med.department}"
            )

        return context

    # ────────────────────────────────────────────────────────────
    # Config Helpers
    # ────────────────────────────────────────────────────────────
    def _set_config_seed(self, seed: int, num_patients: int = None):
        """Temporarily set seed in config."""
        settings_path = os.path.join(self.config_dir, "settings.yaml")
        with open(settings_path) as f:
            settings = yaml.safe_load(f)
        settings["simulation"]["random_seed"] = seed
        if num_patients is not None:
            settings["simulation"]["num_patients"] = num_patients
        with open(settings_path, "w") as f:
            yaml.dump(settings, f, default_flow_style=False)

    def _set_config_patients(self, num_patients: int):
        """Temporarily set patient count in config."""
        self._set_config_seed(self.base_seed, num_patients)

    def _restore_config(self):
        """Restore original config values."""
        self._set_config_seed(self.base_seed, self.base_num_patients)

    # ────────────────────────────────────────────────────────────
    # LaTeX Table Generation
    # ────────────────────────────────────────────────────────────
    def generate_latex_tables(self, results: dict) -> str:
        """Generate LaTeX table code from experiment results."""
        sections = []
        sections.append("% ============================================================")
        sections.append("% MedAgentNet Experiment Results — Auto-generated Tables")
        sections.append(f"% Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        sections.append("% ============================================================")
        sections.append("")

        # Table 1: Tier Comparison
        if "tier_comparison" in results:
            sections.append(self._latex_tier_comparison(results["tier_comparison"]))

        # Table 2: Per-Scenario Detection (from multi-seed)
        if "multi_seed_variance" in results:
            sections.append(self._latex_per_scenario_detection(results["multi_seed_variance"]))

        # Table 3: Multi-Seed Summary
        if "multi_seed_variance" in results:
            sections.append(self._latex_variance_summary(results["multi_seed_variance"]))

        # Table 4: Consent Restriction
        if "consent_restriction" in results:
            sections.append(self._latex_consent_restriction(results["consent_restriction"]))

        # Table 5: Scalability
        if "scalability" in results:
            sections.append(self._latex_scalability(results["scalability"]))

        # Table 6: Provider Comparison
        if "provider_comparison" in results and "error" not in results["provider_comparison"]:
            sections.append(self._latex_provider_comparison(results["provider_comparison"]))

        return "\n\n".join(sections)

    def _latex_tier_comparison(self, data: dict) -> str:
        """Generate tier comparison LaTeX table."""
        tiers = data.get("results_by_tier", {})
        t1 = tiers.get(1, {})
        t2 = tiers.get(2, {})
        t3 = tiers.get(3, {})

        def _get(tier_data, *keys):
            d = tier_data
            for k in keys:
                d = d.get(k, {}) if isinstance(d, dict) else 0
            return d if not isinstance(d, dict) else 0

        lines = [
            r"\begin{table}[!t]",
            r"\centering",
            r"\caption{Impact of disclosure tier on detection accuracy. Tier~2 (Clinical Summary) achieves optimal detection with moderate privacy exposure.}",
            r"\label{tab:tier_comparison}",
            r"\small",
            r"\begin{tabularx}{\columnwidth}{@{}lXXX@{}}",
            r"\toprule",
            r"\textbf{Metric} & \textbf{Tier 1} & \textbf{Tier 2} & \textbf{Tier 3} \\",
            r"& \textbf{(Flag Only)} & \textbf{(Clinical)} & \textbf{(Full Context)} \\",
            r"\midrule",
        ]

        cd1 = _get(t1, "conflict_detection", "detection_rate")
        cd2 = _get(t2, "conflict_detection", "detection_rate")
        cd3 = _get(t3, "conflict_detection", "detection_rate")
        lines.append(
            f"Conflict Detection Rate & {_fmt(cd1)} & {_fmt(cd2)} & {_fmt(cd3)} \\\\"
        )

        pd1 = _get(t1, "pattern_detection", "detection_rate")
        pd2 = _get(t2, "pattern_detection", "detection_rate")
        pd3 = _get(t3, "pattern_detection", "detection_rate")
        lines.append(
            f"Pattern Detection Rate & {_fmt(pd1)} & {_fmt(pd2)} & {_fmt(pd3)} \\\\"
        )

        rt1 = _get(t1, "performance", "average_response_time_seconds")
        rt2 = _get(t2, "performance", "average_response_time_seconds")
        rt3 = _get(t3, "performance", "average_response_time_seconds")
        lines.append(
            f"Avg. Response Time (ms) & {_fmt_ms(rt1)} & {_fmt_ms(rt2)} & {_fmt_ms(rt3)} \\\\"
        )

        lines.extend([
            r"\bottomrule",
            r"\end{tabularx}",
            r"\end{table}",
        ])
        return "\n".join(lines)

    def _latex_per_scenario_detection(self, data: dict) -> str:
        """Generate per-scenario detection table from multi-seed results."""
        per_conflict = data.get("per_conflict_aggregate", {})
        per_pattern = data.get("per_pattern_aggregate", {})
        n_seeds = data.get("num_seeds", 1)

        # Friendly names for conflict types
        conflict_names = {
            "warfarin_dental_extraction": "Warfarin + Dental Extraction",
            "triple_whammy_aki": "Triple Whammy (ACE+NSAID+Diuretic)",
            "metformin_renal_failure": "Metformin + Renal Failure",
            "beta_blocker_respiratory": "Beta-Blocker + Asthma/COPD",
            "anticoagulant_nsaid_bleeding": "Warfarin + NSAID (GI Bleed)",
            "ophthalmic_systemic_beta_blocker": "Timolol + Metoprolol",
            "methotrexate_nsaid_renal": "Methotrexate + NSAID",
            "carbamazepine_warfarin": "Carbamazepine + Warfarin",
        }
        pattern_names = {
            "undiagnosed_diabetes": "Undiagnosed Diabetes",
            "ckd_progression": "CKD Progression",
            "thyroid_cardiac_connection": "Thyroid--Cardiac Connection",
        }

        lines = [
            r"\begin{table}[!t]",
            r"\centering",
            r"\caption{Per-scenario detection rates (mean $\pm$ std over " + str(n_seeds) + r" runs, $n=" + str(data.get("num_patients", 100)) + r"$ patients per run).}",
            r"\label{tab:per_scenario}",
            r"\small",
            r"\begin{tabularx}{\columnwidth}{@{}Xr@{}}",
            r"\toprule",
            r"\textbf{Scenario} & \textbf{Detection Rate} \\",
            r"\midrule",
            r"\multicolumn{2}{@{}l}{\textit{Medication Conflicts}} \\",
        ]

        for key in sorted(per_conflict.keys()):
            name = conflict_names.get(key, key.replace("_", " ").title())
            mean_val = per_conflict[key]["mean"]
            std_val = per_conflict[key]["std"]
            if std_val > 0:
                lines.append(f"\\quad {name} & ${mean_val * 100:.0f} \\pm {std_val * 100:.0f}$\\% \\\\")
            else:
                lines.append(f"\\quad {name} & {_fmt(mean_val)} \\\\")

        lines.append(r"\midrule")
        lines.append(r"\multicolumn{2}{@{}l}{\textit{Diagnostic Patterns}} \\")

        for key in sorted(per_pattern.keys()):
            name = pattern_names.get(key, key.replace("_", " ").title())
            mean_val = per_pattern[key]["mean"]
            std_val = per_pattern[key]["std"]
            if std_val > 0:
                lines.append(f"\\quad {name} & ${mean_val * 100:.0f} \\pm {std_val * 100:.0f}$\\% \\\\")
            else:
                lines.append(f"\\quad {name} & {_fmt(mean_val)} \\\\")

        lines.extend([
            r"\bottomrule",
            r"\end{tabularx}",
            r"\end{table}",
        ])
        return "\n".join(lines)

    def _latex_variance_summary(self, data: dict) -> str:
        """Generate multi-seed summary table."""
        agg = data.get("aggregate", {})
        n_seeds = data.get("num_seeds", 1)
        n_patients = data.get("num_patients", 100)

        lines = [
            r"\begin{table}[!t]",
            r"\centering",
            r"\caption{Simulation robustness: aggregate metrics over " + str(n_seeds) + r" independent runs ($n=" + str(n_patients) + r"$ patients, different random seeds).}",
            r"\label{tab:variance}",
            r"\small",
            r"\begin{tabularx}{\columnwidth}{@{}Xcccc@{}}",
            r"\toprule",
            r"\textbf{Metric} & \textbf{Mean} & \textbf{Std} & \textbf{Min} & \textbf{Max} \\",
            r"\midrule",
        ]

        metric_labels = {
            "conflict_detection_rate": "Conflict Detection Rate",
            "pattern_detection_rate": "Pattern Detection Rate",
            "false_positive_rate": "False Positive Rate",
            "avg_response_time_s": "Avg. Response Time (s)",
        }

        for key, label in metric_labels.items():
            vals = agg.get(key, {})
            if key == "avg_response_time_s":
                lines.append(
                    f"{label} & {vals.get('mean', 0) * 1000:.1f}ms & "
                    f"{vals.get('std', 0) * 1000:.1f}ms & "
                    f"{vals.get('min', 0) * 1000:.1f}ms & "
                    f"{vals.get('max', 0) * 1000:.1f}ms \\\\"
                )
            else:
                lines.append(
                    f"{label} & {_fmt(vals.get('mean', 0))} & "
                    f"{_fmt(vals.get('std', 0))} & "
                    f"{_fmt(vals.get('min', 0))} & "
                    f"{_fmt(vals.get('max', 0))} \\\\"
                )

        lines.extend([
            r"\bottomrule",
            r"\end{tabularx}",
            r"\end{table}",
        ])
        return "\n".join(lines)

    def _latex_consent_restriction(self, data: dict) -> str:
        """Generate consent restriction impact table."""
        scenarios = data.get("results_by_scenario", {})

        def _get(scenario_key, *keys):
            d = scenarios.get(scenario_key, {})
            for k in keys:
                d = d.get(k, {}) if isinstance(d, dict) else 0
            return d if not isinstance(d, dict) else 0

        lines = [
            r"\begin{table}[!t]",
            r"\centering",
            r"\caption{Impact of patient consent restrictions on detection accuracy. Results demonstrate the privacy--utility tradeoff inherent in federated architectures.}",
            r"\label{tab:consent}",
            r"\small",
            r"\begin{tabularx}{\columnwidth}{@{}lXXXX@{}}",
            r"\toprule",
            r"\textbf{Metric} & \textbf{Full} & \textbf{30\%} & \textbf{60\%} & \textbf{Opt-} \\",
            r"& \textbf{Consent} & \textbf{Restricted} & \textbf{Restricted} & \textbf{Out} \\",
            r"\midrule",
        ]

        scenario_keys = ["full", "partial_30", "partial_60", "opt_out"]
        for label, path in [
            ("Conflict Detection", ("conflict_detection", "detection_rate")),
            ("Pattern Detection", ("pattern_detection", "detection_rate")),
        ]:
            vals = [_fmt(_get(s, *path)) for s in scenario_keys]
            lines.append(f"{label} & {' & '.join(vals)} \\\\")

        lines.extend([
            r"\bottomrule",
            r"\end{tabularx}",
            r"\end{table}",
        ])
        return "\n".join(lines)

    def _latex_scalability(self, data: dict) -> str:
        """Generate scalability analysis table."""
        by_count = data.get("results_by_count", {})

        # Sort by patient count
        counts = sorted(by_count.keys(), key=lambda x: int(x))

        lines = [
            r"\begin{table}[!t]",
            r"\centering",
            r"\caption{Scalability analysis: detection accuracy and performance across patient population sizes (Mock LLM provider).}",
            r"\label{tab:scalability}",
            r"\small",
            r"\begin{tabularx}{\columnwidth}{@{}rXXXrr@{}}",
            r"\toprule",
            r"\textbf{Patients} & \textbf{Conflict} & \textbf{Pattern} & \textbf{FP} & \textbf{Avg.} & \textbf{Total} \\",
            r"& \textbf{Det.} & \textbf{Det.} & \textbf{Rate} & \textbf{(ms)} & \textbf{(s)} \\",
            r"\midrule",
        ]

        for count in counts:
            d = by_count[count]
            lines.append(
                f"{count} & "
                f"{_fmt(d['conflict_detection_rate'])} & "
                f"{_fmt(d['pattern_detection_rate'])} & "
                f"{_fmt(d['false_positive_rate'])} & "
                f"{d['avg_response_time_ms']:.1f} & "
                f"{d['total_simulation_time_s']:.1f} \\\\"
            )

        lines.extend([
            r"\bottomrule",
            r"\end{tabularx}",
            r"\end{table}",
        ])
        return "\n".join(lines)

    def _latex_provider_comparison(self, data: dict) -> str:
        """Generate LLM provider comparison LaTeX table."""
        real_name = data.get("real_provider", "Real LLM")
        num_runs = data.get("num_runs", 1)
        num_patients = data.get("num_patients", 100)
        mock_agg = data.get("mock", {}).get("aggregate", {})
        real_agg = data.get("real", {}).get("aggregate", {})
        agreement = data.get("agreement", {})

        def _val(agg: dict, key: str) -> str:
            """Format mean +/- std for a metric."""
            vals = agg.get(key, {})
            m = vals.get("mean", 0)
            s = vals.get("std", 0)
            if key in ("avg_response_time_s", "total_time_s"):
                if s > 0:
                    return f"{m * 1000:.0f} $\\pm$ {s * 1000:.0f}ms"
                return f"{m * 1000:.0f}ms"
            if s > 0:
                return f"{m * 100:.1f} $\\pm$ {s * 100:.1f}\\%"
            return _fmt(m)

        # Shorten real provider name for column header
        short_name = real_name
        if len(short_name) > 20:
            short_name = short_name[:18] + ".."

        lines = [
            r"\begin{table}[!t]",
            r"\centering",
            r"\caption{LLM provider comparison: Mock (rule-based) vs " + real_name
            + r" (mean $\pm$ std over " + str(num_runs) + r" runs, $n=" + str(num_patients) + r"$ patients).}",
            r"\label{tab:provider_comparison}",
            r"\small",
            r"\begin{tabularx}{\columnwidth}{@{}lXX@{}}",
            r"\toprule",
            r"\textbf{Metric} & \textbf{MockLLM} & \textbf{" + short_name + r"} \\",
            r"\midrule",
        ]

        metric_labels = [
            ("conflict_detection_rate", "Conflict Detection Rate"),
            ("pattern_detection_rate", "Pattern Detection Rate"),
            ("false_positive_rate", "False Positive Rate"),
            ("avg_response_time_s", "Avg. Response Time"),
            ("total_time_s", "Total Simulation Time"),
        ]

        for key, label in metric_labels:
            mock_val = _val(mock_agg, key)
            real_val = _val(real_agg, key)
            lines.append(f"{label} & {mock_val} & {real_val} \\\\")

        # Add agreement row
        agr_rate = agreement.get("agreement_rate", 0)
        lines.append(r"\midrule")
        lines.append(f"Scenario Agreement & \\multicolumn{{2}}{{c}}{{{_fmt(agr_rate)}}} \\\\")

        lines.extend([
            r"\bottomrule",
            r"\end{tabularx}",
            r"\end{table}",
        ])
        return "\n".join(lines)
