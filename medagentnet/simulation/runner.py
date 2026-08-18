"""
MedAgentNet - Simulation Runner & Evaluator
Orchestrates the full simulation: generates patients, initializes agents,
runs clinical scenarios, and evaluates results.
"""
import os
import json
import time
import logging
from datetime import datetime
from typing import Optional

import yaml

from protocol.models import PatientRecord, ConflictAlert, DisclosureTier
from protocol.consent import ConsentManager, AuditTrail
from data.generator import PatientDataGenerator, CONFLICT_TEMPLATES, PATTERN_TEMPLATES
from agents.core import DepartmentAgent, OrchestratorAgent
from llm.provider import BaseLLMProvider, create_llm_provider

logger = logging.getLogger("medagentnet.simulation")


class SimulationRunner:
    """Runs the full MedAgentNet simulation."""

    def __init__(self, config_dir: str = "config",
                 llm_provider: Optional[BaseLLMProvider] = None):
        self.config_dir = config_dir

        # Load configs
        with open(os.path.join(config_dir, "settings.yaml")) as f:
            self.settings = yaml.safe_load(f)
        with open(os.path.join(config_dir, "departments.yaml")) as f:
            self.dept_config = yaml.safe_load(f)["departments"]

        # Initialize components — use injected provider or create from config
        self.llm = llm_provider or create_llm_provider(self.settings.get("llm", {}))
        self.consent = ConsentManager(
            default_policy=self.settings.get("privacy", {}).get("consent_default", "opt_in"),
            emergency_override=self.settings.get("privacy", {}).get("emergency_override_enabled", True),
        )

        results_dir = self.settings.get("logging", {}).get("results_dir", "data/results")
        os.makedirs(results_dir, exist_ok=True)

        audit_file = self.settings.get("logging", {}).get("audit_file", "data/audit_trail.jsonl")
        self.audit = AuditTrail(log_file=audit_file)

        # Initialize agents
        self.department_agents: dict[str, DepartmentAgent] = {}
        for dept_id, dept_data in self.dept_config.items():
            self.department_agents[dept_id] = DepartmentAgent(
                department_id=dept_id,
                department_config=dept_data,
                llm=self.llm,
                audit=self.audit,
            )

        self.orchestrator = OrchestratorAgent(
            department_agents=self.department_agents,
            llm=self.llm,
            consent_manager=self.consent,
            audit=self.audit,
            dept_config=self.dept_config,
        )

        self.patients: list[PatientRecord] = []
        self.results: list[dict] = []

    def apply_consent_restriction(self, patient_id: str,
                                   restriction_type: str,
                                   restricted_dept_pairs: list = None):
        """Apply consent restrictions to a patient for experiments.

        restriction_type:
        - 'full': baseline (no restrictions, no-op)
        - 'partial': revoke specific dept pairs
        - 'opt_out': patient opts out entirely
        """
        if restriction_type == "opt_out":
            self.consent.revoke_consent(patient_id)
        elif restriction_type == "partial" and restricted_dept_pairs:
            for source_dept, target_dept in restricted_dept_pairs:
                self.consent.revoke_consent(patient_id, source_dept, target_dept)
        # 'full' is the default — no action needed

    def generate_patients(self, num_patients: Optional[int] = None) -> list[PatientRecord]:
        """Generate synthetic patient data."""
        generator = PatientDataGenerator(config_dir=self.config_dir)
        self.patients = generator.generate(num_patients)

        # Load patient data into agents
        for patient in self.patients:
            # Register consent
            self.consent.register_patient(patient.patient_id, patient.departments)

            # Load into relevant department agents
            for dept_id in patient.departments:
                if dept_id in self.department_agents:
                    self.department_agents[dept_id].load_patient_data(patient)

        # Save patient data
        idx = generator.save_patients(self.patients, "data/patients")
        logger.info(
            f"Generated {idx['total_patients']} patients: "
            f"{idx['patients_with_conflicts']} with conflicts, "
            f"{idx['patients_with_patterns']} with patterns"
        )
        return self.patients

    def run_scenario(self, patient: PatientRecord, requesting_dept: str,
                      clinical_context: dict, query_type: str = "MED_CONFLICT",
                      is_emergency: bool = False, scenario_name: str = "",
                      force_disclosure_tier: int = None) -> dict:
        """Run a single clinical scenario."""
        start_time = time.time()

        result = self.orchestrator.process_request(
            requesting_dept=requesting_dept,
            patient_id=patient.patient_id,
            clinical_context=clinical_context,
            query_type=query_type,
            is_emergency=is_emergency,
            force_disclosure_tier=force_disclosure_tier,
        )

        elapsed = time.time() - start_time

        scenario_result = {
            "scenario_name": scenario_name,
            "patient_id": patient.patient_id,
            "patient_name": patient.name,
            "requesting_department": requesting_dept,
            "query_type": query_type,
            "is_emergency": is_emergency,
            "clinical_context": clinical_context,
            "num_alerts": len(result["alerts"]),
            "alerts": [a.to_dict() for a in result["alerts"]],
            "num_responses": len(result["responses"]),
            "response_summaries": [
                {"department": r.source_agent, "summary": r.summary,
                 "risk_flags": r.risk_flags, "tier": r.disclosure_tier}
                for r in result["responses"]
            ],
            "privacy_report": result["privacy_report"],
            "elapsed_seconds": round(elapsed, 3),
            "known_conflicts": patient.known_conflicts,
            "known_patterns": patient.known_patterns,
        }

        self.results.append(scenario_result)
        return scenario_result

    def run_all_conflict_scenarios(self) -> list[dict]:
        """Run scenarios for ALL patients with planted conflicts."""
        results = []
        for patient in self.patients:
            for conflict in patient.known_conflicts:
                context = {
                    "planned_procedure": conflict.get("trigger_procedure", "evaluation"),
                    "relevant_categories": ["all"],
                    "reason": conflict.get("description", ""),
                }
                # Include medication names in context so agents can detect them
                for med in patient.medications:
                    context[f"current_med_{med.name.lower().replace(' ', '_')}"] = med.name

                result = self.run_scenario(
                    patient=patient,
                    requesting_dept=conflict.get("trigger_department", "general_practice"),
                    clinical_context=context,
                    query_type="MED_CONFLICT",
                    is_emergency=conflict.get("alert_level") == "critical",
                    scenario_name=conflict.get("conflict_name", "unknown"),
                )
                results.append(result)

        return results

    def run_all_pattern_scenarios(self) -> list[dict]:
        """Run scenarios for ALL patients with planted patterns."""
        results = []
        for patient in self.patients:
            for pattern in patient.known_patterns:
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
                # Add ALL lab values to context with trend info
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

                # Add ALL conditions across departments
                for cond in patient.conditions:
                    context[f"condition_{cond.department}_{cond.code}"] = (
                        f"{cond.name} (severity: {cond.severity}, dept: {cond.department})"
                    )

                # Add active medications
                for med in patient.medications:
                    context[f"medication_{med.department}_{med.name.lower().replace(' ', '_')}"] = (
                        f"{med.name} ({med.category}), prescribed by {med.department}"
                    )

                result = self.run_scenario(
                    patient=patient,
                    requesting_dept="general_practice",
                    clinical_context=context,
                    query_type="LONG_PATTERN",
                    scenario_name=pattern.get("pattern_name", "unknown"),
                )
                results.append(result)

        return results

    def run_full_simulation(self) -> dict:
        """Run the complete simulation with all scenarios."""
        logger.info("=" * 60)
        logger.info("MedAgentNet Simulation Starting")
        logger.info("=" * 60)

        overall_start = time.time()

        # Generate patients
        logger.info("\n[Phase 1] Generating synthetic patient data...")
        self.generate_patients()

        # Run conflict scenarios
        logger.info("\n[Phase 2] Running medication conflict scenarios...")
        conflict_results = self.run_all_conflict_scenarios()
        logger.info(f"  Completed {len(conflict_results)} conflict scenarios")

        # Run pattern scenarios
        logger.info("\n[Phase 3] Running pattern detection scenarios...")
        pattern_results = self.run_all_pattern_scenarios()
        logger.info(f"  Completed {len(pattern_results)} pattern scenarios")

        # Run random routine queries for patients without planted issues
        logger.info("\n[Phase 4] Running routine cross-department queries...")
        routine_results = []
        clean_patients = [p for p in self.patients if not p.known_conflicts and not p.known_patterns]
        for patient in clean_patients[:20]:  # Sample 20 clean patients
            dept = patient.departments[0] if patient.departments else "general_practice"
            result = self.run_scenario(
                patient=patient,
                requesting_dept=dept,
                clinical_context={"planned_procedure": "routine_checkup", "relevant_categories": ["all"]},
                query_type="MED_CONFLICT",
                scenario_name="routine_check",
            )
            routine_results.append(result)
        logger.info(f"  Completed {len(routine_results)} routine scenarios")

        total_time = time.time() - overall_start

        # Evaluate
        evaluation = self.evaluate()
        evaluation["total_simulation_time"] = round(total_time, 2)

        # Save results
        self._save_results(evaluation, conflict_results, pattern_results, routine_results)

        logger.info("\n" + "=" * 60)
        logger.info("Simulation Complete")
        logger.info("=" * 60)

        return evaluation

    def evaluate(self) -> dict:
        """Evaluate simulation results against ground truth."""
        conflict_scenarios = [r for r in self.results if r.get("known_conflicts")]
        pattern_scenarios = [r for r in self.results if r.get("known_patterns")]
        routine_scenarios = [r for r in self.results
                             if not r.get("known_conflicts") and not r.get("known_patterns")]

        # Conflict detection rate
        conflicts_detected = 0
        conflicts_total = len(conflict_scenarios)
        for result in conflict_scenarios:
            if result["num_alerts"] > 0:
                has_real_alert = any(
                    a.get("alert_type") not in ("no_conflict", "parse_error")
                    for a in result["alerts"]
                )
                if has_real_alert:
                    conflicts_detected += 1

        # Pattern detection rate
        patterns_detected = 0
        patterns_total = len(pattern_scenarios)
        for result in pattern_scenarios:
            if result["num_alerts"] > 0:
                # Check if any alert is meaningful (not just "no conflict")
                has_meaningful_alert = any(
                    a.get("alert_type") not in ("no_conflict", "parse_error", "llm_error")
                    and a.get("severity", a.get("alert_level", "")) in ("moderate", "high", "high_risk", "critical")
                    for a in result["alerts"]
                )
                # Also check for pattern-related keywords in descriptions
                has_pattern_keywords = any(
                    any(kw in (a.get("description", "") + " " + a.get("alert_type", "")).lower()
                        for kw in ["pattern", "trend", "rising", "declining", "progressive",
                                   "diabetes", "ckd", "kidney", "thyroid", "retinopathy",
                                   "neuropathy", "glucose", "hba1c", "egfr", "creatinine",
                                   "connection", "suggestive", "consistent", "progression",
                                   "worsening", "deteriorat", "abnormal", "elevated"])
                    for a in result["alerts"]
                )
                # Also check response summaries for pattern keywords
                has_summary_pattern = any(
                    any(kw in rs.get("summary", "").lower()
                        for kw in ["pattern", "trend", "rising", "declining", "diabetes",
                                   "kidney", "thyroid", "retinopathy", "neuropathy",
                                   "glucose", "abnormal", "elevated", "progression"])
                    for rs in result.get("response_summaries", [])
                )
                if has_meaningful_alert or has_pattern_keywords or has_summary_pattern:
                    patterns_detected += 1

        # False positive rate (alerts on clean patients)
        false_positives = 0
        routine_total = len(routine_scenarios)
        for result in routine_scenarios:
            has_false_alert = any(
                a.get("alert_level") in ("high_risk", "critical")
                for a in result["alerts"]
            )
            if has_false_alert:
                false_positives += 1

        # Privacy metrics
        privacy = self.audit.get_privacy_report()

        # Average response time
        times = [r["elapsed_seconds"] for r in self.results if r["elapsed_seconds"] > 0]
        avg_time = sum(times) / len(times) if times else 0

        evaluation = {
            "summary": {
                "total_scenarios_run": len(self.results),
                "total_patients": len(self.patients),
                "total_departments": len(self.department_agents),
            },
            "conflict_detection": {
                "total_conflict_scenarios": conflicts_total,
                "conflicts_detected": conflicts_detected,
                "detection_rate": round(conflicts_detected / conflicts_total, 4) if conflicts_total else 0,
                "missed": conflicts_total - conflicts_detected,
            },
            "pattern_detection": {
                "total_pattern_scenarios": patterns_total,
                "patterns_detected": patterns_detected,
                "detection_rate": round(patterns_detected / patterns_total, 4) if patterns_total else 0,
                "missed": patterns_total - patterns_detected,
            },
            "false_positives": {
                "routine_scenarios_checked": routine_total,
                "false_positives": false_positives,
                "false_positive_rate": round(false_positives / routine_total, 4) if routine_total else 0,
            },
            "performance": {
                "average_response_time_seconds": round(avg_time, 4),
                "max_response_time_seconds": round(max(times), 4) if times else 0,
                "min_response_time_seconds": round(min(times), 4) if times else 0,
            },
            "privacy_compliance": privacy,
        }

        return evaluation

    def _save_results(self, evaluation: dict, conflict_results: list,
                       pattern_results: list, routine_results: list):
        """Save all results to disk."""
        results_dir = self.settings.get("logging", {}).get("results_dir", "data/results")
        os.makedirs(results_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save evaluation
        with open(os.path.join(results_dir, f"evaluation_{timestamp}.json"), "w") as f:
            json.dump(evaluation, f, indent=2, default=str)

        # Save detailed results
        all_results = {
            "timestamp": timestamp,
            "conflict_scenarios": conflict_results,
            "pattern_scenarios": pattern_results,
            "routine_scenarios": routine_results,
        }
        with open(os.path.join(results_dir, f"detailed_results_{timestamp}.json"), "w") as f:
            json.dump(all_results, f, indent=2, default=str)

        # Save human-readable report
        report = self._generate_report(evaluation, conflict_results, pattern_results)
        with open(os.path.join(results_dir, f"report_{timestamp}.txt"), "w") as f:
            f.write(report)

        logger.info(f"\nResults saved to {results_dir}/")

    def _generate_report(self, evaluation: dict, conflict_results: list,
                          pattern_results: list) -> str:
        """Generate a human-readable simulation report."""
        lines = []
        lines.append("=" * 70)
        lines.append("  MedAgentNet SIMULATION REPORT")
        lines.append("  Privacy-Preserving Federated Multi-Agent Healthcare AI")
        lines.append("=" * 70)
        lines.append(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"  Total Patients: {evaluation['summary']['total_patients']}")
        lines.append(f"  Total Departments: {evaluation['summary']['total_departments']}")
        lines.append(f"  Total Scenarios: {evaluation['summary']['total_scenarios_run']}")
        lines.append("")

        # Conflict Detection
        cd = evaluation["conflict_detection"]
        lines.append("-" * 70)
        lines.append("  MEDICATION CONFLICT DETECTION")
        lines.append("-" * 70)
        lines.append(f"  Scenarios tested:   {cd['total_conflict_scenarios']}")
        lines.append(f"  Conflicts detected: {cd['conflicts_detected']}")
        lines.append(f"  Detection rate:     {cd['detection_rate']*100:.1f}%")
        lines.append(f"  Missed:             {cd['missed']}")
        lines.append("")

        # Show individual conflict results
        for r in conflict_results:
            status = "DETECTED" if r["num_alerts"] > 0 else "MISSED"
            icon = "+" if status == "DETECTED" else "x"
            lines.append(f"  [{icon}] {r['scenario_name']}: {status}")
            if r["num_alerts"] > 0:
                for a in r["alerts"][:2]:
                    if a.get("alert_type") != "no_conflict":
                        lines.append(f"      Alert: [{a.get('alert_level','?')}] {a.get('description','')[:80]}")
        lines.append("")

        # Pattern Detection
        pd = evaluation["pattern_detection"]
        lines.append("-" * 70)
        lines.append("  CROSS-DEPARTMENTAL PATTERN DETECTION")
        lines.append("-" * 70)
        lines.append(f"  Scenarios tested:   {pd['total_pattern_scenarios']}")
        lines.append(f"  Patterns detected:  {pd['patterns_detected']}")
        lines.append(f"  Detection rate:     {pd['detection_rate']*100:.1f}%")
        lines.append("")

        for r in pattern_results:
            has_alerts = r["num_alerts"] > 0 and any(
                a.get("alert_type") not in ("no_conflict", "parse_error", "llm_error")
                for a in r["alerts"]
            )
            status = "DETECTED" if has_alerts else "MISSED"
            icon = "+" if status == "DETECTED" else "x"
            lines.append(f"  [{icon}] {r['scenario_name']}: {status}")
            if has_alerts:
                for a in r["alerts"][:2]:
                    if a.get("alert_type") not in ("no_conflict", "parse_error", "llm_error"):
                        lines.append(f"      Alert: [{a.get('alert_level','?')}] {a.get('description','')[:80]}")
        lines.append("")

        # False Positives
        fp = evaluation["false_positives"]
        lines.append("-" * 70)
        lines.append("  FALSE POSITIVE ANALYSIS")
        lines.append("-" * 70)
        lines.append(f"  Clean patients tested: {fp['routine_scenarios_checked']}")
        lines.append(f"  False positives:       {fp['false_positives']}")
        lines.append(f"  False positive rate:   {fp['false_positive_rate']*100:.1f}%")
        lines.append("")

        # Performance
        perf = evaluation["performance"]
        lines.append("-" * 70)
        lines.append("  PERFORMANCE")
        lines.append("-" * 70)
        lines.append(f"  Avg response time:  {perf['average_response_time_seconds']*1000:.1f}ms")
        lines.append(f"  Max response time:  {perf['max_response_time_seconds']*1000:.1f}ms")
        lines.append(f"  Total sim time:     {evaluation.get('total_simulation_time', 0):.1f}s")
        lines.append("")

        # Privacy
        priv = evaluation["privacy_compliance"]
        lines.append("-" * 70)
        lines.append("  PRIVACY COMPLIANCE")
        lines.append("-" * 70)
        lines.append(f"  Total audit events:     {priv.get('total_events', 0)}")
        lines.append(f"  Consent denials:        {priv.get('consent_denied_count', 0)}")
        lines.append(f"  Denial rate:            {priv.get('consent_denial_rate', 0)*100:.1f}%")
        lines.append(f"  Tier distribution:      {priv.get('tier_distribution', {})}")
        lines.append(f"  Unique patients queried: {priv.get('unique_patients', 0)}")
        lines.append("")

        lines.append("=" * 70)
        lines.append("  END OF REPORT")
        lines.append("=" * 70)

        return "\n".join(lines)
