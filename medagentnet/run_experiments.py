#!/usr/bin/env python3
"""
MedAgentNet Experiments Runner
==============================
Executes comprehensive experiments for the academic paper Results section.

Usage:
    python run_experiments.py                        # Run all experiments (Mock only)
    python run_experiments.py --tier-only            # Only tier comparison
    python run_experiments.py --consent-only         # Only consent restriction
    python run_experiments.py --variance-only        # Only multi-seed variance
    python run_experiments.py --scalability-only     # Only scalability analysis
    python run_experiments.py --compare-providers    # Run Mock vs real LLM comparison
    python run_experiments.py --provider ollama      # Use Ollama instead of Mock
    python run_experiments.py --seeds 10             # Custom seed count
    python run_experiments.py --patients 50 100 250  # Custom patient counts
    python run_experiments.py --base-patients 200    # Base patient count
    python run_experiments.py --comparison-runs 5    # Runs per provider for comparison
    python run_experiments.py --verbose              # Debug output

Provider comparison requires a real LLM configured in config/settings.yaml:
    llm:
      provider: ollama          # or openai_compatible, huggingface
      ollama:
        model: llama3.1:8b
        base_url: http://localhost:11434
"""
import os
import sys
import json
import shutil
import argparse
import logging
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulation.experiments import ExperimentRunner
from llm.provider import (
    BaseLLMProvider, MockLLMProvider, OllamaProvider,
    OpenAICompatibleProvider, create_llm_provider,
)


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)-32s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )


def print_banner():
    print("""
    ================================================================
       MEDAGENTNET — Comprehensive Experiment Suite
       Privacy-Preserving Federated Multi-Agent Healthcare AI
    ================================================================
    """)


def resolve_provider(args, settings: dict) -> BaseLLMProvider | None:
    """Resolve an LLM provider from CLI args, returning None for default (config-based)."""
    provider_name = args.provider
    if provider_name is None or provider_name == "config":
        return None  # Use whatever settings.yaml says (default path)

    if provider_name == "mock":
        return MockLLMProvider()

    llm_config = settings.get("llm", {})

    if provider_name == "ollama":
        cfg = llm_config.get("ollama", {})
        if args.model:
            cfg["model"] = args.model
        provider = OllamaProvider(**cfg)
        if not provider.is_available():
            print(f"  ERROR: Ollama not available at {cfg.get('base_url', 'localhost:11434')}")
            print(f"         Make sure Ollama is running: ollama serve")
            print(f"         And a model is pulled: ollama pull {cfg.get('model', 'llama3.1:8b')}")
            sys.exit(1)
        return provider

    if provider_name == "openai_compatible":
        cfg = llm_config.get("openai_compatible", {})
        if args.model:
            cfg["model"] = args.model
        provider = OpenAICompatibleProvider(**cfg)
        if not provider.is_available():
            print(f"  ERROR: OpenAI-compatible API not available at {cfg.get('base_url', '')}")
            sys.exit(1)
        return provider

    # For huggingface or unknown, fall back to config-based creation
    llm_config_override = dict(llm_config)
    llm_config_override["provider"] = provider_name
    return create_llm_provider(llm_config_override)


def save_results(results: dict, runner: 'ExperimentRunner', args) -> str:
    """Save all results in an organized directory structure.

    Creates:
        data/experiment_results/run_YYYYMMDD_HHMMSS/
            ├── results.json           # Full JSON results
            ├── tables.tex             # LaTeX tables for paper
            ├── report.txt             # Human-readable report
            ├── config_snapshot.yaml   # Config used for this run
            └── README.txt             # Run metadata

    Returns the path to the run directory.
    """
    import yaml

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build a descriptive folder name: run_<provider>_<patients>p_<seeds>s_<timestamp>
    provider_tag = "mock"
    if args.provider and args.provider != "config":
        provider_tag = args.provider
        if args.model:
            provider_tag += f"_{args.model.replace(':', '-').replace('/', '-')}"
    run_dir = os.path.join(
        runner.results_dir,
        f"run_{provider_tag}_{args.base_patients}p_{args.seeds}s_{timestamp}",
    )
    os.makedirs(run_dir, exist_ok=True)

    # 1. Save full JSON results
    json_path = os.path.join(run_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # 2. Generate and save LaTeX tables
    latex = runner.generate_latex_tables(results)
    latex_path = os.path.join(run_dir, "tables.tex")
    with open(latex_path, "w") as f:
        f.write(latex)

    # 3. Generate human-readable report
    report = generate_report(results, args)
    report_path = os.path.join(run_dir, "report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    # 4. Snapshot the config used
    settings_path = os.path.join(args.config_dir, "settings.yaml")
    if os.path.exists(settings_path):
        shutil.copy2(settings_path, os.path.join(run_dir, "config_snapshot.yaml"))

    # 5. Write run metadata
    provider_info = "from config"
    if args.provider:
        provider_info = args.provider
        if args.model:
            provider_info += f" ({args.model})"
    readme = (
        f"MedAgentNet Experiment Run\n"
        f"{'=' * 40}\n"
        f"Timestamp:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Provider:        {provider_info}\n"
        f"Base patients:   {args.base_patients}\n"
        f"Seeds:           {args.seeds}\n"
        f"Scalability:     {args.patients or [50, 100, 250, 500]}\n"
        f"Comparison runs: {args.comparison_runs}\n"
        f"\nExperiments run:\n"
    )
    for key in ["tier_comparison", "consent_restriction", "multi_seed_variance",
                 "scalability", "provider_comparison"]:
        if key in results:
            status = "ERROR" if isinstance(results[key], dict) and "error" in results[key] else "OK"
            readme += f"  - {key}: {status}\n"

    total = results.get("total_experiment_time_s", 0)
    readme += f"\nTotal time: {total:.1f}s\n"

    readme_path = os.path.join(run_dir, "README.txt")
    with open(readme_path, "w") as f:
        f.write(readme)

    # 6. Also keep a "latest" symlink / copy for convenience
    latest_json = os.path.join(runner.results_dir, "latest_results.json")
    latest_tex = os.path.join(runner.results_dir, "latest_tables.tex")
    shutil.copy2(json_path, latest_json)
    shutil.copy2(latex_path, latest_tex)

    return run_dir


def generate_report(results: dict, args) -> str:
    """Generate a comprehensive human-readable text report."""
    lines = []
    lines.append("=" * 72)
    lines.append("  MEDAGENTNET — EXPERIMENT RESULTS REPORT")
    lines.append("  Privacy-Preserving Federated Multi-Agent Healthcare AI")
    lines.append("=" * 72)
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Base patients: {args.base_patients}  |  Seeds: {args.seeds}")
    lines.append("")

    # ── Experiment 1: Tier Comparison ──
    if "tier_comparison" in results:
        tc = results["tier_comparison"]["results_by_tier"]
        lines.append("-" * 72)
        lines.append("  EXPERIMENT 1: Disclosure Tier Comparison")
        lines.append("-" * 72)
        lines.append(f"  {'Tier':<25} {'Conflict Det.':<18} {'Pattern Det.':<18} {'Avg Time (ms)'}")
        lines.append(f"  {'-'*25} {'-'*18} {'-'*18} {'-'*15}")
        for tier in [1, 2, 3]:
            d = tc.get(tier, tc.get(str(tier), {}))
            tier_name = {1: "Tier 1 (Flag Only)", 2: "Tier 2 (Clinical)", 3: "Tier 3 (Full)"}[tier]
            cd = d.get("conflict_detection", {}).get("detection_rate", 0)
            pd_r = d.get("pattern_detection", {}).get("detection_rate", 0)
            rt = d.get("performance", {}).get("average_response_time_seconds", 0)
            lines.append(f"  {tier_name:<25} {cd*100:>10.1f}%       {pd_r*100:>10.1f}%       {rt*1000:>8.1f}")
        lines.append("")

    # ── Experiment 2: Consent Restriction ──
    if "consent_restriction" in results:
        cs = results["consent_restriction"]["results_by_scenario"]
        lines.append("-" * 72)
        lines.append("  EXPERIMENT 2: Consent Restriction Impact")
        lines.append("-" * 72)
        lines.append(f"  {'Scenario':<25} {'Conflict Det.':<18} {'Pattern Det.'}")
        lines.append(f"  {'-'*25} {'-'*18} {'-'*18}")
        for scenario in ["full", "partial_30", "partial_60", "opt_out"]:
            d = cs.get(scenario, {})
            label = {"full": "Full Consent", "partial_30": "30% Restricted",
                     "partial_60": "60% Restricted", "opt_out": "Full Opt-Out"}.get(scenario, scenario)
            cd = d.get("conflict_detection", {}).get("detection_rate", 0)
            pd_r = d.get("pattern_detection", {}).get("detection_rate", 0)
            lines.append(f"  {label:<25} {cd*100:>10.1f}%       {pd_r*100:>10.1f}%")
        lines.append("")

    # ── Experiment 3: Multi-Seed Variance ──
    if "multi_seed_variance" in results:
        mv = results["multi_seed_variance"]
        agg = mv["aggregate"]
        lines.append("-" * 72)
        lines.append(f"  EXPERIMENT 3: Multi-Seed Variance ({mv['num_seeds']} seeds)")
        lines.append("-" * 72)
        lines.append(f"  {'Metric':<30} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
        lines.append(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
        for key, label in [
            ("conflict_detection_rate", "Conflict Detection Rate"),
            ("pattern_detection_rate", "Pattern Detection Rate"),
            ("false_positive_rate", "False Positive Rate"),
        ]:
            vals = agg.get(key, {})
            lines.append(
                f"  {label:<30} {vals.get('mean',0)*100:>8.1f}% {vals.get('std',0)*100:>8.1f}% "
                f"{vals.get('min',0)*100:>8.1f}% {vals.get('max',0)*100:>8.1f}%"
            )
        rt = agg.get("avg_response_time_s", {})
        lines.append(
            f"  {'Avg Response Time (ms)':<30} {rt.get('mean',0)*1000:>8.1f}  {rt.get('std',0)*1000:>8.1f}  "
            f"{rt.get('min',0)*1000:>8.1f}  {rt.get('max',0)*1000:>8.1f}"
        )

        # Per-scenario breakdown
        if mv.get("per_conflict_aggregate"):
            lines.append("")
            lines.append("  Per-Conflict Detection Rates:")
            for ctype, vals in sorted(mv["per_conflict_aggregate"].items()):
                name = ctype.replace("_", " ").title()
                lines.append(f"    {name:<40} {vals['mean']*100:.1f}% +/- {vals['std']*100:.1f}%")

        if mv.get("per_pattern_aggregate"):
            lines.append("")
            lines.append("  Per-Pattern Detection Rates:")
            for ptype, vals in sorted(mv["per_pattern_aggregate"].items()):
                name = ptype.replace("_", " ").title()
                lines.append(f"    {name:<40} {vals['mean']*100:.1f}% +/- {vals['std']*100:.1f}%")
        lines.append("")

    # ── Experiment 4: Scalability ──
    if "scalability" in results:
        sc = results["scalability"]["results_by_count"]
        lines.append("-" * 72)
        lines.append("  EXPERIMENT 4: Scalability Analysis")
        lines.append("-" * 72)
        lines.append(f"  {'Patients':>8} {'Scenarios':>10} {'Conflict':>10} {'Pattern':>10} {'FP Rate':>10} {'Time(s)':>10}")
        lines.append(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
        for count in sorted(sc.keys(), key=lambda x: int(x)):
            d = sc[count]
            lines.append(
                f"  {int(count):>8} {d.get('total_scenarios', 0):>10} "
                f"{d['conflict_detection_rate']*100:>8.1f}% "
                f"{d['pattern_detection_rate']*100:>8.1f}% "
                f"{d['false_positive_rate']*100:>8.1f}% "
                f"{d['total_simulation_time_s']:>9.1f}"
            )
        lines.append("")

    # ── Experiment 5: Provider Comparison ──
    if "provider_comparison" in results:
        pc = results["provider_comparison"]
        lines.append("-" * 72)
        lines.append("  EXPERIMENT 5: LLM Provider Comparison")
        lines.append("-" * 72)
        if "error" in pc:
            lines.append(f"  {pc['error']}")
        else:
            mock_agg = pc["mock"]["aggregate"]
            real_agg = pc["real"]["aggregate"]
            rp = pc["real_provider"]
            lines.append(f"  Mock (rule-based) vs {rp}")
            lines.append(f"  Runs: {pc['num_runs']}  |  Patients/run: {pc['num_patients']}")
            lines.append("")
            lines.append(f"  {'Metric':<30} {'MockLLM':>15} {rp[:20]:>20}")
            lines.append(f"  {'-'*30} {'-'*15} {'-'*20}")
            for key, label in [
                ("conflict_detection_rate", "Conflict Detection"),
                ("pattern_detection_rate", "Pattern Detection"),
                ("false_positive_rate", "False Positive Rate"),
                ("avg_response_time_s", "Avg Response Time"),
            ]:
                m = mock_agg[key]
                r = real_agg[key]
                if key == "avg_response_time_s":
                    lines.append(f"  {label:<30} {m['mean']*1000:>11.1f}ms  {r['mean']*1000:>16.1f}ms")
                else:
                    lines.append(f"  {label:<30} {m['mean']*100:>12.1f}%  {r['mean']*100:>17.1f}%")
            agr = pc["agreement"]
            lines.append(f"\n  Scenario Agreement: {agr['agreement_rate']*100:.1f}%")
            if agr.get("disagreed_scenarios"):
                lines.append(f"  Disagreements ({len(agr['disagreed_scenarios'])}):")
                for d in agr["disagreed_scenarios"][:10]:
                    lines.append(
                        f"    seed={d['seed']} {d['scenario']}: "
                        f"mock={'detected' if d['mock_detected'] else 'missed'} "
                        f"real={'detected' if d['real_detected'] else 'missed'}"
                    )
        lines.append("")

    # ── Footer ──
    total = results.get("total_experiment_time_s", 0)
    lines.append("=" * 72)
    lines.append(f"  Total experiment time: {total:.1f}s")
    lines.append("=" * 72)

    return "\n".join(lines)


def print_summary(results: dict):
    """Print a human-readable summary of all results."""
    print("\n" + "=" * 64)
    print("  EXPERIMENT RESULTS SUMMARY")
    print("=" * 64)

    # Tier comparison
    if "tier_comparison" in results:
        tc = results["tier_comparison"]["results_by_tier"]
        print("\n  [1] Disclosure Tier Comparison")
        for tier in [1, 2, 3]:
            d = tc.get(tier, tc.get(str(tier), {}))
            cd = d.get("conflict_detection", {}).get("detection_rate", 0)
            pd_rate = d.get("pattern_detection", {}).get("detection_rate", 0)
            print(f"      Tier {tier}: Conflict={cd * 100:.1f}%  Pattern={pd_rate * 100:.1f}%")

    # Consent restriction
    if "consent_restriction" in results:
        cs = results["consent_restriction"]["results_by_scenario"]
        print("\n  [2] Consent Restriction Impact")
        for scenario in ["full", "partial_30", "partial_60", "opt_out"]:
            d = cs.get(scenario, {})
            cd = d.get("conflict_detection", {}).get("detection_rate", 0)
            pd_rate = d.get("pattern_detection", {}).get("detection_rate", 0)
            print(f"      {scenario:>12}: Conflict={cd * 100:.1f}%  Pattern={pd_rate * 100:.1f}%")

    # Multi-seed variance
    if "multi_seed_variance" in results:
        mv = results["multi_seed_variance"]["aggregate"]
        print(f"\n  [3] Multi-Seed Variance ({results['multi_seed_variance']['num_seeds']} seeds)")
        for key in ["conflict_detection_rate", "pattern_detection_rate", "false_positive_rate"]:
            vals = mv.get(key, {})
            label = key.replace("_", " ").title()
            print(f"      {label}: {vals.get('mean', 0) * 100:.1f}% +/- {vals.get('std', 0) * 100:.1f}%")

    # Scalability
    if "scalability" in results:
        sc = results["scalability"]["results_by_count"]
        print("\n  [4] Scalability Analysis")
        for count in sorted(sc.keys(), key=lambda x: int(x)):
            d = sc[count]
            cd = d["conflict_detection_rate"]
            pd_rate = d["pattern_detection_rate"]
            t = d["total_simulation_time_s"]
            print(f"      {int(count):>4} patients: Conflict={cd * 100:.1f}%  Pattern={pd_rate * 100:.1f}%  Time={t:.1f}s")

    # Provider comparison
    if "provider_comparison" in results:
        pc = results["provider_comparison"]
        if "error" in pc:
            print(f"\n  [5] Provider Comparison: {pc['error']}")
        else:
            print(f"\n  [5] Provider Comparison (Mock vs {pc['real_provider']})")
            mock_agg = pc["mock"]["aggregate"]
            real_agg = pc["real"]["aggregate"]
            print(f"      {'Metric':<30} {'MockLLM':>12}  {pc['real_provider'][:15]:>15}")
            print(f"      {'-' * 60}")
            for key, label in [
                ("conflict_detection_rate", "Conflict Detection"),
                ("pattern_detection_rate", "Pattern Detection"),
                ("false_positive_rate", "False Positive Rate"),
                ("avg_response_time_s", "Avg Response Time"),
            ]:
                m = mock_agg[key]
                r = real_agg[key]
                if key == "avg_response_time_s":
                    print(f"      {label:<30} {m['mean']*1000:>9.1f}ms   {r['mean']*1000:>12.1f}ms")
                else:
                    print(f"      {label:<30} {m['mean']*100:>10.1f}%   {r['mean']*100:>13.1f}%")

            agr = pc["agreement"]
            print(f"      {'Agreement Rate':<30} {agr['agreement_rate']*100:>10.1f}%")
            if agr["disagreed_scenarios"]:
                print(f"      Disagreements ({len(agr['disagreed_scenarios'])}):")
                for d in agr["disagreed_scenarios"][:5]:
                    print(f"        seed={d['seed']} {d['scenario']}: "
                          f"mock={'Y' if d['mock_detected'] else 'N'} "
                          f"real={'Y' if d['real_detected'] else 'N'}")

    total = results.get("total_experiment_time_s", 0)
    print(f"\n  Total experiment time: {total:.1f}s")
    print("=" * 64)


def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive experiments for MedAgentNet academic paper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Experiment selection
    exp_group = parser.add_argument_group("Experiment Selection")
    exp_group.add_argument("--tier-only", action="store_true",
                           help="Run only tier comparison experiment")
    exp_group.add_argument("--consent-only", action="store_true",
                           help="Run only consent restriction experiment")
    exp_group.add_argument("--variance-only", action="store_true",
                           help="Run only multi-seed variance experiment")
    exp_group.add_argument("--scalability-only", action="store_true",
                           help="Run only scalability experiment")
    exp_group.add_argument("--compare-providers", action="store_true",
                           help="Run Mock vs real LLM provider comparison")

    # LLM provider
    llm_group = parser.add_argument_group("LLM Provider")
    llm_group.add_argument("--provider", type=str, default=None,
                           choices=["mock", "ollama", "openai_compatible", "huggingface", "config"],
                           help="LLM provider to use for experiments (default: from config)")
    llm_group.add_argument("--model", type=str, default=None,
                           help="Override model name for the chosen provider")

    # Experiment parameters
    param_group = parser.add_argument_group("Parameters")
    param_group.add_argument("--seeds", type=int, default=5,
                             help="Number of seeds for variance experiment (default: 5)")
    param_group.add_argument("--patients", type=int, nargs="+", default=None,
                             help="Patient counts for scalability (default: 50 100 250 500)")
    param_group.add_argument("--base-patients", type=int, default=100,
                             help="Base patient count for tier/consent/variance experiments (default: 100)")
    param_group.add_argument("--comparison-runs", type=int, default=3,
                             help="Runs per provider for comparison experiment (default: 3)")
    param_group.add_argument("--config-dir", type=str, default="config",
                             help="Path to config directory")
    param_group.add_argument("--verbose", action="store_true",
                             help="Enable debug logging")

    args = parser.parse_args()

    setup_logging(args.verbose)
    print_banner()

    # Load settings for provider resolution
    import yaml
    settings_path = os.path.join(args.config_dir, "settings.yaml")
    with open(settings_path) as f:
        settings = yaml.safe_load(f)

    # Resolve LLM provider
    llm_provider = resolve_provider(args, settings)
    if llm_provider:
        provider_name = type(llm_provider).__name__
        print(f"  Using LLM provider: {provider_name}")
        if hasattr(llm_provider, 'model'):
            print(f"  Model: {llm_provider.model}")
    else:
        print(f"  Using LLM provider from config: {settings.get('llm', {}).get('provider', 'mock')}")

    runner = ExperimentRunner(config_dir=args.config_dir, llm_provider=llm_provider)

    any_specific = any([
        args.tier_only, args.consent_only,
        args.variance_only, args.scalability_only,
        args.compare_providers,
    ])

    import time as _time
    overall_start = _time.time()

    if not any_specific:
        # Run all (without provider comparison by default — it's slow with real LLMs)
        print("  Running ALL experiments...\n")
        results = runner.run_all_experiments(
            num_patients=args.base_patients,
            num_seeds=args.seeds,
            patient_counts=args.patients,
        )
    else:
        results = {"timestamp": datetime.now().isoformat()}

        if args.tier_only:
            print("  Running tier comparison experiment...\n")
            results["tier_comparison"] = runner.run_tier_comparison(args.base_patients)

        if args.consent_only:
            print("  Running consent restriction experiment...\n")
            results["consent_restriction"] = runner.run_consent_restriction(args.base_patients)

        if args.variance_only:
            print(f"  Running multi-seed variance ({args.seeds} seeds)...\n")
            results["multi_seed_variance"] = runner.run_multi_seed_variance(
                args.seeds, args.base_patients
            )

        if args.scalability_only:
            counts = args.patients or [50, 100, 250, 500]
            print(f"  Running scalability analysis ({counts})...\n")
            results["scalability"] = runner.run_scalability(counts)

        if args.compare_providers:
            print(f"  Running provider comparison ({args.comparison_runs} runs)...\n")
            results["provider_comparison"] = runner.run_provider_comparison(
                args.base_patients, args.comparison_runs
            )

    results["total_experiment_time_s"] = round(_time.time() - overall_start, 2)

    # Print summary to console
    print_summary(results)

    # Save all results in organized directory
    run_dir = save_results(results, runner, args)
    print(f"\n  All results saved to: {run_dir}/")
    print(f"    results.json       — Full experiment data")
    print(f"    tables.tex         — LaTeX tables (paste into paper)")
    print(f"    report.txt         — Human-readable report")
    print(f"    config_snapshot.yaml — Config used for this run")
    print(f"\n  Latest results also at:")
    print(f"    {runner.results_dir}/latest_results.json")
    print(f"    {runner.results_dir}/latest_tables.tex")
    print()


if __name__ == "__main__":
    main()
