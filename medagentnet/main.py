#!/usr/bin/env python3
"""
MedAgentNet - Main Entry Point
===================================
Privacy-Preserving Federated Multi-Agent Healthcare AI Simulation

Usage:
    python main.py                    # Run full simulation
    python main.py --patients 200     # Custom patient count
    python main.py --provider ollama  # Use Ollama LLM
    python main.py --verbose          # Debug output
"""
import os
import sys
import argparse
import logging
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulation.runner import SimulationRunner


def setup_logging(level: str = "INFO"):
    """Configure logging for the simulation."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(name)-28s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # Reduce noise from libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def print_banner():
    """Print the MedAgentNet banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║   ███╗   ███╗███████╗██████╗  █████╗  ██████╗ ███████╗      ║
    ║   ████╗ ████║██╔════╝██╔══██╗██╔══██╗██╔════╝ ██╔════╝      ║
    ║   ██╔████╔██║█████╗  ██║  ██║███████║██║  ███╗█████╗        ║
    ║   ██║╚██╔╝██║██╔══╝  ██║  ██║██╔══██║██║   ██║██╔══╝        ║
    ║   ██║ ╚═╝ ██║███████╗██████╔╝██║  ██║╚██████╔╝███████╗      ║
    ║   ╚═╝     ╚═╝╚══════╝╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝      ║
    ║                    N T  N E T                                ║
    ║                                                              ║
    ║   Privacy-Preserving Federated Multi-Agent Healthcare AI     ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def main():
    parser = argparse.ArgumentParser(
        description="MedAgentNet - Federated Multi-Agent Healthcare AI Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          Run full simulation (100 patients, mock LLM)
  python main.py --patients 500           Generate 500 patients
  python main.py --provider ollama        Use local Ollama model
  python main.py --provider openai_compatible  Use vLLM/LMStudio
  python main.py --verbose                Enable debug logging
  python main.py --scenario conflict      Run only conflict detection
  python main.py --scenario pattern       Run only pattern detection
        """,
    )
    parser.add_argument("--patients", type=int, default=None,
                        help="Number of patients to generate (default: from config)")
    parser.add_argument("--provider", type=str, default=None,
                        choices=["mock", "ollama", "huggingface", "openai_compatible"],
                        help="LLM provider to use (default: from config)")
    parser.add_argument("--scenario", type=str, default="all",
                        choices=["all", "conflict", "pattern", "routine"],
                        help="Which scenarios to run")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable debug logging")
    parser.add_argument("--config-dir", type=str, default="config",
                        help="Path to config directory")

    args = parser.parse_args()

    # Setup
    setup_logging("DEBUG" if args.verbose else "INFO")
    print_banner()

    # Override settings if CLI args provided
    config_dir = args.config_dir
    if args.provider or args.patients:
        import yaml
        settings_path = os.path.join(config_dir, "settings.yaml")
        with open(settings_path) as f:
            settings = yaml.safe_load(f)

        if args.provider:
            settings["llm"]["provider"] = args.provider
            print(f"  LLM Provider: {args.provider}")

        if args.patients:
            settings["simulation"]["num_patients"] = args.patients
            print(f"  Patients: {args.patients}")

        with open(settings_path, "w") as f:
            yaml.dump(settings, f, default_flow_style=False)

    # Run simulation
    runner = SimulationRunner(config_dir=config_dir)

    if args.scenario == "all":
        evaluation = runner.run_full_simulation()
    else:
        # Generate patients first
        runner.generate_patients(args.patients)

        if args.scenario == "conflict":
            results = runner.run_all_conflict_scenarios()
            print(f"\n  Completed {len(results)} conflict scenarios")
        elif args.scenario == "pattern":
            results = runner.run_all_pattern_scenarios()
            print(f"\n  Completed {len(results)} pattern scenarios")

        evaluation = runner.evaluate()

    # Print summary
    print("\n")
    print("  ┌─────────────────────────────────────────────┐")
    print("  │           SIMULATION RESULTS SUMMARY         │")
    print("  ├─────────────────────────────────────────────┤")

    cd = evaluation.get("conflict_detection", {})
    pd_eval = evaluation.get("pattern_detection", {})
    fp = evaluation.get("false_positives", {})
    perf = evaluation.get("performance", {})

    print(f"  │  Conflict Detection Rate:  {cd.get('detection_rate', 0)*100:6.1f}%          │")
    print(f"  │  Pattern Detection Rate:   {pd_eval.get('detection_rate', 0)*100:6.1f}%          │")
    print(f"  │  False Positive Rate:      {fp.get('false_positive_rate', 0)*100:6.1f}%          │")
    print(f"  │  Avg Response Time:        {perf.get('average_response_time_seconds', 0)*1000:6.1f}ms       │")
    print(f"  │  Total Simulation Time:    {evaluation.get('total_simulation_time', 0):6.1f}s        │")
    print("  └─────────────────────────────────────────────┘")
    print()
    print("  Results saved to: data/results/")
    print("  Audit trail at:   data/audit_trail.jsonl")
    print()

    return evaluation


if __name__ == "__main__":
    main()
