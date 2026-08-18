#!/usr/bin/env python3
"""
MedAgentNet - R1 experiment driver
==================================

Examples
--------
  # validate the whole suite without any model server (fast)
  python run_r1.py --provider mock --patients 150

  # the reported run
  python run_r1.py --provider ollama --model llama3.1:8b --patients 150 \
                   --experiments e9 e1 e2 e3 e4 e5 e7 e8

  # backend matrix (each model must already be pulled)
  python run_r1.py --provider ollama --model llama3.1:8b \
                   --backend-matrix llama3.1:8b qwen2.5:7b mistral:7b phi3:mini \
                   --experiments e6

Notes
-----
* The provider is never silently substituted. If a model is not reachable the
  run stops rather than producing Mock numbers under an LLM label.
* Every run writes results.json and tables.tex into its own directory; the
  manuscript tables are generated from that file, not transcribed by hand.
"""
from __future__ import annotations

import os
import sys
import json
import logging
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm.provider import (
    MockLLMProvider, OllamaProvider, OpenAICompatibleProvider,
    HuggingFaceProvider, create_llm_provider, ProviderUnavailable,
)
from simulation.experiments_v2 import R1Experiments

ALL_EXPERIMENTS = ["e9", "e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8"]
DEFAULT_EXPERIMENTS = ["e9", "e1", "e2", "e3", "e4", "e5", "e7", "e8"]


def build_provider(args):
    if args.provider == "mock":
        return MockLLMProvider()
    if args.provider == "ollama":
        p = OllamaProvider(base_url=args.ollama_url, model=args.model,
                           temperature=args.temperature,
                           max_tokens=args.max_tokens)
        if not p.is_available():
            raise ProviderUnavailable(
                f"Ollama model '{args.model}' is not available at "
                f"{args.ollama_url}. Start `ollama serve` and run "
                f"`ollama pull {args.model}`.")
        return p
    if args.provider == "openai_compatible":
        p = OpenAICompatibleProvider(base_url=args.openai_url, model=args.model,
                                     temperature=args.temperature,
                                     max_tokens=args.max_tokens)
        if not p.is_available():
            raise ProviderUnavailable(f"No endpoint at {args.openai_url}")
        return p
    if args.provider == "huggingface":
        return HuggingFaceProvider(model_id=args.model)
    return create_llm_provider({"provider": args.provider})


def build_backend_matrix(args) -> dict:
    out = {}
    for model in args.backend_matrix or []:
        p = OllamaProvider(base_url=args.ollama_url, model=model,
                           temperature=args.temperature, max_tokens=args.max_tokens)
        if not p.is_available():
            raise ProviderUnavailable(
                f"Backend matrix: '{model}' is not pulled. Run `ollama pull {model}`.")
        out[model] = p
    if args.include_mock_in_matrix:
        out["mock_rule_based"] = MockLLMProvider()
    return out


def main():
    ap = argparse.ArgumentParser(description="MedAgentNet R1 experiments")
    ap.add_argument("--provider", default="mock",
                    choices=["mock", "ollama", "openai_compatible", "huggingface"])
    ap.add_argument("--model", default="llama3.1:8b")
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--openai-url", default="http://localhost:8000/v1")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--max-tokens", type=int, default=1024)

    ap.add_argument("--patients", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--experiments", nargs="+", default=DEFAULT_EXPERIMENTS,
                    choices=ALL_EXPERIMENTS)
    ap.add_argument("--backend-matrix", nargs="*", default=None,
                    help="model names for E6; each must already be pulled")
    ap.add_argument("--include-mock-in-matrix", action="store_true")
    ap.add_argument("--out-dir", default="data/results_r1")
    ap.add_argument("--tag", default="")
    ap.add_argument("--config-dir", default="config")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s | %(name)-28s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("medagentnet.agents").setLevel(logging.WARNING)

    try:
        provider = build_provider(args)
        matrix = build_backend_matrix(args) if args.backend_matrix else None
    except ProviderUnavailable as e:
        print(f"\nERROR: {e}\n")
        return 2

    print("=" * 72)
    print("  MedAgentNet - R1 experiment suite")
    print("=" * 72)
    print(f"  backend      : {provider.describe()}")
    print(f"  patients     : {args.patients}")
    print(f"  base seed    : {args.seed}")
    print(f"  experiments  : {', '.join(args.experiments)}")
    if matrix:
        print(f"  backend matrix: {', '.join(matrix)}")
    print("=" * 72)

    exp = R1Experiments(config_dir=args.config_dir, llm_provider=provider,
                        base_seed=args.seed, num_patients=args.patients,
                        out_dir=args.out_dir)
    results = exp.run_all(which=args.experiments, providers=matrix)
    path = exp.save(results, tag=args.tag or args.provider)

    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    for key in args.experiments:
        block = results.get(key, {})
        if "error" in block:
            print(f"  [{key}] FAILED: {block['error']}")
            continue
        print(f"  [{key}] {block.get('experiment', '')} "
              f"({block.get('_wall_seconds', 0)}s)")
    print(f"\n  results : {os.path.join(path, 'results.json')}")
    print(f"  tables  : {os.path.join(path, 'tables.tex')}")
    print(f"  total   : {results['meta']['total_seconds']}s")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
