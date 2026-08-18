#!/usr/bin/env python3
"""Measure how much parallelism the model server actually delivers.

Ollama's server-side concurrency is set by OLLAMA_NUM_PARALLEL *in the server's
own environment*. Exporting it in the shell you launch the experiments from has
no effect: the server is usually a systemd unit started at install time, with
its own environment. The symptom is a run that proceeds at single-stream speed
no matter what --concurrency you pass.

This fires the same prompts the harness uses, at several concurrency levels, and
reports throughput. If throughput does not rise with concurrency, the server is
serialising and the experiments will take many times longer than they should.

    python scripts/bench_backend.py                       # llama3.1:8b
    python scripts/bench_backend.py --model llama3.1:70b --levels 1 2 4
"""
import argparse
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                "medagentnet"))

from llm.provider import (  # noqa: E402
    OllamaProvider, OpenAICompatibleProvider)

SYSTEM = ("You are a specialized medical AI agent for the Cardiology department. "
          "Answer only from the data given. Respond ONLY with valid JSON.")
USER = """QUERY TYPE: MED_CONFLICT
FROM: dental
DISCLOSURE TIER: 2
CLINICAL CONTEXT: {
  "planned_procedure": "tooth_extraction",
  "relevant_categories": ["all"],
  "query_reason": "Extraction planned at this visit. Pre-treatment safety check."
}

PATIENT DATA (from Cardiology records only):
Medications on file (2 active, 0 inactive):
  - [ACTIVE] Warfarin (anticoagulant): 5mg daily, prescribed 2025-11-02
  - [ACTIVE] Metoprolol (beta_blocker): 50mg daily, prescribed 2025-08-14
Conditions on file (1 active, 0 resolved/inactive):
  - [ACTIVE] Atrial Fibrillation (I48): severity=serious, since 2024-06-01
Laboratory results (2):
  - 2026-01-11: INR = 2.6 (reference 2.0-3.0)
  - 2026-04-02: INR = 2.9 (reference 2.0-3.0)

Answer the clinical query using only the data above. Report only findings
relevant to the query. Return structured JSON."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.1:8b")
    ap.add_argument("--url", default=None,
                    help="default: 11434 for ollama, 8000/v1 for vllm")
    ap.add_argument("--provider", default="ollama",
                    choices=["ollama", "openai_compatible"])
    ap.add_argument("--num-ctx", type=int, default=4096)
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 4, 8, 16])
    ap.add_argument("--calls-per-level", type=int, default=None)
    args = ap.parse_args()

    if args.provider == "ollama":
        url = args.url or "http://127.0.0.1:11434"
        p = OllamaProvider(base_url=url, model=args.model, temperature=0.3,
                           max_tokens=1024, num_ctx=args.num_ctx,
                           request_timeout=600)
    else:
        url = args.url or "http://127.0.0.1:8000/v1"
        p = OpenAICompatibleProvider(base_url=url, model=args.model,
                                     temperature=0.3, max_tokens=1024,
                                     request_timeout=600)
    if not p.is_available():
        print(f"ERROR: {args.model} is not available at {url}")
        return 2

    print(f"model      : {args.model}")
    print(f"num_ctx    : {args.num_ctx}")
    print(f"server env : OLLAMA_NUM_PARALLEL is set in the SERVER's environment,")
    print(f"             not this shell. See the note printed at the end.\n")
    print(f"{'concurrency':>12}{'calls':>8}{'wall s':>9}{'calls/s':>10}"
          f"{'speed-up':>10}{'mean s':>9}{'out tok':>9}{'tok/s':>9}")

    base = None
    for c in args.levels:
        n = args.calls_per_level or max(8, c * 3)
        p.reset_stats()
        lat = []

        def one(_):
            t = time.time()
            try:
                p.generate(SYSTEM, USER)
            except Exception as e:
                print(f"    call failed: {type(e).__name__}: {e}")
            return time.time() - t

        # warm the model so the first level is not charged for loading
        if base is None:
            p.generate(SYSTEM, USER)
            p.reset_stats()

        t0 = time.time()
        with ThreadPoolExecutor(max_workers=c) as pool:
            lat = list(pool.map(one, range(n)))
        wall = time.time() - t0
        rate = n / wall
        base = base or rate
        st = p.stats
        out_tok = st.get("approx_completion_tokens", 0) / max(1, st.get("calls", 1))
        print(f"{c:>12}{n:>8}{wall:>9.1f}{rate:>10.2f}{rate / base:>9.2f}x"
              f"{statistics.mean(lat):>9.2f}{out_tok:>9.0f}{out_tok * rate:>9.0f}")

    print()
    print("Interpretation")
    print("--------------")
    print("'out tok' is the mean completion length. If it sits near the")
    print("num_predict cap the model is rambling and capping it lower will")
    print("speed everything up proportionally; if it is well under, decode")
    print("length is not the bottleneck and there is nothing to gain there.")
    print()
    print("A llama.cpp-based server (Ollama) typically plateaus at 2-4x however")
    print("many slots it is given -- confirm the slot count in the server log")
    print("before concluding it is misconfigured:")
    print("  journalctl -u ollama --no-pager | grep -o 'Parallel:[0-9]*' | tail -1")
    print("If it already says the number you asked for, and 'ollama ps' shows")
    print("100% GPU, then this is the backend's ceiling rather than a setting.")
    print()
    print("Speed-up should rise roughly with concurrency until the GPU saturates.")
    print("If it stays near 1.00x, the server is serialising requests. Fix it by")
    print("setting the variable in the SERVER's environment and restarting it:")
    print()
    print("  sudo mkdir -p /etc/systemd/system/ollama.service.d")
    print("  sudo tee /etc/systemd/system/ollama.service.d/override.conf >/dev/null <<'EOF'")
    print("  [Service]")
    print('  Environment="OLLAMA_NUM_PARALLEL=16"')
    print('  Environment="OLLAMA_MAX_LOADED_MODELS=1"')
    print('  Environment="OLLAMA_KEEP_ALIVE=-1"')
    print('  Environment="OLLAMA_FLASH_ATTENTION=1"')
    print("  EOF")
    print("  sudo systemctl daemon-reload && sudo systemctl restart ollama")
    print()
    print("KV cache scales with num_ctx x parallelism. For a 70B model at 8192")
    print("context, use 4 rather than 16 or the card will run out of memory.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
