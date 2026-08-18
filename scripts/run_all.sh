#!/usr/bin/env bash
# Reproduce every figure in the paper.
#
#   ./scripts/run_all.sh mock              harness self-test, no model server (~2 min)
#   ./scripts/run_all.sh ollama            the reported run (~6-9 h on one consumer GPU)
#   ./scripts/run_all.sh backends          the backend matrix only
#
set -euo pipefail
MODE="${1:-mock}"
cd "$(dirname "$0")/../medagentnet"

case "$MODE" in
  mock)
    python run_r1.py --provider mock --patients 200 --tag validation
    ;;
  ollama)
    command -v ollama >/dev/null || { echo "ollama not on PATH"; exit 1; }
    ollama pull llama3.1:8b
    python run_r1.py --provider ollama --model llama3.1:8b --patients 200 \
           --tag reported --experiments e9 e1 e2 e3 e4 e5 e7 e8
    ;;
  backends)
    for m in llama3.1:8b qwen2.5:7b mistral:7b phi3:mini; do ollama pull "$m"; done
    python run_r1.py --provider ollama --model llama3.1:8b \
           --backend-matrix llama3.1:8b qwen2.5:7b mistral:7b phi3:mini \
           --include-mock-in-matrix --experiments e6 --tag backends
    ;;
  *)
    echo "usage: $0 {mock|ollama|backends}"; exit 2
    ;;
esac

echo
echo "results : medagentnet/data/results_r1/latest.json"
echo "tables  : medagentnet/data/results_r1/latest_tables.tex"
echo
echo "Paste tables.tex over the generated block in the manuscript."
