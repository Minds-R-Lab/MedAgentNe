#!/usr/bin/env bash
# One-shot setup for a single-GPU Linux box (tested against an H100 on a
# Google Cloud Vertex AI Workbench instance).
#
#   ./scripts/setup_gpu.sh                      # 8B models only  (~16 GB disk)
#   ./scripts/setup_gpu.sh --with-70b           # adds llama3.1:70b (~56 GB disk)
#
# Installs Ollama, starts it with settings tuned for a large-VRAM card, pulls
# the models, and verifies the GPU is actually being used.
set -euo pipefail

WITH_70B=0
[[ "${1:-}" == "--with-70b" ]] && WITH_70B=1

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

say "GPU"
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
  echo "nvidia-smi not found. Ollama will fall back to CPU and the full suite"
  echo "will take days rather than hours. Stopping."
  exit 1
fi

say "Disk"
NEED=$(( WITH_70B == 1 ? 70 : 25 ))
AVAIL=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc '0-9')
echo "available in \$HOME: ${AVAIL}G, needed: ~${NEED}G"
if (( AVAIL < NEED )); then
  echo "Not enough space. Either resize the boot disk, or point Ollama at a"
  echo "larger volume with:  export OLLAMA_MODELS=/mnt/disks/<something>/ollama"
  exit 1
fi

say "Install Ollama"
if command -v ollama >/dev/null; then
  echo "already installed: $(ollama --version 2>&1 | head -1)"
elif sudo -n true 2>/dev/null; then
  curl -fsSL https://ollama.com/install.sh | sh
else
  # No sudo: unpack into ~/.local and use that.
  echo "no passwordless sudo; installing to ~/.local"
  mkdir -p "$HOME/.local"
  curl -fsSL https://ollama.com/download/ollama-linux-amd64.tgz \
    | tar -xz -C "$HOME/.local"
  export PATH="$HOME/.local/bin:$PATH"
  grep -q 'HOME/.local/bin' "$HOME/.bashrc" 2>/dev/null || \
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
fi

say "Start the server"
# Tuning notes:
#   NUM_PARALLEL       requests served concurrently; match --concurrency
#   MAX_LOADED_MODELS  1, so a model is not evicted mid-experiment
#   KEEP_ALIVE         -1, never unload between experiments
#   FLASH_ATTENTION    faster attention kernels on Ampere and later
export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_NUM_PARALLEL=8
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_KEEP_ALIVE=-1
export OLLAMA_FLASH_ATTENTION=1

if curl -sf "http://$OLLAMA_HOST/api/tags" >/dev/null 2>&1; then
  echo "a server is already listening on $OLLAMA_HOST"
  echo "if it was not started with the settings above, restart it:"
  echo "  pkill ollama   # or: sudo systemctl stop ollama"
else
  nohup ollama serve > "$HOME/ollama.log" 2>&1 &
  for _ in $(seq 1 40); do
    curl -sf "http://$OLLAMA_HOST/api/tags" >/dev/null 2>&1 && break
    sleep 1
  done
  curl -sf "http://$OLLAMA_HOST/api/tags" >/dev/null \
    || { echo "server did not come up; see $HOME/ollama.log"; exit 1; }
  echo "started, logging to $HOME/ollama.log"
fi

say "Pull models"
MODELS=(llama3.1:8b qwen2.5:7b mistral:7b phi3:mini)
(( WITH_70B == 1 )) && MODELS+=(llama3.1:70b)
for m in "${MODELS[@]}"; do
  echo "--- $m"
  ollama pull "$m"
done

say "Confirm the GPU is in use"
ollama run llama3.1:8b "reply with the single word: ok" >/dev/null 2>&1 || true
sleep 2
ollama ps || true
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
echo
echo "If 'ollama ps' shows 100% GPU, you are set. If it shows CPU, check"
echo "$HOME/ollama.log for a CUDA error."

say "Ready"
cat <<'NEXT'
Put these in your shell before running experiments (the script exported them
for this shell only):

  export OLLAMA_HOST=127.0.0.1:11434
  export OLLAMA_NUM_PARALLEL=8
  export OLLAMA_MAX_LOADED_MODELS=1
  export OLLAMA_KEEP_ALIVE=-1
  export OLLAMA_FLASH_ATTENTION=1

Then see docs/GPU_RUN.md.
NEXT
