# MedAgentNet — How to Generate Results for the Paper

This guide walks you through generating all experimental results for the academic paper.

---

## Quick Start (Mock LLM — no setup needed)

The Mock provider uses rule-based logic to simulate LLM responses. It runs instantly and produces deterministic results, making it ideal for verifying the pipeline and generating baseline numbers.

```bash
cd medagentnet

# Run ALL 4 experiments (takes ~30s for 100 patients)
python run_experiments.py

# Run with paper-quality settings (larger patient count + more seeds)
python run_experiments.py --base-patients 200 --seeds 10 --patients 50 100 250 500 1000
```

Results are saved automatically to `data/experiment_results/run_YYYYMMDD_HHMMSS/`.

---

## Full Pipeline (Real LLM)

### Step 1: Install and Start Ollama

```bash
# Install Ollama (macOS/Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Pull a medical-capable model
ollama pull llama3.1:8b          # Good balance of quality and speed
# OR for better medical reasoning:
ollama pull meditron:7b           # Medical-specific model
ollama pull medllama2:7b          # Another medical option

# Verify it's running
ollama list
```

### Step 2: Configure `config/settings.yaml`

Edit the LLM section to point to your Ollama instance:

```yaml
llm:
  provider: ollama               # Change from 'mock' to 'ollama'
  ollama:
    base_url: http://localhost:11434
    model: llama3.1:8b           # Must match the model you pulled
    max_tokens: 1024
    temperature: 0.3
```

### Step 3: Run Experiments

```bash
cd medagentnet

# Option A: Run all experiments with the configured LLM
python run_experiments.py --base-patients 100 --seeds 5

# Option B: Override provider from CLI (no config change needed)
python run_experiments.py --provider ollama --model llama3.1:8b

# Option C: Run Mock vs Real LLM comparison
python run_experiments.py --compare-providers --comparison-runs 3

# Option D: Run one experiment at a time (useful for debugging)
python run_experiments.py --provider ollama --tier-only --base-patients 50
python run_experiments.py --provider ollama --consent-only --base-patients 50
python run_experiments.py --provider ollama --variance-only --seeds 5 --base-patients 50
python run_experiments.py --provider ollama --scalability-only --patients 50 100 250
```

### Step 4: Recommended Paper-Quality Run

For the final results to include in the paper:

```bash
# Full experiments with real LLM (may take 30-60 minutes with Ollama)
python run_experiments.py \
    --provider ollama \
    --model llama3.1:8b \
    --base-patients 200 \
    --seeds 10 \
    --patients 50 100 250 500 1000 \
    --verbose

# Provider comparison (Mock vs real LLM)
python run_experiments.py \
    --compare-providers \
    --comparison-runs 5 \
    --base-patients 200 \
    --verbose
```

---

## Alternative LLM Providers

### OpenAI-Compatible API (vLLM, LM Studio, text-generation-inference)

```yaml
# In config/settings.yaml:
llm:
  provider: openai_compatible
  openai_compatible:
    base_url: http://localhost:8000/v1
    model: BioMistral/BioMistral-7B
    api_key: not-needed            # or your API key
```

```bash
python run_experiments.py --provider openai_compatible
```

### HuggingFace (local transformers)

```yaml
# In config/settings.yaml:
llm:
  provider: huggingface
  huggingface:
    model_id: BioMistral/BioMistral-7B
    device: auto
    quantization: 4bit
```

```bash
pip install transformers torch bitsandbytes accelerate
python run_experiments.py --provider huggingface
```

---

## Output Structure

Every run creates a timestamped directory:

```
data/experiment_results/
├── run_20260224_143000/
│   ├── results.json            # Full JSON with all metrics
│   ├── tables.tex              # LaTeX tables — paste into paper
│   ├── report.txt              # Human-readable summary
│   ├── config_snapshot.yaml    # Exact config used
│   └── README.txt              # Run metadata
├── latest_results.json         # Symlink to most recent JSON
└── latest_tables.tex           # Symlink to most recent tables
```

### Using the LaTeX Tables in Your Paper

The `tables.tex` file contains ready-to-paste LaTeX table blocks. To use them:

1. Open `tables.tex` from your latest run
2. Copy the tables you need into `Paper/main.tex`
3. Each table uses `booktabs` and `tabularx` (already in the paper's preamble)
4. Tables are labeled `\label{tab:tier_comparison}`, `\label{tab:per_scenario}`, etc.

---

## CLI Reference

```
python run_experiments.py [OPTIONS]

Experiment Selection:
  --tier-only              Only tier comparison
  --consent-only           Only consent restriction
  --variance-only          Only multi-seed variance
  --scalability-only       Only scalability analysis
  --compare-providers      Mock vs real LLM comparison

LLM Provider:
  --provider PROVIDER      mock | ollama | openai_compatible | huggingface | config
  --model MODEL            Override model name

Parameters:
  --base-patients N        Patient count for most experiments (default: 100)
  --seeds N                Random seeds for variance experiment (default: 5)
  --patients N [N ...]     Patient counts for scalability (default: 50 100 250 500)
  --comparison-runs N      Runs per provider for comparison (default: 3)
  --config-dir PATH        Config directory (default: config)
  --verbose                Debug logging
```

---

## Troubleshooting

**Ollama not connecting:**
```bash
ollama serve                      # Start the server
curl http://localhost:11434/api/tags   # Check it's running
```

**Model not found:**
```bash
ollama list                       # See installed models
ollama pull llama3.1:8b          # Pull the model
```

**Out of memory:**
- Use a smaller model (e.g., `llama3.2:3b`)
- Reduce `--base-patients` to 50
- Run experiments one at a time with `--tier-only`, etc.

**Slow performance:**
- Real LLM experiments take 10-60x longer than Mock
- Use `--verbose` to see progress
- Start with small runs (`--base-patients 20`) to validate before full runs
