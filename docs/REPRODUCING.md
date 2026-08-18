# Reproducing the paper

## 1. Harness self-test (no model server, ~80 s)

```bash
cd medagentnet
pip install -r ../requirements.txt
python -m pytest tests -q
python run_r1.py --provider mock --patients 200 --tag validation
```

This exercises every experiment path end to end and is what CI runs. The numbers
it produces are in `results/validation_mock/` for comparison.

## 2. The reported run

```bash
ollama serve &
ollama pull llama3.1:8b
cd medagentnet
python run_r1.py --provider ollama --model llama3.1:8b --patients 200 \
                 --tag reported --experiments e9 e1 e2 e3 e4 e5 e7 e8
```

## 3. The backend matrix

```bash
for m in llama3.1:8b qwen2.5:7b mistral:7b phi3:mini; do ollama pull "$m"; done
python run_r1.py --provider ollama --model llama3.1:8b \
                 --backend-matrix llama3.1:8b qwen2.5:7b mistral:7b phi3:mini \
                 --include-mock-in-matrix --experiments e6 --tag backends
```

## 4. Regenerate the paper's tables

Each run writes `data/results_r1/<tag>_<timestamp>/tables.tex`. Replace the block
between the `R1 RESULT TABLES` banner and `END GENERATED BLOCK` in the
manuscript with its contents. Every table in the paper is produced this way, so
any figure can be traced to the run that produced it.

To regenerate tables from an existing results file without re-running:

```python
import json, sys; sys.path.insert(0, "medagentnet")
from simulation.tables import generate_all_tables
d = json.load(open("results/validation_mock/results.json"))
open("tables.tex", "w").write(generate_all_tables(d))
```

## Determinism

Everything is seeded. The same seed and the same patient count reproduce the
same cohort exactly. With a language-model backend, sampling at temperature 0.3
introduces run-to-run variation in the responses; the cohort and the queries do
not vary.

Note that `run_experiments.py` (the R0 driver, retained for comparison) rewrites
`config/settings.yaml` in place during a run and restores it at the end. A crash
mid-run leaves the config mutated, and two concurrent runs corrupt each other.
`run_r1.py` does not do this.
