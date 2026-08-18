# Running the full suite on a single large-VRAM GPU

Written against an NVIDIA H100 80GB on a Google Cloud Vertex AI Workbench
instance. Any single card with 24GB or more works for the 8B models; the 70B
arm needs roughly 48GB.

## Setup

```bash
git clone https://github.com/Minds-R-Lab/MedAgentNe.git
cd MedAgentNe
pip install -r requirements.txt
./scripts/setup_gpu.sh --with-70b        # omit the flag to skip the 70B model
```

### The one setting that is easy to get wrong

`OLLAMA_NUM_PARALLEL` must be set **in the environment of the Ollama server
process**, not in the shell you launch experiments from. If Ollama was installed
with its own install script it runs as a systemd unit with its own environment,
and exports in your shell have no effect on it. The symptom is a run that
proceeds at single-stream speed no matter what `--concurrency` you pass —
roughly 0.1 scenarios per second on an H100 instead of 1–2.

If Ollama is a systemd service:

```bash
sudo mkdir -p /etc/systemd/system/ollama.service.d
sudo tee /etc/systemd/system/ollama.service.d/override.conf >/dev/null <<'EOF'
[Service]
Environment="OLLAMA_NUM_PARALLEL=16"
Environment="OLLAMA_MAX_LOADED_MODELS=1"
Environment="OLLAMA_KEEP_ALIVE=-1"
Environment="OLLAMA_FLASH_ATTENTION=1"
EOF
sudo systemctl daemon-reload && sudo systemctl restart ollama
systemctl show ollama -p Environment      # confirm it took
```

If you are running `ollama serve` yourself, export the variables *before*
starting it, in that same shell.

**Verify before committing hours to a run:**

```bash
python scripts/bench_backend.py --model llama3.1:8b
```

It reports throughput at concurrency 1, 4, 8 and 16. Speed-up should rise with
concurrency. If it stays at 1.00x the server is serialising and the fix above
has not taken effect.

KV cache scales with `num_ctx` × parallelism, and Ollama silently reduces the
number of slots it opens if the product does not fit comfortably. That is the
usual reason a run scales to only 2-3x instead of 10x or more. The measured
worst-case prompt in this system is **720 tokens**, so the default is 4096
rather than 8192: a larger window reserves cache for text that never arrives and
costs concurrency. A 70B model needs roughly 1.4 GB per slot at 4096 on top of
42 GB of weights — use 4 to 8 slots there, not 16.

### If Ollama still will not scale

Ollama's scheduler is built for interactive use rather than batch throughput.
If `bench_backend.py` plateaus around 2-3x, vLLM will do considerably better on
the same card, and the harness speaks to it through the existing
OpenAI-compatible backend:

```bash
pip install vllm
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --served-model-name llama3.1:8b \
  --max-model-len 4096 --gpu-memory-utilization 0.90 &

python scripts/bench_backend.py --provider openai_compatible --model llama3.1:8b
```

Then run the experiments against it:

```bash
python run_r1.py --provider openai_compatible --model llama3.1:8b \
                 --openai-url http://127.0.0.1:8000/v1 \
                 --patients 200 --concurrency 32 --tag reported \
                 --experiments e9 e1 e2 e3 e4 e5 e7 e8
```

`--served-model-name` matters: the backend verifies that the endpoint actually
serves the model named in the results file, and refuses to run otherwise, so a
results file cannot be mislabelled. vLLM needs the HuggingFace weights rather
than the Ollama copy, and gated repositories require `huggingface-cli login`.

## Two settings that matter

**`--concurrency`.** Scenarios are independent of one another, so the harness
can evaluate several at once. It defaults to 1, which leaves a large card almost
idle. Set it to match `OLLAMA_NUM_PARALLEL`; on an H100 running an 8B model,
16–24 is comfortable.

One caveat: per-scenario latency is wall-clock, so at concurrency above 1 it
includes queueing at the model server and is *not* per-request latency. The run
metadata records the concurrency used and the generated tables carry a warning
header. Take the latency figures reported in the paper from E7, whose patient
and department sweeps are pinned to concurrency 1 for exactly this reason.

Concurrency is a throughput knob only. Results are reassembled in scenario
order before scoring, so a run at any concurrency produces byte-identical
metrics to a sequential one — with the exception of E7, which varies
concurrency deliberately and pins its other sweeps to 1 so they stay
comparable.

**`--num-ctx`.** Ollama's own default context is 2048 tokens. The largest
prompt this system produces is about 720 tokens of record plus up to 1024
tokens of completion, which fits — but with little margin, and Ollama truncates
from the *start* of the prompt silently when it does not fit, which would
remove the system prompt and the clinical context while still returning a
plausible-looking answer. The harness therefore sets 8192 explicitly and counts
any prompt that comes close to the limit. Check `provider.truncation_suspected`
in the results file after a run; it should be 0.

## The reported run

```bash
python run_r1.py \
  --provider ollama --model llama3.1:8b \
  --patients 200 --concurrency 8 --num-ctx 8192 \
  --tag reported \
  --experiments e9 e1 e2 e3 e4 e5 e7 e8
```

### Cost

One run at 200 patients is 232 scenarios and about 1,400 backend calls. The full
suite performs 158 such runs — **roughly 222,000 calls** — because several
experiments sweep: E2 is six restriction levels × ten sampled graphs, E4 is ten
configurations × three seeds plus the paired comparisons.

Measured against sustained aggregate throughput, on one H100:

| Concurrency | `OLLAMA_NUM_PARALLEL` | llama3.1:8b, full suite |
|---|---|---|
| 8 | 8 | ~9 h |
| 16 | 16 | ~4–5 h |
| 24 | 24 | ~3–4 h |

An 8B model occupies 5 GB of an 80 GB card, so parallelism is limited by KV
cache rather than weights and 16–24 is comfortable. Start it in the evening.

The **backend matrix is cheap by comparison** — E6 runs two seeds per model, not
the whole suite, so each arm is 2,800 calls: minutes for the 7–8B models and
under an hour for 70B.

### Keeping it alive, and watching it

A closed Jupyter terminal tab kills whatever is running in it. `nohup` detaches
the process from the tab while still letting you follow the output:

```bash
nohup python run_r1.py ... > ~/reported.log 2>&1 &
tail -f ~/reported.log
```

`Ctrl-c` stops the `tail`, not the run. Reattach to the output at any time with
the same `tail -f`. `tmux` is an alternative if you prefer it, but it is not
required.

The run prints a heading for each experiment, a line per sub-step of every
sweep, and a progress line with rate and ETA roughly eight times per run:

```
  experiment E2  (3 of 8)
  [E2 14/60] revoked 20%, graph 4/10
    58/232 scenarios  1.4/s  elapsed 0.7m  eta 2.1m
```

## The backend matrix

This is the experiment the reviewers asked for and the one a large card makes
practical: several model families, and two sizes within one family.

```bash
python run_r1.py \
  --provider ollama --model llama3.1:8b \
  --patients 200 --concurrency 8 --num-ctx 8192 \
  --backend-matrix llama3.1:8b llama3.1:70b \
                   qwen2.5:7b qwen2.5:14b qwen2.5:32b \
                   mistral:7b phi3:mini \
  --include-mock-in-matrix \
  --experiments e6 --tag backends
```

Every model must be pulled first; the harness refuses to substitute one model
for another, so a results file cannot be labelled with a model that never ran.
Allow two to three hours, most of it the 70B arm.

This list gives two size ladders rather than a flat set of models — llama3.1 at
8B and 70B, qwen2.5 at 7B, 14B and 32B — which is what separates a claim about
model *family* from a claim about model *scale*.

Reduce `OLLAMA_NUM_PARALLEL` and `--concurrency` to 4 for the 70B model if you
see the GPU running out of memory: each parallel request holds its own KV
cache.

## Checking a run afterwards

```bash
python - <<'PY'
import json
d = json.load(open("data/results_r1/latest.json"))
m = d["meta"]
print("backend :", m["provider"])
s = m.get("provider_stats", {})
print("calls   :", s.get("calls"), " mean latency:", s.get("mean_latency_s"), "s")
print("format failures:", s.get("format_failure_rate"))
print("truncation suspected:", s.get("truncation_suspected"), "(must be 0)")
print("max prompt seen:", s.get("max_prompt_tokens_seen"), "tokens, num_ctx:", s.get("num_ctx"))
c = d["e4"]["rows"]["medagentnet"]["f1"]["mean"]
a = d["e4"]["rows"]["ablate_synthesis"]["f1"]["mean"]
print(f"full F1 {c:.3f} vs assembly-ablated {a:.3f}")
PY
```

`format_failure_rate` above a few percent means the model is not honouring the
JSON contract; that is a finding to report in the backend table rather than a
problem to fix, but check it is not a truncation artefact first.

## Getting the results back

Each run writes `data/results_r1/<tag>_<timestamp>/results.json` and
`tables.tex`. The LaTeX block goes straight into the manuscript.

```bash
tar czf medagentnet-results.tar.gz data/results_r1/
# then, from your laptop:
gcloud compute scp <instance>:~/MedAgentNe/medagentnet/medagentnet-results.tar.gz . \
    --zone <zone>
```

Or open `data/results_r1/latest_tables.tex` in the Jupyter file browser and
download it directly.

## If something goes wrong

| Symptom | Cause |
|---|---|
| `ProviderUnavailable` at startup | Server not running, or the model is not pulled. `ollama list` to check. This is deliberate — the harness will not silently fall back to the rule-based backend. |
| `ollama ps` shows `100% CPU` | The CUDA runtime was not found at install time. See `~/ollama.log`; reinstalling after the driver is present usually fixes it. |
| Run dies partway | Use `tmux`. Experiments are independent, so you can also rerun just the missing ones with `--experiments e2 e7`. |
| `truncation_suspected` > 0 | Raise `--num-ctx` to 16384 and rerun. |
| Out of memory with 70B | Lower `OLLAMA_NUM_PARALLEL` and `--concurrency` to 2–4. |
