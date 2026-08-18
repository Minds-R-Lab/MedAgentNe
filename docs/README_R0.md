# MedAgentNet

**Privacy-Preserving Federated Multi-Agent Healthcare AI Simulation**

A fully functional simulation of a federated multi-agent system where each medical department operates an autonomous AI agent. Agents communicate through structured, privacy-preserving clinical queries — sharing only the minimum information necessary for cross-departmental decision-making.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Option A: Launch the Clinical Demo GUI (recommended)
streamlit run app.py

# Option B: Run the CLI simulation
python main.py
```

## 🖥️ Clinical Demo GUI

The Streamlit GUI provides a full interactive experience:

```bash
streamlit run app.py
```

**Features:**
- **🩺 Clinical Query** — Select a patient, pick your department role, run a cross-departmental query, and watch agents communicate in real-time
- **👥 Patient Browser** — Browse all patients, see their medications, conditions, lab results, and planted scenarios
- **📊 Batch Simulation** — Run all conflict and pattern scenarios at once and see detection rates
- **🔒 Audit & Privacy** — View every inter-agent message, consent check, and disclosure tier used
- **ℹ️ How It Works** — Explains the architecture for non-technical audiences

**LLM Selection:** Choose your AI engine directly in the sidebar:
- Mock (rule-based, instant, no GPU)
- Ollama (local open-source models)
- OpenAI-compatible API (vLLM, LMStudio, etc.)

## CLI Usage

```bash
# Install dependencies
pip install pyyaml

# Run the full simulation (100 patients, 10 departments, mock LLM)
cd medagentnet
python main.py

# Run with more patients
python main.py --patients 500

# Run with a real open-source LLM via Ollama
ollama pull llama3:8b-instruct       # or: biomistral, meditron
python main.py --provider ollama

# Run with verbose output
python main.py --verbose
```

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Dental     │     │  Cardiology  │     │  Nephrology  │
│   Agent      │     │    Agent     │     │    Agent     │
│  ┌────────┐  │     │  ┌────────┐  │     │  ┌────────┐  │
│  │Dental  │  │     │  │Cardio  │  │     │  │Nephro  │  │
│  │EHR     │  │     │  │EHR     │  │     │  │EHR     │  │
│  └────────┘  │     │  └────────┘  │     │  └────────┘  │
└──────┬───────┘     └──────┬───────┘     └──────┬───────┘
       │                    │                    │
       └────────────┬───────┴───────┬────────────┘
                    │               │
              ┌─────┴─────┐  ┌──────┴──────┐
              │Orchestrator│  │   Consent   │
              │  Agent     │  │   Manager   │
              └────────────┘  └─────────────┘
```

## 10 Departments

| # | Department | Focus |
|---|-----------|-------|
| 1 | Cardiology | Heart, anticoagulants, arrhythmia |
| 2 | Dentistry | Oral health, extractions, pain management |
| 3 | General Practice | Primary care, referrals, chronic conditions |
| 4 | Endocrinology | Diabetes, thyroid, metabolic disorders |
| 5 | Ophthalmology | Eye, retinal disease, glaucoma |
| 6 | Nephrology | Kidney, renal function, dialysis |
| 7 | Neurology | Brain, seizures, neuropathy |
| 8 | Rheumatology | Joints, autoimmune, immunosuppression |
| 9 | Pulmonology | Lungs, asthma, COPD |
| 10 | Laboratory | Lab tests, blood work, trending |

## Adding a New Department

Edit `config/departments.yaml` and add a new entry:

```yaml
  dermatology:
    name: "Dermatology"
    description: "Skin conditions and treatments"
    specialization_keywords: ["skin", "rash", "dermatitis"]
    relevant_for_procedures: ["biopsy"]
    common_medications:
      - {name: "Isotretinoin", category: "retinoid",
         interactions: ["tetracycline", "vitamin_a"],
         risk_flags: ["teratogenic_risk", "liver_risk"]}
    common_conditions:
      - {code: "L20", name: "Atopic Dermatitis", severity: "chronic"}
```

That's it — the system auto-discovers new departments on next run.

## LLM Providers

| Provider | Setup | Best For |
|----------|-------|----------|
| **mock** (default) | No setup needed | Testing, development, CI |
| **ollama** | `ollama pull llama3:8b-instruct` | Local GPU inference |
| **huggingface** | `pip install transformers bitsandbytes` | Custom models |
| **openai_compatible** | Any OpenAI-compatible API (vLLM, LMStudio) | Production |

### Recommended Open-Source Medical LLMs

- **BioMistral-7B** — Fine-tuned on PubMed/medical corpus
- **Meditron-7B** — Medical domain LLM from EPFL
- **Llama-3-8B-Instruct** — Strong general reasoning
- **OpenBioLLM-8B** — Biomedical/clinical focused

Configure in `config/settings.yaml`:
```yaml
llm:
  provider: "ollama"
  ollama:
    model: "biomistral:latest"
```

## Planted Clinical Scenarios

The simulation plants 8 known medication conflicts and 3 cross-departmental patterns:

### Medication Conflicts
1. **Warfarin + Dental Extraction** — Bleeding risk
2. **Triple Whammy (ACE+NSAID+Diuretic)** — Acute kidney injury
3. **Metformin + Renal Failure** — Lactic acidosis
4. **Beta-Blocker + Asthma/COPD** — Bronchospasm
5. **Warfarin + NSAID** — GI bleeding
6. **Timolol + Metoprolol** — Additive bradycardia
7. **Methotrexate + NSAID** — Methotrexate toxicity
8. **Carbamazepine + Warfarin** — Subtherapeutic anticoagulation

### Diagnostic Patterns
1. **Undiagnosed Diabetes** — Rising glucose + retinal changes + neuropathy
2. **CKD Progression** — Declining eGFR + hypertension + rising creatinine
3. **Thyroid-Cardiac Connection** — New A-fib + weight loss + suppressed TSH

## Output Files

```
data/
├── patients/          # 100+ synthetic patient JSON records
│   ├── PAT-xxxx.json
│   └── _index.json    # Summary index
├── results/
│   ├── evaluation_*.json      # Quantitative metrics
│   ├── detailed_results_*.json # Full scenario results
│   └── report_*.txt           # Human-readable report
└── audit_trail.jsonl          # Complete privacy audit log
```

## Privacy Model

- **Data Locality**: Raw patient data never leaves its department agent
- **Tiered Disclosure**: 3 levels (Flag Only → Summary → Full Context)
- **Consent-Governed**: Every query validated against patient consent
- **Full Audit Trail**: Every inter-agent message logged and auditable
- **Minimum Necessary**: Agents share only what the query requires

## Project Structure

```
medagentnet/
├── main.py              # CLI entry point
├── config/
│   ├── departments.yaml # Department definitions (add new ones here!)
│   └── settings.yaml    # Global configuration
├── data/
│   └── generator.py     # Synthetic patient data generator
├── agents/
│   └── core.py          # DepartmentAgent + OrchestratorAgent
├── protocol/
│   ├── models.py        # Data models (Patient, Query, Response, Alert)
│   └── consent.py       # Consent management + Audit trail
├── llm/
│   ├── provider.py      # LLM abstraction (Mock/Ollama/HF/OpenAI)
│   └── prompts.py       # Agent system prompts
└── simulation/
    └── runner.py        # Simulation orchestration + evaluation
```
