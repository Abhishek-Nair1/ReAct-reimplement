# ReAct: Synergizing Reasoning and Acting in Language Models

Paper: [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) (ICLR 2023)  
Authors: Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao (Princeton & Google Research)

---

## 📁 Repository Structure

```text
ReAct/
├── wikienv.py                 # Wikipedia environment (Search, Lookup, Finish)
├── run.py                     # HotpotQA single-question runner
├── run_fever.py               # FEVER single-claim runner
├── evaluate.py                # Reproducible evaluation runner with metrics (EM, F1)
├── data/
│   ├── hotpot_dev_v1_simplified.json   # Official HotpotQA dev set (7,405 samples)
│   └── paper_dev.jsonl                 # Official FEVER dev set
├── prompts/
│   ├── prompts_naive.json     # HotpotQA few-shot exemplars (webthink_simple6)
│   └── fever.json             # FEVER few-shot exemplars (webthink_simple3)
├── requirements.txt           # Minimal dependencies (requests, beautifulsoup4)
└── README.md
```

---

## 🚀 How to Run (100% Local GPU via Ollama)

All inference runs locally through [Ollama](https://ollama.com/) using `qwen2.5:3b`. Zero external API keys needed.

### 1. Interactive Single Runs
```bash
# HotpotQA
python run.py "What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?"

# FEVER
python run_fever.py "Albert Einstein was awarded the 1921 Nobel Prize in Physics."
```

---

## 📊 Running Reproducible Experiments (`evaluate.py`)

Run evaluations on the official paper dev sets using the exact paper sampling seed (`seed=233`):

### HotpotQA Evaluation
```bash
# Evaluate 5 samples (quick test)
python evaluate.py --task hotpotqa --count 5

# Evaluate 20 samples with quiet output
python evaluate.py --task hotpotqa --count 20 --quiet --output results_hotpotqa.json

# Run full 500 samples (as in paper)
python evaluate.py --task hotpotqa --count 500 --output results_hotpotqa_500.json
```

### FEVER Fact Verification Evaluation
```bash
# Evaluate 5 samples
python evaluate.py --task fever --count 5

# Run 50 samples
python evaluate.py --task fever --count 50 --output results_fever.json
```

**Output Metrics:**
* **EM (Exact Match):** Normalized exact string match with ground-truth.
* **F1 Score:** Token-level precision & recall (for HotpotQA).
* **Live Checkpoint Saving:** Results and step-by-step trajectories are automatically saved to JSON after every sample, so progress is never lost if stopped early.

---

## 📌 Note on the Other Paper Benchmarks (ALFWorld & WebShop)

* **ALFWorld:** Requires Linux / WSL2 to build the underlying C++ `textworld`/`jericho` text engine and download the 134 evaluation game files (`alfworld-download`).
* **WebShop:** Requires hosting the local WebShop Flask + Lucene search engine with the 1.18M Amazon product dataset (the paper's public demo AWS server is permanently offline).
