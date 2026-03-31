# Fidelity Agentic Stack

Layer-wise fidelity evaluation of an MCP → A2A → A2UI pipeline, measuring BERTScore at each intermediate against ground-truth httpx documentation answers.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # set OPENAI_API_KEY

python scripts/fetch_docs.py     # download httpx docs
python mcp_server/index.py       # build FAISS index
PYTHONPATH=. python eval/run_eval.py  # run full evaluation
```

Results are written to `eval/results/` and `paper_results/table1.md`.

## Manual Inspection

```bash
PYTHONPATH=. streamlit run ui/app.py
```
