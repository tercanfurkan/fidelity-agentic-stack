# Fidelity Agentic Stack

Experimental evaluation harness for the paper *"Fidelity of the Agentic Stack: Measuring Information Preservation Across MCP, A2A, and A2UI Layers."*

Implements an end-to-end MCP → A2A → A2UI pipeline and measures how faithfully information is preserved at each layer using BERTScore. Results fill Tables 1a/1b in the paper.

```
Query → MCP retrieve → R₀ (raw chunks) → LLM synthesise → R₁ (NL answer) → LLM format → R₂ (JSON payload)
```

| Layer | What | Delta |
|-------|------|-------|
| R₀ | Raw FAISS chunks from MCP retrieval | — |
| R₁ | NL answer synthesised by gpt-4o-mini | Δ₁ = F1(R₁) − F1(R₀) |
| R₂ | JSON payload `{summary, key_points, ...}` | Δ₂ = F1(R₂) − F1(R₁) |

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                  # set OPENAI_API_KEY

python scripts/fetch_docs.py          # download httpx docs → data/httpx_docs/
python mcp_server/index.py            # build FAISS index → data/faiss_index/
PYTHONPATH=. python eval/run_eval.py  # run full evaluation (starts/stops MCP server)
```

Outputs:

| File | Contents |
|------|----------|
| `eval/results/raw_logs.json` | Full R₀/R₁/R₂ text per query |
| `eval/results/scores.csv` | Per-query P/R/F1 per layer + deltas |
| `paper_results/table1.md` | Paper-ready Tables 1a and 1b |

## Custom Test Sets

Replace `data/test_set.json` with any array of `{id, query, reference}` objects and re-run. The pipeline is count-agnostic.

```json
[{"id": "Q01", "query": "Your question", "reference": "Ground-truth answer"}]
```

To swap the corpus: place markdown files in `data/httpx_docs/`, rebuild the index (`python mcp_server/index.py`), and write matching references.

## Controlled Variables

Changing any of these invalidates comparability with published results.

| Variable | Value |
|----------|-------|
| LLM | `gpt-4o-mini`, temperature=0 |
| Retrieved chunks (k) | 3 |
| BERTScore | `roberta-large`, `bert-score==0.3.13`, CPU |
| R₂ scoring | `summary + key_points` only (`code_example` excluded) |

## Streamlit UI

```bash
PYTHONPATH=. streamlit run ui/app.py  # manual inspection, not part of eval
```
