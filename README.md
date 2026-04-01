# Fidelity Agentic Stack

Experimental evaluation harness for the paper *"Fidelity of the Agentic Stack: Measuring Information Preservation Across MCP, A2A, and A2UI Layers."*

This repository implements an end-to-end MCP → A2A → A2UI pipeline and measures how faithfully information is preserved as it flows through each layer — from raw document retrieval, through LLM synthesis, to structured UI-ready output. Results fill Tables 1a and 1b in the paper.

## Architecture

```
Query → MCP retrieve → R₀ (raw chunks) → LLM synthesise → R₁ (NL answer) → LLM format → R₂ (JSON payload)
```

The pipeline captures three intermediates and computes BERTScore F1/Precision/Recall at each layer against a human-written reference answer. Two deltas quantify fidelity change across layer transitions:

- **R₀**: Raw document chunks returned by the MCP retrieval tool (FAISS over httpx docs)
- **R₁**: Natural-language answer synthesised by `gpt-4o-mini` from R₀
- **R₂**: Structured JSON payload (`{summary, key_points, code_example, source_ref}`) formatted from R₁
- **Δ₁** = F1(R₁) − F1(R₀) — measures fidelity change from A2A synthesis
- **Δ₂** = F1(R₂) − F1(R₁) — measures fidelity change from A2UI formatting

## Prerequisites

- Python 3.10+
- An OpenAI API key (used for embeddings and `gpt-4o-mini` inference)

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # then set OPENAI_API_KEY
```

## Reproducing the Evaluation

### Step 1: Prepare the corpus and index

```bash
python scripts/fetch_docs.py      # downloads httpx docs → data/httpx_docs/
python mcp_server/index.py        # builds FAISS index → data/faiss_index/
```

Both scripts are idempotent and safe to re-run.

### Step 2: Run the evaluation

```bash
PYTHONPATH=. python eval/run_eval.py
```

This will:

1. Start the MCP server as a subprocess on `localhost:8765`
2. Run each query in `data/test_set.json` through the LangGraph pipeline
3. Capture R₀, R₁, R₂ for every query
4. Compute BERTScore (roberta-large, CPU) per layer in batched calls
5. Write results and stop the MCP server

### Step 3: Inspect outputs

| File | Contents |
|------|----------|
| `eval/results/raw_logs.json` | Full R₀/R₁/R₂ text and JSON for every query |
| `eval/results/scores.csv` | Per-query BERTScore P/R/F1 per layer, Δ₁, Δ₂, Δ₁ᴿ, Δ₂ᴿ, plus a Mean row |
| `paper_results/table1.md` | Table 1a (F1 per layer) and Table 1b (P/R decomposition) for the paper |

A summary is also printed to stdout at the end of the run.

## Using a Custom Test Set

The evaluation pipeline reads queries from `data/test_set.json`. To use your own test set:

### Test set format

Create a JSON file with an array of objects, each containing:

```json
[
  {
    "id": "Q01",
    "query": "Your question here",
    "reference": "Human-written ground-truth answer"
  }
]
```

- **`id`**: Unique identifier (appears in output CSVs and tables)
- **`query`**: The question sent to the retrieval + synthesis pipeline
- **`reference`**: The ground-truth answer that BERTScore is computed against

### Replacing the test set

1. Replace `data/test_set.json` with your file (same format as above)
2. Re-run the evaluation: `PYTHONPATH=. python eval/run_eval.py`

The pipeline is count-agnostic — it works with any number of queries. No code changes are needed.

### Writing good reference answers

- Source references from primary documentation, not LLM output
- Include specific API names, parameter values, and defaults
- Keep answers factual and concise (2-4 sentences)
- Ensure the query requires retrieval — if an LLM can answer correctly from parametric knowledge alone, the MCP layer adds no measurable value

### Using a different corpus

To evaluate against a different documentation set:

1. Place your markdown files in `data/httpx_docs/` (or change the path in `mcp_server/index.py`)
2. Rebuild the FAISS index: `python mcp_server/index.py`
3. Write a matching `data/test_set.json` with queries and references grounded in your corpus
4. Run the evaluation as usual

## Controlled Variables

These are held constant across all queries. Changing any of them invalidates comparability with published results.

| Variable | Value |
|----------|-------|
| LLM | `gpt-4o-mini`, temperature=0 |
| Retrieved chunks (k) | 3 |
| Chunk separator (R₀) | `"\n\n---\n\n"` |
| BERTScore backbone | `roberta-large` |
| bert-score version | `0.3.13` (pinned in requirements.txt) |
| BERTScore device | `cpu` |
| R₂ scoring fields | `summary + " " + " ".join(key_points)` — `code_example` excluded |

## Manual Inspection UI

```bash
PYTHONPATH=. streamlit run ui/app.py
```

This launches a Streamlit app for browsing individual query results interactively. The UI is not part of the evaluation loop.

## Project Structure

```
fidelity-agentic-stack/
├── data/
│   ├── test_set.json              # 30 queries + reference answers
│   ├── httpx_docs/                # Raw httpx markdown docs
│   └── faiss_index/               # FAISS index (built by mcp_server/index.py)
├── mcp_server/
│   ├── server.py                  # FastMCP server (streamable-http, port 8765)
│   └── index.py                   # FAISS index builder
├── agent/
│   ├── graph.py                   # LangGraph StateGraph (orchestrator → formatter)
│   ├── orchestrator.py            # MCP retrieval (R₀) + LLM synthesis (R₁)
│   └── formatter.py               # JSON formatting (R₂)
├── eval/
│   ├── run_eval.py                # Evaluation runner
│   ├── score.py                   # BERTScore computation + delta calculation
│   └── results/                   # Generated at runtime
├── paper_results/
│   └── table1.md                  # Paper-ready tables
├── ui/
│   └── app.py                     # Streamlit inspection UI
└── scripts/
    └── fetch_docs.py              # httpx docs downloader
```
