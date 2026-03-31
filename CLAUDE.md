# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

This repo implements an end-to-end MCP→A2A→A2UI pipeline and layer-wise fidelity evaluation harness producing experimental results for an academic paper. The output fills **Table 1** in `/Users/furkan/mcppaper/output/paper.md`. All implementation details, data formats, and controlled variables are in `SPEC.md` — read it fully before writing any code.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then set OPENAI_API_KEY
```

## Running the Pipeline

```bash
# One-time setup (idempotent — safe to re-run)
python scripts/fetch_docs.py      # Download httpx docs → data/httpx_docs/
python mcp_server/index.py        # Build FAISS index → data/faiss_index/

# Full evaluation run (starts/stops MCP server automatically)
python eval/run_eval.py

# Streamlit UI for manual inspection only (not part of eval)
streamlit run ui/app.py
```

## Architecture

The pipeline captures three intermediates against the same ground-truth reference and computes BERTScore F1/P/R at each layer:

```
Query → MCP retrieve → R₀ (raw chunks) → LLM synthesise → R₁ (NL answer) → LLM format → R₂ (JSON payload)
```

- **`mcp_server/server.py`** — FastMCP HTTP/SSE server on `localhost:8765`; exposes a `retrieve` tool and a `/health` endpoint. Loads FAISS index at startup.
- **`mcp_server/index.py`** — Builds FAISS index from `data/httpx_docs/` using `text-embedding-3-small`.
- **`agent/graph.py`** — LangGraph `StateGraph` with `PipelineState` TypedDict; wires orchestrator → formatter → END.
- **`agent/orchestrator.py`** — Node 1: calls MCP tool via `asyncio.run()`, captures R₀, then calls `gpt-4o-mini` (temperature=0) to synthesise R₁.
- **`agent/formatter.py`** — Node 2: calls `gpt-4o-mini` with JSON response format to produce R₂ `{summary, key_points, code_example, source_ref}`.
- **`eval/run_eval.py`** — Manages MCP server subprocess, runs all 15 queries, writes `eval/results/raw_logs.json`, `eval/results/scores.csv`, and `paper_results/table1.md`.
- **`eval/score.py`** — Batches BERTScore P/R/F1 per layer; computes Δ₁ and Δ₂ on both F1 and Recall.

## Critical Constraints

These must never be changed — any deviation invalidates experimental results:

| Item | Fixed value |
|---|---|
| LLM | `gpt-4o-mini`, temperature=0 |
| Retrieved chunks k | 3 |
| Chunk separator (R₀) | `"\n\n---\n\n"` |
| BERTScore backbone | `roberta-large`, device=`cpu` |
| bert-score version | `0.3.13` (pinned) |
| R₂ scoring fields | `summary + " " + " ".join(key_points)` only (`code_example` excluded) |
| `data/test_set.json` | **Immutable ground truth — do not modify** |

## Output Format

`paper_results/table1.md` must contain two tables (Table 1a: F1 per layer with Δ₁/Δ₂; Table 1b: P/R per layer). All values to 4 decimal places. See SPEC.md "Output Required for Paper" for exact column headers and stdout summary requirements.

## Process Management Rule

The MCP server subprocess is started and stopped **only** in `eval/run_eval.py` and `ui/app.py`. Never start the server inside `agent/` code.
