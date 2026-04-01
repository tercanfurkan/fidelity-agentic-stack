# fidelity-agentic-stack — Implementation Specification

## Purpose

This repository implements the end-to-end MCP→A2A→A2UI pipeline and the layer-wise fidelity evaluation harness described in the paper:

> **Layer-Wise Fidelity Evaluation of Multi-Layer Agentic AI Pipelines: MCP, A2A, and A2UI**
> Paper source: `/Users/furkan/mcppaper/output/paper.md`

The goal is to produce the experimental results that fill the placeholders in Section 5 of the paper (Table 1, Findings). Everything built here feeds directly into the paper — no scope creep.

---

## Research Context

The paper proposes *layer-wise fidelity scoring*: BERTScore F1 computed independently at three pipeline intermediates, each against the same ground-truth reference answer:

| Intermediate | What it captures | Pipeline stage |
|---|---|---|
| **R₀** | Raw MCP tool return (retrieved chunks) | Captured by orchestrator after MCP tool call |
| **R₁** | A2A agent's synthesised NL answer | Captured by orchestrator after LLM synthesis |
| **R₂** | A2UI JSON payload text fields | Captured by formatter after JSON structuring |

Four attribution deltas are derived. Δ₁ and Δ₂ are F1 differences; Recall deltas diagnose length-confound effects:
- **Δ₁ = F1(R₁) − F1(R₀)** — A2A synthesis effect on F1
- **Δ₁ᴿ = R(R₁) − R(R₀)** — A2A effect on Recall; stable/positive alongside positive Δ₁ confirms condensation not loss
- **Δ₂ = F1(R₂) − F1(R₁)** — A2UI formatting effect on F1
- **Δ₂ᴿ = R(R₂) − R(R₁)** — A2UI effect on Recall; large negative indicates JSON schema discards content

Three research questions:
- **Q1**: Does A2A synthesis improve or degrade fidelity vs. raw retrieval? (Δ₁)
- **Q2**: Does A2UI formatting further degrade or improve coherence? (Δ₂)
- **Q3**: How much fidelity survives the full pipeline? (F1(R₀) vs F1(R₂))

Three testable hypotheses (any outcome is a valid finding):
- **H-A**: Lossy pipeline — F1(R₀) > F1(R₁) > F1(R₂)
- **H-B**: Synthesis adds value — F1(R₁) > F1(R₀), Δ₁ > 0
- **H-C**: A2UI is the main loss point — F1(R₁) >> F1(R₂), Δ₂ << 0

---

## Domain and Test Set

**Corpus**: httpx Python library documentation (https://www.python-httpx.org/).
Chosen because answers are precise, keyword-dense, and grounded in a single authoritative source — making reference answer construction tractable and BERTScore meaningful.

**Test queries**: 30 queries covering:
- Parameter behaviour (e.g., timeout configuration, follow_redirects)
- Error handling (e.g., httpx.TimeoutException, ConnectError)
- Authentication patterns (e.g., BasicAuth, BearerAuth)
- Method signatures (e.g., httpx.Client, httpx.AsyncClient)

**Critical constraint**: All 30 queries must require grounded retrieval — parametric LLM knowledge alone must be insufficient to answer them correctly without tool access. This is what makes the MCP layer scientifically necessary.

**Test set authorship**: The complete `data/test_set.json` with all 30 queries and human-written reference answers is **pre-populated in this repository** and must not be modified by Claude Code under any circumstances. Reference answers were written from primary httpx documentation, independent of any system output. Claude Code must treat this file as immutable ground truth.

---

## Tech Stack

| Component | Choice | Reason |
|---|---|---|
| MCP server | FastMCP (HTTP/SSE transport) | Python-native, easy subprocess management |
| MCP client | `fastmcp.Client` | Official FastMCP Python client |
| Embedding model | `text-embedding-3-small` (OpenAI) | Cheap, fast, good quality |
| Agent framework | LangGraph | Explicit node graph, easy to intercept intermediates |
| LLM | `gpt-4o-mini`, temperature=0 | Deterministic, low cost |
| UI | Streamlit | Fast to build |
| Scoring | `bert-score==0.3.13`, `roberta-large` backbone | Pinned version for reproducibility |

---

## Repository Structure

```
fidelity-agentic-stack/
├── SPEC.md                        # This file — do not modify
├── README.md                      # Short usage guide (generate after build)
├── requirements.txt               # Pinned Python dependencies
├── .env.example                   # Environment variable template
│
├── data/
│   ├── test_set.json              # 30 queries + reference answers — DO NOT MODIFY
│   ├── httpx_docs/                # Raw httpx markdown docs (populated by scripts/fetch_docs.py)
│   └── faiss_index/               # FAISS index files (populated by mcp_server/index.py)
│       ├── index.faiss
│       └── index.pkl
│
├── scripts/
│   └── fetch_docs.py              # Downloads httpx docs from GitHub into data/httpx_docs/
│
├── mcp_server/
│   ├── __init__.py
│   ├── server.py                  # FastMCP server, HTTP/SSE transport, port 8765
│   └── index.py                   # FAISS index builder — reads data/httpx_docs/, writes data/faiss_index/
│
├── agent/
│   ├── __init__.py
│   ├── graph.py                   # LangGraph graph definition + state schema
│   ├── orchestrator.py            # Node 1: MCP call → R₀ capture → LLM synthesis → R₁ capture
│   └── formatter.py               # Node 2: R₁ → JSON schema → R₂ capture
│
├── ui/
│   ├── __init__.py
│   └── app.py                     # Streamlit app — manual inspection only, not part of eval loop
│
├── eval/
│   ├── __init__.py
│   ├── run_eval.py                # Evaluation runner — manages MCP server subprocess, runs all 30 queries
│   ├── score.py                   # BERTScore F1 computation for R₀, R₁, R₂
│   └── results/                   # Created at runtime — do not commit contents
│       ├── raw_logs.json
│       └── scores.csv
│
└── paper_results/
    └── table1.md                  # Generated by run_eval.py — ready for paper copy-paste
```

---

## Data Format

### `data/test_set.json` (pre-populated, immutable)

```json
[
  {
    "id": "Q01",
    "query": "How do I set a timeout for an httpx request?",
    "reference": "Pass a timeout value in seconds to the request or Client: httpx.get(url, timeout=10.0). To disable timeouts use timeout=None. The default is 5 seconds. Set a client-wide default with httpx.Client(timeout=10.0)."
  },
  {
    "id": "Q02",
    "query": "What are the four timeout types in httpx and how do I configure them individually?",
    "reference": "httpx supports connect, read, write, and pool timeouts. Configure them individually with httpx.Timeout: httpx.Timeout(10.0, connect=60.0) sets a 60s connect timeout and 10s for all others. Pass the instance to a Client: httpx.Client(timeout=timeout)."
  },
  {
    "id": "Q03",
    "query": "Does httpx follow redirects by default, and how do I enable it?",
    "reference": "httpx does not follow redirects by default — follow_redirects=False is the default for all request methods. Pass follow_redirects=True on a per-request basis or when constructing a Client to enable automatic redirect following."
  },
  {
    "id": "Q04",
    "query": "What exception does httpx raise when a request times out?",
    "reference": "httpx raises httpx.TimeoutException as the base class. Subclasses are ConnectTimeout (connection), ReadTimeout (receiving data), WriteTimeout (sending data), and PoolTimeout (waiting for a pool connection)."
  },
  {
    "id": "Q05",
    "query": "What exception does httpx raise when it cannot connect to a host?",
    "reference": "httpx raises httpx.ConnectError when a connection cannot be established. The base class httpx.NetworkError covers all network-related errors. httpx.HTTPStatusError is separate and is only raised when calling response.raise_for_status() on 4xx/5xx responses."
  },
  {
    "id": "Q06",
    "query": "How do I reuse TCP connections across multiple httpx requests?",
    "reference": "Use httpx.Client as a context manager: with httpx.Client() as client: client.get(url). The Client uses HTTP connection pooling, reusing the underlying TCP connection to the same host. The top-level API (httpx.get()) creates a new connection for every request."
  },
  {
    "id": "Q07",
    "query": "How do I configure a base URL on an httpx Client so I don't repeat it in every request?",
    "reference": "Pass base_url to the Client constructor: httpx.Client(base_url='http://httpbin.org'). Subsequent requests using relative URLs are resolved against this base: client.get('/headers') sends to http://httpbin.org/headers."
  },
  {
    "id": "Q08",
    "query": "What happens when I set headers on both the httpx Client and an individual request?",
    "reference": "Headers are merged — client-level and request-level headers are combined. The same applies to query parameters and cookies. For all other parameters such as auth, the request-level value takes priority over the client-level value."
  },
  {
    "id": "Q09",
    "query": "How do I make asynchronous HTTP requests with httpx?",
    "reference": "Use httpx.AsyncClient with async with and await: async with httpx.AsyncClient() as client: r = await client.get(url). All request methods on AsyncClient are coroutines. To close without a context manager, use await client.aclose()."
  },
  {
    "id": "Q10",
    "query": "How do I disable SSL certificate verification in httpx?",
    "reference": "Pass verify=False to the request: httpx.get(url, verify=False). For a Client instance, pass verify=False at instantiation. To use a custom CA bundle, pass an SSL context object as the verify argument."
  },
  {
    "id": "Q11",
    "query": "How do I stream a large response with httpx without loading it into memory?",
    "reference": "Use the stream() context manager: with client.stream('GET', url) as response: for chunk in response.iter_bytes(): ... For async streaming use async with client.stream(...) and async for chunk in response.aiter_bytes()."
  },
  {
    "id": "Q12",
    "query": "How do I use HTTP Basic authentication with httpx?",
    "reference": "Pass a (username, password) tuple as the auth argument: httpx.get(url, auth=('user', 'pass')). Or use httpx.BasicAuth(username='user', password='secret'). Auth can be set per-request or on the Client constructor to apply to all requests."
  },
  {
    "id": "Q13",
    "query": "How do I use HTTP Digest authentication with httpx?",
    "reference": "Use httpx.DigestAuth(username='user', password='secret') as the auth argument. Digest auth requires an additional round-trip to negotiate and can be used over unencrypted HTTP connections, unlike Basic auth."
  },
  {
    "id": "Q14",
    "query": "How do I raise an exception for error HTTP responses in httpx?",
    "reference": "Call response.raise_for_status() on the response. This raises httpx.HTTPStatusError for 4xx and 5xx status codes. Successful responses return the response instance unchanged, so it can be chained: r = httpx.get(url).raise_for_status()."
  },
  {
    "id": "Q15",
    "query": "How do I send a Bearer token with httpx?",
    "reference": "Set the Authorization header directly: httpx.get(url, headers={'Authorization': 'Bearer your-token'}). httpx does not have a built-in BearerAuth class — bearer tokens are passed as a custom header, either per-request or via the Client constructor."
  }
]
```

### `eval/results/raw_logs.json` (generated at runtime)

```json
[
  {
    "id": "Q01",
    "query": "How do I configure a timeout for an httpx request?",
    "r0": "<concatenated raw chunk text separated by \\n\\n---\\n\\n>",
    "r1": "<synthesised NL answer from orchestrator LLM>",
    "r2": {
      "summary": "<1-2 sentence summary>",
      "key_points": ["<point 1>", "<point 2>"],
      "code_example": "<code snippet or empty string>",
      "source_ref": "httpx documentation"
    },
    "r2_text": "<summary + space + key_points joined by space — used for BERTScore>",
    "error": null
  }
]
```

On failure for a query, set `"r0": null, "r1": null, "r2": null, "r2_text": null, "error": "<error message>"`.

### `eval/results/scores.csv` (generated at runtime)

Capture Precision (P), Recall (R), and F1 for each layer. This is required to diagnose the length-disparity effect between R₀ (long) and R₁ (short) — see SPEC section on Length Disparity.

```
id,p_r0,r_r0,f1_r0,p_r1,r_r1,f1_r1,p_r2,r_r2,f1_r2,delta1_f1,delta2_f1,delta1_recall,delta2_recall
Q01,0.7812,0.9134,0.8421,...
...
Mean,...
```

On failure for a query, write empty strings for score fields. Do not include failed queries in the Mean calculation.

**Why P and R matter**: R₀ will naturally have low Precision (long text, many tokens irrelevant to short reference) but high Recall (answer is present somewhere in the chunks). R₁ should have higher Precision (condensed) with maintained Recall (faithful). Reporting both allows the paper to distinguish condensation (Precision rise) from information loss (Recall drop).

---

## MCP ↔ LangGraph Integration Architecture

This is the most critical architectural decision. The integration uses **FastMCP HTTP/SSE transport** with the **FastMCP Python client** called from within a LangGraph node.

### Transport choice: HTTP/SSE (not stdio)

Use HTTP/SSE transport because:
- The eval runner needs to start the MCP server as a managed subprocess and connect to it programmatically
- stdio transport requires the client to own the server process lifecycle, which is harder to manage in an eval loop
- HTTP/SSE allows the server to run independently and the Streamlit UI to connect to it too

### Server: `mcp_server/server.py`

```python
from fastmcp import FastMCP
# server listens on http://localhost:8765
mcp = FastMCP("httpx-retriever", host="localhost", port=8765)
```

### Client usage inside LangGraph orchestrator node

```python
import asyncio
from fastmcp import Client

async def call_mcp_retrieve(query: str, k: int = 3) -> str:
    async with Client("http://localhost:8765/sse") as client:
        result = await client.call_tool("retrieve", {"query": query, "k": k})
        return result[0].text  # FastMCP tool result is a list of content items

# In the orchestrator node (LangGraph nodes are sync by default):
def orchestrator_node(state: PipelineState) -> PipelineState:
    r0 = asyncio.run(call_mcp_retrieve(state["query"], k=3))
    ...
```

### Process management in `eval/run_eval.py`

The eval runner starts the MCP server as a subprocess, waits for it to be ready, runs all queries, then terminates it:

```python
import subprocess, time, httpx, signal

def start_mcp_server() -> subprocess.Popen:
    proc = subprocess.Popen(
        ["python", "-m", "mcp_server.server"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    # Wait until server is accepting connections (max 15 seconds)
    for _ in range(30):
        try:
            httpx.get("http://localhost:8765/health", timeout=0.5)
            return proc
        except Exception:
            time.sleep(0.5)
    proc.terminate()
    raise RuntimeError("MCP server failed to start within 15 seconds")

def stop_mcp_server(proc: subprocess.Popen):
    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=5)
```

The MCP server must expose a `/health` endpoint that returns HTTP 200. Add this to `server.py`.

---

## LangGraph State Schema

Define the state as a TypedDict in `agent/graph.py`. All nodes read from and write to this schema.

```python
from typing import TypedDict, Optional

class PipelineState(TypedDict):
    query: str           # Input query string
    r0: Optional[str]    # Raw MCP tool return (concatenated chunks)
    r1: Optional[str]    # Synthesised NL answer from orchestrator LLM
    r2: Optional[dict]   # JSON payload dict from formatter
    r2_text: Optional[str]  # Concatenated r2 text fields for scoring
    error: Optional[str] # Error message if any node fails
```

The graph definition:

```python
from langgraph.graph import StateGraph, END

def build_graph() -> StateGraph:
    graph = StateGraph(PipelineState)
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("formatter", formatter_node)
    graph.set_entry_point("orchestrator")
    graph.add_edge("orchestrator", "formatter")
    graph.add_edge("formatter", END)
    return graph.compile()
```

---

## Implementation Requirements

### httpx Docs Fetching (`scripts/fetch_docs.py`)

Download the httpx documentation markdown files from the official GitHub repository into `data/httpx_docs/`. Do not scrape the website.

```python
# Download from: https://github.com/encode/httpx/tree/master/docs
# Use the GitHub raw content API or git clone --depth=1 --filter=blob:none --sparse
# Target files: all *.md files under the docs/ directory
# Save to: data/httpx_docs/<filename>.md
```

Use `git clone --depth=1 https://github.com/encode/httpx.git /tmp/httpx_repo` then copy `docs/*.md` to `data/httpx_docs/`. Clean up the temp clone after copying.

The fetch script must be idempotent — if `data/httpx_docs/` already contains files, skip the download.

### FAISS Index Builder (`mcp_server/index.py`)

- Read all `.md` files from `data/httpx_docs/`
- Chunk with size ~300 tokens, overlap 50 tokens using `langchain.text_splitter.RecursiveCharacterTextSplitter`
- Embed with `text-embedding-3-small`
- Save FAISS index to `data/faiss_index/index.faiss` and metadata to `data/faiss_index/index.pkl`
- Must be idempotent — if `data/faiss_index/` already exists with both files, skip rebuild

### MCP Server (`mcp_server/server.py`)

```python
from fastmcp import FastMCP
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import pickle, os

mcp = FastMCP("httpx-retriever", host="localhost", port=8765)

# Load index at startup
FAISS_PATH = "data/faiss_index/index.faiss"
PKL_PATH = "data/faiss_index/index.pkl"

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.load_local("data/faiss_index", embeddings, allow_dangerous_deserialization=True)

@mcp.tool()
def retrieve(query: str, k: int = 3) -> str:
    """Retrieve top-k relevant chunks from httpx documentation."""
    docs = vectorstore.similarity_search(query, k=k)
    # Join with explicit separator — used verbatim as R₀
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

@mcp.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    mcp.run(transport="sse")
```

**R₀ is NOT logged in the server.** The server only returns the concatenated string. R₀ is captured by the orchestrator node when it receives the tool result.

Chunk separator is fixed as `"\n\n---\n\n"` — do not change this.

### LangGraph Orchestrator Node (`agent/orchestrator.py`)

```python
from langchain_openai import ChatOpenAI
from agent.graph import PipelineState
import asyncio
from fastmcp import Client

ORCHESTRATOR_SYSTEM_PROMPT = """You are a documentation assistant. \
Answer the user's question using ONLY the retrieved context provided. \
Do not add information not present in the context. \
Be concise and precise. Do not hedge or qualify your answer."""

def orchestrator_node(state: PipelineState) -> PipelineState:
    # Step 1: Call MCP tool and capture R₀
    r0 = asyncio.run(_call_mcp(state["query"], k=3))

    # Step 2: Synthesise answer using LLM and capture R₁
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    messages = [
        {"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {state['query']}\n\nRetrieved context:\n{r0}"}
    ]
    response = llm.invoke(messages)
    r1 = response.content

    return {**state, "r0": r0, "r1": r1}

async def _call_mcp(query: str, k: int) -> str:
    async with Client("http://localhost:8765/sse") as client:
        result = await client.call_tool("retrieve", {"query": query, "k": k})
        return result[0].text
```

### LangGraph Formatter Node (`agent/formatter.py`)

The formatter uses a second constrained LLM call with `response_format={"type": "json_object"}` and temperature=0 to convert R₁ into the fixed JSON schema. This is not a free-form generation step — it is a structured reformatting of R₁.

```python
from langchain_openai import ChatOpenAI
from agent.graph import PipelineState
import json

FORMATTER_SYSTEM_PROMPT = """You are a structured formatting assistant. \
Convert the provided answer into a JSON object with exactly these fields:
- "summary": a 1-2 sentence summary of the answer (string)
- "key_points": a list of 2-4 key points extracted from the answer (list of strings)
- "code_example": a relevant code snippet if present in the answer, otherwise an empty string (string)
- "source_ref": always set to "httpx documentation" (string)

Return ONLY valid JSON. Do not add any fields not listed above."""

def formatter_node(state: PipelineState) -> PipelineState:
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        model_kwargs={"response_format": {"type": "json_object"}}
    )
    messages = [
        {"role": "system", "content": FORMATTER_SYSTEM_PROMPT},
        {"role": "user", "content": state["r1"]}
    ]
    response = llm.invoke(messages)

    # Parse JSON — on failure, record error and set r2 fields to None
    try:
        r2 = json.loads(response.content)
        # Validate required fields are present
        for field in ["summary", "key_points", "code_example", "source_ref"]:
            if field not in r2:
                raise ValueError(f"Missing field: {field}")
        # Build r2_text for scoring:
        # code_example is intentionally excluded — code token patterns distort BERTScore
        r2_text = r2["summary"] + " " + " ".join(r2["key_points"])
        return {**state, "r2": r2, "r2_text": r2_text}
    except (json.JSONDecodeError, ValueError) as e:
        return {**state, "r2": None, "r2_text": None, "error": f"Formatter JSON error: {e}"}
```

**Why `code_example` is excluded from R₂ scoring**: Code token patterns (variable names, syntax symbols) distort BERTScore contextual similarity in ways that do not reflect semantic fidelity. Scoring on `summary + key_points` captures the informational content of R₂ without the noise introduced by code tokens.

### Scoring (`eval/score.py`)

```python
from bert_score import score as bert_score
from typing import List, Optional

def compute_bertscore(
    candidates: List[str],
    references: List[str]
) -> List[dict]:
    """
    Compute BERTScore Precision, Recall, and F1 for a list of (candidate, reference) pairs.
    Returns a list of dicts with keys 'p', 'r', 'f1'.
    Uses roberta-large backbone, pinned via bert-score==0.3.13.
    """
    P, R, F1 = bert_score(
        candidates,
        references,
        model_type="roberta-large",
        lang="en",
        verbose=False,
        device="cpu"  # use cpu for reproducibility; change to "cuda" if available
    )
    return [
        {"p": round(p, 4), "r": round(r, 4), "f1": round(f, 4)}
        for p, r, f in zip(P.tolist(), R.tolist(), F1.tolist())
    ]

def score_all(
    r0_list: List[Optional[str]],
    r1_list: List[Optional[str]],
    r2_text_list: List[Optional[str]],
    reference_list: List[str]
) -> List[dict]:
    """
    Score R₀, R₁, R₂ for all queries in a single batched call per layer.
    Returns list of dicts with p/r/f1 for each layer, plus F1 and Recall deltas.
    Handles None values (failed queries) by returning None for all score fields.

    IMPORTANT: P and R are both reported because R₀ is ~900 tokens (3 chunks) while
    the reference is 1-3 sentences. R₀ will naturally have low Precision but high Recall.
    R₁ should have higher Precision (condensed) with maintained Recall (faithful).
    delta1_recall distinguishes genuine condensation from information loss.
    """
    empty = {
        "p_r0": None, "r_r0": None, "f1_r0": None,
        "p_r1": None, "r_r1": None, "f1_r1": None,
        "p_r2": None, "r_r2": None, "f1_r2": None,
        "delta1_f1": None, "delta2_f1": None,
        "delta1_recall": None, "delta2_recall": None
    }
    results = [dict(empty) for _ in reference_list]

    for prefix, candidates in [("r0", r0_list), ("r1", r1_list), ("r2", r2_text_list)]:
        valid_indices = [i for i, c in enumerate(candidates) if c is not None]
        if not valid_indices:
            continue
        valid_candidates = [candidates[i] for i in valid_indices]
        valid_references = [reference_list[i] for i in valid_indices]
        scores = compute_bertscore(valid_candidates, valid_references)
        for idx, s in zip(valid_indices, scores):
            results[idx][f"p_{prefix}"] = s["p"]
            results[idx][f"r_{prefix}"] = s["r"]
            results[idx][f"f1_{prefix}"] = s["f1"]

    # Compute deltas on F1 and Recall
    for r in results:
        if r["f1_r0"] is not None and r["f1_r1"] is not None:
            r["delta1_f1"] = round(r["f1_r1"] - r["f1_r0"], 4)
            r["delta1_recall"] = round(r["r_r1"] - r["r_r0"], 4)
        if r["f1_r1"] is not None and r["f1_r2"] is not None:
            r["delta2_f1"] = round(r["f1_r2"] - r["f1_r1"], 4)
            r["delta2_recall"] = round(r["r_r2"] - r["r_r1"], 4)

    return results
```

**Important**: Score all candidates for a given layer in a single `bert_score()` call (not one call per query). This is both faster and ensures consistent internal normalisation.

### Evaluation Runner (`eval/run_eval.py`)

```python
"""
Evaluation runner. Manages MCP server subprocess, runs all 30 queries,
captures R₀/R₁/R₂, scores with BERTScore, writes results.
"""
import json, csv, subprocess, time, signal, sys
import httpx as _httpx
from pathlib import Path
from agent.graph import build_graph, PipelineState
from eval.score import score_all

RESULTS_DIR = Path("eval/results")
PAPER_RESULTS_DIR = Path("paper_results")

def start_mcp_server() -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, "-m", "mcp_server.server"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    for _ in range(30):
        try:
            _httpx.get("http://localhost:8765/health", timeout=0.5)
            print("MCP server ready.")
            return proc
        except Exception:
            time.sleep(0.5)
    proc.terminate()
    raise RuntimeError("MCP server failed to start within 15 seconds.")

def stop_mcp_server(proc: subprocess.Popen):
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()

def run():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load test set
    test_set = json.loads(Path("data/test_set.json").read_text())

    # Build LangGraph pipeline
    graph = build_graph()

    # Start MCP server subprocess
    mcp_proc = start_mcp_server()

    raw_logs = []
    try:
        for item in test_set:
            print(f"Running {item['id']}: {item['query'][:60]}...")
            try:
                initial_state: PipelineState = {
                    "query": item["query"],
                    "r0": None, "r1": None,
                    "r2": None, "r2_text": None,
                    "error": None
                }
                result = graph.invoke(initial_state)
                raw_logs.append({
                    "id": item["id"],
                    "query": item["query"],
                    "r0": result["r0"],
                    "r1": result["r1"],
                    "r2": result["r2"],
                    "r2_text": result["r2_text"],
                    "error": result["error"]
                })
            except Exception as e:
                print(f"  ERROR: {e}")
                raw_logs.append({
                    "id": item["id"],
                    "query": item["query"],
                    "r0": None, "r1": None,
                    "r2": None, "r2_text": None,
                    "error": str(e)
                })
    finally:
        stop_mcp_server(mcp_proc)

    # Write raw logs
    (RESULTS_DIR / "raw_logs.json").write_text(
        json.dumps(raw_logs, indent=2, ensure_ascii=False)
    )

    # Score all layers
    references = [item["reference"] for item in test_set]
    r0_list = [log["r0"] for log in raw_logs]
    r1_list = [log["r1"] for log in raw_logs]
    r2_list = [log["r2_text"] for log in raw_logs]

    scores = score_all(r0_list, r1_list, r2_list, references)

    # Write scores.csv — 14 columns matching score_all() output keys
    SCORE_KEYS = [
        "p_r0", "r_r0", "f1_r0",
        "p_r1", "r_r1", "f1_r1",
        "p_r2", "r_r2", "f1_r2",
        "delta1_f1", "delta2_f1",
        "delta1_recall", "delta2_recall"
    ]
    with open(RESULTS_DIR / "scores.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id"] + SCORE_KEYS)
        for item, score in zip(test_set, scores):
            writer.writerow([item["id"]] + [score[k] if score[k] is not None else "" for k in SCORE_KEYS])
        # Mean row — exclude failed queries (None values)
        valid = [s for s in scores if s["f1_r0"] is not None]
        if valid:
            mean_row = ["Mean"] + [
                round(sum(s[k] for s in valid if s[k] is not None) / len(valid), 4)
                for k in SCORE_KEYS
            ]
            writer.writerow(mean_row)

    # Generate paper_results/table1.md
    _write_paper_table(test_set, scores, PAPER_RESULTS_DIR / "table1.md")

    print("\n=== EVALUATION COMPLETE ===")
    print(f"Raw logs: {RESULTS_DIR}/raw_logs.json")
    print(f"Scores:   {RESULTS_DIR}/scores.csv")
    print(f"Table 1:  {PAPER_RESULTS_DIR}/table1.md")

def _write_paper_table(test_set, scores, path: Path):
    fmt = lambda v: f"{v:.4f}" if v is not None else "—"
    valid_scores = [s for s in scores if s["f1_r0"] is not None]
    def mean(k): return round(sum(s[k] for s in valid_scores if s[k] is not None) / len(valid_scores), 4)

    # Table 1a: F1 per layer
    lines = [
        "## Table 1a. BERTScore F1 per layer",
        "",
        "| Query ID | F1(R₀) | F1(R₁) | F1(R₂) | Δ₁ | Δ₂ |",
        "|---|---|---|---|---|---|"
    ]
    for item, s in zip(test_set, scores):
        lines.append(
            f"| {item['id']} | {fmt(s['f1_r0'])} | {fmt(s['f1_r1'])} | "
            f"{fmt(s['f1_r2'])} | {fmt(s['delta1_f1'])} | {fmt(s['delta2_f1'])} |"
        )
    if valid_scores:
        lines.append(
            f"| **Mean** | **{mean('f1_r0')}** | **{mean('f1_r1')}** | "
            f"**{mean('f1_r2')}** | **{mean('delta1_f1')}** | **{mean('delta2_f1')}** |"
        )

    # Table 1b: Precision and Recall per layer
    lines += [
        "",
        "## Table 1b. BERTScore Precision and Recall per layer (P/R decomposition)",
        "",
        "| Query ID | P(R₀) | R(R₀) | P(R₁) | R(R₁) | P(R₂) | R(R₂) | Δ₁ᴿ | Δ₂ᴿ |",
        "|---|---|---|---|---|---|---|---|---|"
    ]
    for item, s in zip(test_set, scores):
        lines.append(
            f"| {item['id']} | {fmt(s['p_r0'])} | {fmt(s['r_r0'])} | "
            f"{fmt(s['p_r1'])} | {fmt(s['r_r1'])} | "
            f"{fmt(s['p_r2'])} | {fmt(s['r_r2'])} | "
            f"{fmt(s['delta1_recall'])} | {fmt(s['delta2_recall'])} |"
        )
    if valid_scores:
        lines.append(
            f"| **Mean** | **{mean('p_r0')}** | **{mean('r_r0')}** | "
            f"**{mean('p_r1')}** | **{mean('r_r1')}** | "
            f"**{mean('p_r2')}** | **{mean('r_r2')}** | "
            f"**{mean('delta1_recall')}** | **{mean('delta2_recall')}** |"
        )

    path.write_text("\n".join(lines) + "\n")

if __name__ == "__main__":
    run()
```

### Streamlit UI (`ui/app.py`)

- Accept a query string input via `st.text_input`
- On submit: start a fresh MCP server subprocess if not already running, run the LangGraph pipeline, display R₂ as a structured card
- **This is for manual inspection only — it is not part of the automated eval loop**
- The UI does not write to `eval/results/` or `paper_results/`

Display structure:
- `summary` → `st.info()` callout box
- `key_points` → `st.markdown()` bulleted list
- `code_example` → `st.code()` block (skip if empty string)
- `source_ref` → `st.caption()`

---

## Controlled Variables

These must be held constant across all 30 queries. Any deviation invalidates the experimental results.

| Variable | Fixed value |
|---|---|
| LLM (both nodes) | `gpt-4o-mini` |
| Temperature (both nodes) | `0` |
| Retrieved chunks k | `3` |
| Re-ranking | None — flat cosine similarity only |
| Chunk separator | `"\n\n---\n\n"` |
| BERTScore backbone | `roberta-large` |
| BERTScore device | `cpu` |
| bert-score version | `0.3.13` (pinned) |
| R₂ scoring fields | `summary + " " + " ".join(key_points)` only |
| Reference answers | Pre-populated in `data/test_set.json` — immutable |

---

## Environment Setup

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install pinned dependencies
pip install -r requirements.txt

# Copy and fill environment variables
cp .env.example .env
# Edit .env and set OPENAI_API_KEY

# Step 1: Download httpx documentation
python scripts/fetch_docs.py

# Step 2: Build FAISS index
python mcp_server/index.py

# Step 3: Run evaluation (starts/stops MCP server automatically)
python eval/run_eval.py

# Optional: Launch UI for manual inspection
streamlit run ui/app.py
```

---

## `requirements.txt` (pinned)

```
fastmcp>=0.9.0
langchain>=0.3.0
langchain-community>=0.3.0
langgraph>=0.2.0
langchain-openai>=0.2.0
faiss-cpu>=1.8.0
bert-score==0.3.13
streamlit>=1.35.0
openai>=1.30.0
tiktoken>=0.7.0
python-dotenv>=1.0.0
httpx>=0.27.0
torch>=2.0.0
```

---

## `.env.example`

```
OPENAI_API_KEY=your-openai-api-key-here
```

---

## `__init__.py` files

Create empty `__init__.py` files in every package directory to enable cross-module imports:

```
mcp_server/__init__.py   (empty)
agent/__init__.py        (empty)
eval/__init__.py         (empty)
ui/__init__.py           (empty)
scripts/__init__.py      (empty)
```

---

## Output Required for Paper

The evaluation must produce numbers to fill **Table 1** in Section 5 of the paper at `/Users/furkan/mcppaper/output/paper.md`.

`paper_results/table1.md` must contain **two tables**:

**Table 1a — BERTScore F1 per layer** (primary, for paper Table 1):
```
| Query ID | F1(R₀) | F1(R₁) | F1(R₂) | Δ₁ F1 | Δ₂ F1 |
```

**Table 1b — BERTScore Precision and Recall per layer** (for P/R decomposition, addresses length bias):
```
| Query ID | P(R₀) | Recall(R₀) | P(R₁) | Recall(R₁) | P(R₂) | Recall(R₂) | Δ₁ Recall | Δ₂ Recall |
```

All values to 4 decimal places.

Also required for Section 5.2 (Findings) — print these to stdout at the end of `run_eval.py`:
- Mean F1, P, R for each layer
- Mean Δ₁ F1 with direction label: "A2A synthesis IMPROVES F1" if positive, "DEGRADES" if negative
- Mean Δ₁ Recall: "Recall MAINTAINED (condensation only)" if > -0.01, "Recall DROPPED (information loss)" if < -0.01
- Mean Δ₂ F1 and Recall with equivalent direction labels
- Which layer transition has the largest absolute delta on both F1 and Recall

---

## What Claude Code Must NOT Do

- Do not modify `data/test_set.json` — it is pre-populated and immutable ground truth
- Do not change the R₂ JSON schema — scoring depends on the exact fields
- Do not add re-ranking — k=3 flat retrieval is intentional
- Do not use a different LLM, temperature, or BERTScore version
- Do not add streaming, multi-turn conversation, memory, or auth to the pipeline
- Do not make the Streamlit UI part of the automated eval loop
- Do not score `code_example` as part of R₂ — it distorts BERTScore
- Do not start the MCP server inside `agent/` code — process management belongs in `eval/run_eval.py` and `ui/app.py` only

---

## Success Criteria

The implementation is complete when all of the following are true:

1. `scripts/fetch_docs.py` runs without error and populates `data/httpx_docs/` with httpx markdown files
2. `mcp_server/index.py` runs without error and creates `data/faiss_index/index.faiss` and `data/faiss_index/index.pkl`
3. `python eval/run_eval.py` completes without crashing, starts and stops the MCP server automatically
4. `eval/results/raw_logs.json` contains 30 entries, each with non-null `r0`, `r1`, `r2`, `r2_text`
5. `eval/results/scores.csv` contains 30 data rows plus a Mean row, all score cells populated
6. `paper_results/table1.md` exists and contains a valid markdown table with 4 decimal place scores
7. BERTScore F1 values are in a plausible range (0.80–0.97 for this task type)
8. `streamlit run ui/app.py` renders a structured card for a sample query without errors
