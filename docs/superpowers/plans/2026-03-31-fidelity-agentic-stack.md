# Fidelity Agentic Stack Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an MCP→A2A→A2UI evaluation pipeline that runs 15 httpx doc queries through a three-layer agent pipeline, captures intermediates (R₀, R₁, R₂), scores each with BERTScore, and outputs Table 1 for the paper.

**Architecture:** A FastMCP HTTP/SSE server serves a FAISS-backed `retrieve` tool. A LangGraph two-node graph (orchestrator → formatter) calls MCP, synthesises with GPT-4o-mini, and formats to JSON. An eval runner manages the server subprocess, runs all queries, and writes scored results.

**Tech Stack:** FastMCP, LangGraph, LangChain + OpenAI, FAISS, BERTScore (roberta-large), Streamlit, Python 3.11+

**Spec document:** `SPEC.md` in repo root — authoritative source for all data formats, controlled variables, and constraints.

---

### Task 1: Project Scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `mcp_server/__init__.py`
- Create: `agent/__init__.py`
- Create: `eval/__init__.py`
- Create: `ui/__init__.py`
- Create: `scripts/__init__.py`
- Create: `data/test_set.json`

- [ ] **Step 1: Create requirements.txt**

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

- [ ] **Step 2: Create .env.example**

```
OPENAI_API_KEY=your-openai-api-key-here
```

- [ ] **Step 3: Create empty `__init__.py` files**

Create empty files in: `mcp_server/`, `agent/`, `eval/`, `ui/`, `scripts/`

- [ ] **Step 4: Create data/test_set.json**

This is the immutable ground truth. Copy verbatim from SPEC.md — all 15 queries (Q01–Q15) with `id`, `query`, and `reference` fields. Once created, this file must never be modified.

```json
[
  {
    "id": "Q01",
    "query": "How do I configure a timeout for an httpx request?",
    "reference": "Pass a timeout parameter to the request method or Client constructor. httpx.get(url, timeout=5.0) sets a 5-second timeout. Use httpx.Timeout for fine-grained control over connect, read, write, and pool timeouts."
  },
  {
    "id": "Q02",
    "query": "How do I use BasicAuth with httpx?",
    "reference": "Pass an auth parameter to the request or Client. Use httpx.BasicAuth('username', 'password') or pass a (username, password) tuple directly: httpx.get(url, auth=('user', 'pass'))."
  },
  {
    "id": "Q03",
    "query": "How do I follow redirects in httpx?",
    "reference": "httpx does not follow redirects by default. Set follow_redirects=True on the request or Client to enable automatic redirect following."
  },
  {
    "id": "Q04",
    "query": "What exception is raised when an httpx request times out?",
    "reference": "httpx raises httpx.TimeoutException on timeout. Subclasses include httpx.ConnectTimeout, httpx.ReadTimeout, httpx.WriteTimeout, and httpx.PoolTimeout for finer-grained handling."
  },
  {
    "id": "Q05",
    "query": "How do I send a POST request with JSON data using httpx?",
    "reference": "Pass a json parameter to httpx.post(): httpx.post(url, json={'key': 'value'}). This automatically sets the Content-Type header to application/json and serialises the dict."
  },
  {
    "id": "Q06",
    "query": "How do I create a persistent HTTP connection pool with httpx?",
    "reference": "Use httpx.Client as a context manager: with httpx.Client() as client: client.get(url). The Client maintains a connection pool across requests and should be reused rather than creating a new client per request."
  },
  {
    "id": "Q07",
    "query": "How do I set custom headers on an httpx request?",
    "reference": "Pass a headers dict to the request or Client constructor: httpx.get(url, headers={'X-Custom': 'value'}). Headers set on the Client are merged with per-request headers, with per-request headers taking precedence."
  },
  {
    "id": "Q08",
    "query": "How do I make async HTTP requests with httpx?",
    "reference": "Use httpx.AsyncClient with async/await: async with httpx.AsyncClient() as client: response = await client.get(url). AsyncClient mirrors the Client API but all request methods are coroutines."
  },
  {
    "id": "Q09",
    "query": "How do I handle connection errors in httpx?",
    "reference": "Catch httpx.ConnectError for connection failures. The broader httpx.RequestError catches all request-level errors including timeouts and connection issues. httpx.HTTPStatusError is raised separately for 4xx/5xx responses when raise_for_status() is called."
  },
  {
    "id": "Q10",
    "query": "How do I upload a file using httpx?",
    "reference": "Use the files parameter: httpx.post(url, files={'upload': open('file.txt', 'rb')}). For multipart forms with both files and fields, combine files and data parameters."
  },
  {
    "id": "Q11",
    "query": "How do I set a base URL for all requests in an httpx Client?",
    "reference": "Pass base_url to the Client constructor: client = httpx.Client(base_url='https://api.example.com'). Relative URLs in subsequent requests are resolved against this base URL."
  },
  {
    "id": "Q12",
    "query": "How do I disable SSL certificate verification in httpx?",
    "reference": "Pass verify=False to the request or Client constructor: httpx.get(url, verify=False). To use a custom CA bundle, pass verify='/path/to/ca-bundle.crt'. Disabling verification is not recommended in production."
  },
  {
    "id": "Q13",
    "query": "How do I stream a large response with httpx?",
    "reference": "Use the stream() context manager: with client.stream('GET', url) as response: for chunk in response.iter_bytes(): ... This avoids loading the entire response into memory."
  },
  {
    "id": "Q14",
    "query": "How do I add query parameters to an httpx request?",
    "reference": "Pass a params dict: httpx.get(url, params={'key': 'value', 'page': 2}). httpx URL-encodes the parameters and appends them to the URL. List values produce repeated parameters."
  },
  {
    "id": "Q15",
    "query": "How do I use Bearer token authentication with httpx?",
    "reference": "Use httpx.BearerAuth('token') as the auth parameter, or set an Authorization header directly: headers={'Authorization': 'Bearer your-token'}. BearerAuth can be set on the Client for all requests."
  }
]
```

- [ ] **Step 5: Create virtual environment and install dependencies**

Run:
```bash
cd /Users/furkan/mcppaper/fidelity-agentic-stack
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Expected: All packages install successfully.

- [ ] **Step 6: Create .gitignore**

```
.venv/
.env
__pycache__/
*.pyc
data/httpx_docs/
data/faiss_index/
eval/results/
paper_results/
*.egg-info/
.DS_Store
```

- [ ] **Step 7: Commit scaffolding**

```bash
git add requirements.txt .env.example .gitignore data/test_set.json mcp_server/__init__.py agent/__init__.py eval/__init__.py ui/__init__.py scripts/__init__.py
git commit -m "feat: project scaffolding — deps, test set, package structure"
```

---

### Task 2: Fetch Docs Script

**Files:**
- Create: `scripts/fetch_docs.py`

- [ ] **Step 1: Write fetch_docs.py**

```python
"""Download httpx documentation markdown files from GitHub."""

import shutil
import subprocess
from pathlib import Path

DOCS_DIR = Path("data/httpx_docs")


def fetch():
    if list(DOCS_DIR.glob("*.md")):
        print(f"Docs already present in {DOCS_DIR}, skipping download.")
        return

    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    tmp_dir = Path("/tmp/httpx_repo")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    print("Cloning httpx repository (shallow)...")
    subprocess.run(
        ["git", "clone", "--depth=1", "https://github.com/encode/httpx.git", str(tmp_dir)],
        check=True,
        capture_output=True,
    )

    source_docs = tmp_dir / "docs"
    count = 0
    for md_file in source_docs.rglob("*.md"):
        dest = DOCS_DIR / md_file.name
        shutil.copy2(md_file, dest)
        count += 1

    shutil.rmtree(tmp_dir)
    print(f"Downloaded {count} markdown files to {DOCS_DIR}")


if __name__ == "__main__":
    fetch()
```

- [ ] **Step 2: Run fetch_docs.py**

Run:
```bash
cd /Users/furkan/mcppaper/fidelity-agentic-stack
python scripts/fetch_docs.py
```

Expected: Prints "Downloaded N markdown files to data/httpx_docs". Verify files exist:
```bash
ls data/httpx_docs/ | head -10
```

- [ ] **Step 3: Run again to verify idempotency**

Run:
```bash
python scripts/fetch_docs.py
```

Expected: Prints "Docs already present in data/httpx_docs, skipping download."

- [ ] **Step 4: Commit**

```bash
git add scripts/fetch_docs.py
git commit -m "feat: add httpx docs fetch script"
```

---

### Task 3: FAISS Index Builder

**Files:**
- Create: `mcp_server/index.py`

- [ ] **Step 1: Write index.py**

```python
"""Build FAISS index from httpx documentation markdown files."""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

load_dotenv()

DOCS_DIR = Path("data/httpx_docs")
FAISS_DIR = Path("data/faiss_index")


def build_index():
    index_file = FAISS_DIR / "index.faiss"
    pkl_file = FAISS_DIR / "index.pkl"

    if index_file.exists() and pkl_file.exists():
        print("FAISS index already exists, skipping rebuild.")
        return

    md_files = sorted(DOCS_DIR.glob("*.md"))
    if not md_files:
        raise FileNotFoundError(f"No markdown files found in {DOCS_DIR}. Run scripts/fetch_docs.py first.")

    print(f"Reading {len(md_files)} markdown files...")
    documents = []
    for f in md_files:
        documents.append(f.read_text(encoding="utf-8"))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = splitter.create_documents(documents, metadatas=[{"source": f.name} for f in md_files])
    print(f"Created {len(chunks)} chunks.")

    print("Embedding chunks with text-embedding-3-small...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(FAISS_DIR))
    print(f"Saved FAISS index to {FAISS_DIR}")


if __name__ == "__main__":
    build_index()
```

- [ ] **Step 2: Run index.py**

Run:
```bash
cd /Users/furkan/mcppaper/fidelity-agentic-stack
python mcp_server/index.py
```

Expected: Prints chunk count and "Saved FAISS index to data/faiss_index". Verify:
```bash
ls data/faiss_index/
```
Should show `index.faiss` and `index.pkl`.

- [ ] **Step 3: Run again to verify idempotency**

Run:
```bash
python mcp_server/index.py
```

Expected: "FAISS index already exists, skipping rebuild."

- [ ] **Step 4: Commit**

```bash
git add mcp_server/index.py
git commit -m "feat: add FAISS index builder for httpx docs"
```

---

### Task 4: MCP Server

**Files:**
- Create: `mcp_server/server.py`

- [ ] **Step 1: Write server.py**

```python
"""FastMCP server exposing httpx documentation retrieval via FAISS."""

from dotenv import load_dotenv
from fastmcp import FastMCP
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

load_dotenv()

mcp = FastMCP("httpx-retriever", host="localhost", port=8765)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.load_local(
    "data/faiss_index", embeddings, allow_dangerous_deserialization=True
)


@mcp.tool()
def retrieve(query: str, k: int = 3) -> str:
    """Retrieve top-k relevant chunks from httpx documentation."""
    docs = vectorstore.similarity_search(query, k=k)
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


@mcp.custom_route("/health", methods=["GET"])
async def health(request):
    from starlette.responses import JSONResponse
    return JSONResponse({"status": "ok"})


if __name__ == "__main__":
    mcp.run(transport="sse")
```

**Note on `/health` endpoint:** FastMCP's `@mcp.custom_route` or equivalent may vary by version. If `custom_route` is not available, the health endpoint must be added via a different mechanism — check FastMCP docs at runtime. The critical requirement is that `GET http://localhost:8765/health` returns HTTP 200 with JSON `{"status": "ok"}`. If FastMCP provides no way to add custom HTTP routes, wrap the server with a lightweight ASGI middleware or use a separate health-check approach. Diagnose and adapt at implementation time.

- [ ] **Step 2: Smoke-test the server**

Start the server manually:
```bash
cd /Users/furkan/mcppaper/fidelity-agentic-stack
python -m mcp_server.server &
SERVER_PID=$!
sleep 3
```

Test health endpoint:
```bash
curl -s http://localhost:8765/health
```
Expected: `{"status": "ok"}`

Kill the server:
```bash
kill $SERVER_PID
```

If the health endpoint doesn't work, investigate FastMCP's API for adding custom HTTP routes and fix `server.py`. The health check is required for the eval runner's startup detection.

- [ ] **Step 3: Commit**

```bash
git add mcp_server/server.py
git commit -m "feat: add FastMCP server with retrieve tool and health endpoint"
```

---

### Task 5: LangGraph State and Graph

**Files:**
- Create: `agent/graph.py`

- [ ] **Step 1: Write graph.py**

```python
"""LangGraph graph definition and pipeline state schema."""

from typing import Optional, TypedDict

from langgraph.graph import END, StateGraph


class PipelineState(TypedDict):
    query: str
    r0: Optional[str]
    r1: Optional[str]
    r2: Optional[dict]
    r2_text: Optional[str]
    error: Optional[str]


def build_graph():
    from agent.formatter import formatter_node
    from agent.orchestrator import orchestrator_node

    graph = StateGraph(PipelineState)
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("formatter", formatter_node)
    graph.set_entry_point("orchestrator")
    graph.add_edge("orchestrator", "formatter")
    graph.add_edge("formatter", END)
    return graph.compile()
```

- [ ] **Step 2: Commit**

```bash
git add agent/graph.py
git commit -m "feat: add LangGraph state schema and graph definition"
```

---

### Task 6: Orchestrator Node

**Files:**
- Create: `agent/orchestrator.py`

- [ ] **Step 1: Write orchestrator.py**

```python
"""Orchestrator node: calls MCP retrieve tool, synthesises NL answer with LLM."""

import asyncio

from dotenv import load_dotenv
from fastmcp import Client
from langchain_openai import ChatOpenAI

load_dotenv()

ORCHESTRATOR_SYSTEM_PROMPT = (
    "You are a documentation assistant. "
    "Answer the user's question using ONLY the retrieved context provided. "
    "Do not add information not present in the context. "
    "Be concise and precise. Do not hedge or qualify your answer."
)


async def _call_mcp(query: str, k: int) -> str:
    async with Client("http://localhost:8765/sse") as client:
        result = await client.call_tool("retrieve", {"query": query, "k": k})
        return result[0].text


def orchestrator_node(state: dict) -> dict:
    # Step 1: Call MCP tool and capture R₀
    r0 = asyncio.run(_call_mcp(state["query"], k=3))

    # Step 2: Synthesise answer using LLM and capture R₁
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    messages = [
        {"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {state['query']}\n\nRetrieved context:\n{r0}"},
    ]
    response = llm.invoke(messages)
    r1 = response.content

    return {**state, "r0": r0, "r1": r1}
```

- [ ] **Step 2: Commit**

```bash
git add agent/orchestrator.py
git commit -m "feat: add orchestrator node — MCP retrieve + LLM synthesis"
```

---

### Task 7: Formatter Node

**Files:**
- Create: `agent/formatter.py`

- [ ] **Step 1: Write formatter.py**

```python
"""Formatter node: converts R₁ into structured JSON payload (R₂)."""

import json

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

FORMATTER_SYSTEM_PROMPT = (
    "You are a structured formatting assistant. "
    "Convert the provided answer into a JSON object with exactly these fields:\n"
    '- "summary": a 1-2 sentence summary of the answer (string)\n'
    '- "key_points": a list of 2-4 key points extracted from the answer (list of strings)\n'
    '- "code_example": a relevant code snippet if present in the answer, otherwise an empty string (string)\n'
    '- "source_ref": always set to "httpx documentation" (string)\n\n'
    "Return ONLY valid JSON. Do not add any fields not listed above."
)


def formatter_node(state: dict) -> dict:
    if state.get("error"):
        return state

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        model_kwargs={"response_format": {"type": "json_object"}},
    )
    messages = [
        {"role": "system", "content": FORMATTER_SYSTEM_PROMPT},
        {"role": "user", "content": state["r1"]},
    ]
    response = llm.invoke(messages)

    try:
        r2 = json.loads(response.content)
        for field in ["summary", "key_points", "code_example", "source_ref"]:
            if field not in r2:
                raise ValueError(f"Missing field: {field}")
        # code_example excluded from scoring — distorts BERTScore
        r2_text = r2["summary"] + " " + " ".join(r2["key_points"])
        return {**state, "r2": r2, "r2_text": r2_text}
    except (json.JSONDecodeError, ValueError) as e:
        return {**state, "r2": None, "r2_text": None, "error": f"Formatter JSON error: {e}"}
```

- [ ] **Step 2: Commit**

```bash
git add agent/formatter.py
git commit -m "feat: add formatter node — R₁ to JSON R₂ conversion"
```

---

### Task 8: BERTScore Scoring Module

**Files:**
- Create: `eval/score.py`
- Create: `tests/test_score.py`

- [ ] **Step 1: Write the test file**

```python
"""Tests for eval/score.py — validates scoring logic and delta computation."""

from eval.score import compute_bertscore, score_all


def test_compute_bertscore_returns_correct_structure():
    candidates = ["The cat sat on the mat."]
    references = ["The cat sat on the mat."]
    results = compute_bertscore(candidates, references)
    assert len(results) == 1
    assert set(results[0].keys()) == {"p", "r", "f1"}
    assert 0.0 <= results[0]["f1"] <= 1.0
    assert 0.0 <= results[0]["p"] <= 1.0
    assert 0.0 <= results[0]["r"] <= 1.0


def test_compute_bertscore_identical_texts_high_score():
    text = "httpx raises TimeoutException on timeout."
    results = compute_bertscore([text], [text])
    assert results[0]["f1"] > 0.99


def test_score_all_handles_none_values():
    r0_list = ["some text", None]
    r1_list = ["some answer", "another answer"]
    r2_list = ["formatted text", "formatted answer"]
    references = ["reference one", "reference two"]
    results = score_all(r0_list, r1_list, r2_list, references)
    assert len(results) == 2
    assert results[0]["f1_r0"] is not None
    assert results[0]["delta1_f1"] is not None
    assert results[1]["f1_r0"] is None
    assert results[1]["f1_r1"] is not None
    assert results[1]["delta1_f1"] is None  # can't compute delta if r0 is None


def test_score_all_computes_deltas():
    r0 = ["Pass a timeout parameter to the request method or Client constructor."]
    r1 = ["Use httpx.get(url, timeout=5.0) to set a timeout."]
    r2 = ["Set timeout with httpx.get timeout parameter."]
    ref = ["Pass a timeout parameter to the request method or Client constructor. httpx.get(url, timeout=5.0) sets a 5-second timeout."]
    results = score_all(r0, r1, r2, ref)
    r = results[0]
    assert r["delta1_f1"] is not None
    assert r["delta2_f1"] is not None
    assert abs(r["delta1_f1"] - (r["f1_r1"] - r["f1_r0"])) < 0.0001
    assert abs(r["delta2_f1"] - (r["f1_r2"] - r["f1_r1"])) < 0.0001
    assert abs(r["delta1_recall"] - (r["r_r1"] - r["r_r0"])) < 0.0001
    assert abs(r["delta2_recall"] - (r["r_r2"] - r["r_r1"])) < 0.0001
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
cd /Users/furkan/mcppaper/fidelity-agentic-stack
python -m pytest tests/test_score.py -v
```

Expected: All 4 tests FAIL with `ModuleNotFoundError: No module named 'eval.score'` (module not yet written).

- [ ] **Step 3: Write score.py**

```python
"""BERTScore computation for R₀, R₁, R₂ against reference answers."""

from typing import List, Optional

from bert_score import score as bert_score


def compute_bertscore(candidates: List[str], references: List[str]) -> List[dict]:
    """Compute BERTScore P, R, F1 for (candidate, reference) pairs.

    Uses roberta-large backbone, pinned via bert-score==0.3.13.
    """
    P, R, F1 = bert_score(
        candidates,
        references,
        model_type="roberta-large",
        lang="en",
        verbose=False,
        device="cpu",
    )
    return [
        {"p": round(p, 4), "r": round(r, 4), "f1": round(f, 4)}
        for p, r, f in zip(P.tolist(), R.tolist(), F1.tolist())
    ]


def score_all(
    r0_list: List[Optional[str]],
    r1_list: List[Optional[str]],
    r2_text_list: List[Optional[str]],
    reference_list: List[str],
) -> List[dict]:
    """Score R₀, R₁, R₂ for all queries in batched calls per layer.

    Returns list of dicts with p/r/f1 for each layer, plus F1 and Recall deltas.
    Handles None values (failed queries) by returning None for those score fields.
    """
    empty = {
        "p_r0": None, "r_r0": None, "f1_r0": None,
        "p_r1": None, "r_r1": None, "f1_r1": None,
        "p_r2": None, "r_r2": None, "f1_r2": None,
        "delta1_f1": None, "delta2_f1": None,
        "delta1_recall": None, "delta2_recall": None,
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

    for r in results:
        if r["f1_r0"] is not None and r["f1_r1"] is not None:
            r["delta1_f1"] = round(r["f1_r1"] - r["f1_r0"], 4)
            r["delta1_recall"] = round(r["r_r1"] - r["r_r0"], 4)
        if r["f1_r1"] is not None and r["f1_r2"] is not None:
            r["delta2_f1"] = round(r["f1_r2"] - r["f1_r1"], 4)
            r["delta2_recall"] = round(r["r_r2"] - r["r_r1"], 4)

    return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
python -m pytest tests/test_score.py -v
```

Expected: All 4 tests PASS. Note: first run downloads the `roberta-large` model (~1.4 GB), so it may be slow.

- [ ] **Step 5: Commit**

```bash
git add eval/score.py tests/test_score.py
git commit -m "feat: add BERTScore scoring module with tests"
```

---

### Task 9: Evaluation Runner

**Files:**
- Create: `eval/run_eval.py`

- [ ] **Step 1: Write run_eval.py**

```python
"""Evaluation runner. Manages MCP server subprocess, runs all 15 queries,
captures R₀/R₁/R₂, scores with BERTScore, writes results."""

import csv
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx as _httpx
from dotenv import load_dotenv

load_dotenv()

from agent.graph import build_graph, PipelineState
from eval.score import score_all

RESULTS_DIR = Path("eval/results")
PAPER_RESULTS_DIR = Path("paper_results")


def start_mcp_server() -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, "-m", "mcp_server.server"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
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

    test_set = json.loads(Path("data/test_set.json").read_text())
    graph = build_graph()
    mcp_proc = start_mcp_server()

    raw_logs = []
    try:
        for item in test_set:
            print(f"Running {item['id']}: {item['query'][:60]}...")
            try:
                initial_state: PipelineState = {
                    "query": item["query"],
                    "r0": None,
                    "r1": None,
                    "r2": None,
                    "r2_text": None,
                    "error": None,
                }
                result = graph.invoke(initial_state)
                raw_logs.append({
                    "id": item["id"],
                    "query": item["query"],
                    "r0": result["r0"],
                    "r1": result["r1"],
                    "r2": result["r2"],
                    "r2_text": result["r2_text"],
                    "error": result["error"],
                })
            except Exception as e:
                print(f"  ERROR: {e}")
                raw_logs.append({
                    "id": item["id"],
                    "query": item["query"],
                    "r0": None,
                    "r1": None,
                    "r2": None,
                    "r2_text": None,
                    "error": str(e),
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
        "delta1_recall", "delta2_recall",
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

    # Print summary
    _print_summary(scores)

    print("\n=== EVALUATION COMPLETE ===")
    print(f"Raw logs: {RESULTS_DIR}/raw_logs.json")
    print(f"Scores:   {RESULTS_DIR}/scores.csv")
    print(f"Table 1:  {PAPER_RESULTS_DIR}/table1.md")


def _write_paper_table(test_set, scores, path: Path):
    fmt = lambda v: f"{v:.4f}" if v is not None else "\u2014"
    valid_scores = [s for s in scores if s["f1_r0"] is not None]
    def mean(k): return round(sum(s[k] for s in valid_scores if s[k] is not None) / len(valid_scores), 4)

    # Table 1a: F1 per layer
    lines = [
        "## Table 1a. BERTScore F1 per layer",
        "",
        "| Query ID | F1(R\u2080) | F1(R\u2081) | F1(R\u2082) | \u0394\u2081 | \u0394\u2082 |",
        "|---|---|---|---|---|---|",
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
        "| Query ID | P(R\u2080) | R(R\u2080) | P(R\u2081) | R(R\u2081) | P(R\u2082) | R(R\u2082) | \u0394\u2081\u1d3f | \u0394\u2082\u1d3f |",
        "|---|---|---|---|---|---|---|---|---|",
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


def _print_summary(scores):
    valid = [s for s in scores if s["f1_r0"] is not None]
    if not valid:
        print("No valid scores to summarise.")
        return

    def _mean(k):
        vals = [s[k] for s in valid if s[k] is not None]
        return sum(vals) / len(vals) if vals else 0.0

    print("\n=== SUMMARY ===")
    for layer in ["r0", "r1", "r2"]:
        print(f"  {layer.upper()}: Mean F1={_mean(f'f1_{layer}'):.4f}  P={_mean(f'p_{layer}'):.4f}  R={_mean(f'r_{layer}'):.4f}")

    d1_f1 = _mean("delta1_f1")
    d1_r = _mean("delta1_recall")
    d2_f1 = _mean("delta2_f1")
    d2_r = _mean("delta2_recall")

    label_f1 = lambda d: "IMPROVES" if d > 0 else "DEGRADES"
    label_recall = lambda d: "Recall MAINTAINED (condensation only)" if d > -0.01 else "Recall DROPPED (information loss)"

    print(f"\n  \u0394\u2081 F1  = {d1_f1:+.4f} — A2A synthesis {label_f1(d1_f1)} F1")
    print(f"  \u0394\u2081 Recall = {d1_r:+.4f} — {label_recall(d1_r)}")
    print(f"  \u0394\u2082 F1  = {d2_f1:+.4f} — A2UI formatting {label_f1(d2_f1)} F1")
    print(f"  \u0394\u2082 Recall = {d2_r:+.4f} — {label_recall(d2_r)}")

    abs_d1 = abs(d1_f1)
    abs_d2 = abs(d2_f1)
    print(f"\n  Largest F1 delta: {'Δ₁ (A2A)' if abs_d1 >= abs_d2 else 'Δ₂ (A2UI)'} at {max(abs_d1, abs_d2):.4f}")
    abs_d1r = abs(d1_r)
    abs_d2r = abs(d2_r)
    print(f"  Largest Recall delta: {'Δ₁ (A2A)' if abs_d1r >= abs_d2r else 'Δ₂ (A2UI)'} at {max(abs_d1r, abs_d2r):.4f}")


if __name__ == "__main__":
    run()
```

- [ ] **Step 2: Commit**

```bash
git add eval/run_eval.py
git commit -m "feat: add evaluation runner with MCP server management and paper table output"
```

---

### Task 10: Streamlit UI

**Files:**
- Create: `ui/app.py`

- [ ] **Step 1: Write app.py**

```python
"""Streamlit UI for manual query inspection. Not part of the automated eval loop."""

import asyncio
import json
import subprocess
import sys
import time

import httpx
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


def _ensure_mcp_server():
    """Start MCP server if not already running."""
    if "mcp_proc" not in st.session_state or st.session_state.mcp_proc.poll() is not None:
        proc = subprocess.Popen(
            [sys.executable, "-m", "mcp_server.server"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for _ in range(30):
            try:
                httpx.get("http://localhost:8765/health", timeout=0.5)
                st.session_state.mcp_proc = proc
                return
            except Exception:
                time.sleep(0.5)
        proc.terminate()
        st.error("Failed to start MCP server.")
        st.stop()


st.set_page_config(page_title="Fidelity Pipeline Inspector", layout="wide")
st.title("MCP → A2A → A2UI Pipeline Inspector")

query = st.text_input("Enter a query about httpx:", placeholder="How do I configure a timeout for an httpx request?")

if st.button("Run Pipeline") and query:
    _ensure_mcp_server()

    from agent.graph import build_graph

    graph = build_graph()
    initial_state = {
        "query": query,
        "r0": None,
        "r1": None,
        "r2": None,
        "r2_text": None,
        "error": None,
    }

    with st.spinner("Running pipeline..."):
        result = graph.invoke(initial_state)

    if result.get("error"):
        st.error(f"Pipeline error: {result['error']}")
    else:
        st.subheader("R₂ — Structured Output")
        r2 = result["r2"]
        st.info(r2["summary"])
        st.markdown("\n".join(f"- {pt}" for pt in r2["key_points"]))
        if r2["code_example"]:
            st.code(r2["code_example"], language="python")
        st.caption(r2["source_ref"])

    with st.expander("Raw intermediates"):
        st.subheader("R₀ — Raw MCP chunks")
        st.text(result.get("r0", ""))
        st.subheader("R₁ — Synthesised answer")
        st.text(result.get("r1", ""))
        st.subheader("R₂ — Full JSON")
        st.json(result.get("r2", {}))
```

- [ ] **Step 2: Commit**

```bash
git add ui/app.py
git commit -m "feat: add Streamlit UI for manual pipeline inspection"
```

---

### Task 11: End-to-End Integration Run

**Files:** None (verification only)

- [ ] **Step 1: Run the full evaluation**

```bash
cd /Users/furkan/mcppaper/fidelity-agentic-stack
source .venv/bin/activate
python eval/run_eval.py
```

Expected: All 15 queries run. Output shows per-query progress, summary statistics, and file paths.

- [ ] **Step 2: Verify raw_logs.json**

```bash
python -c "
import json
logs = json.loads(open('eval/results/raw_logs.json').read())
print(f'Total entries: {len(logs)}')
ok = sum(1 for l in logs if l['r0'] and l['r1'] and l['r2'] and l['r2_text'])
err = sum(1 for l in logs if l['error'])
print(f'Successful: {ok}, Errors: {err}')
"
```

Expected: Total entries: 15, Successful: 15, Errors: 0

- [ ] **Step 3: Verify scores.csv**

```bash
python -c "
import csv
with open('eval/results/scores.csv') as f:
    rows = list(csv.reader(f))
print(f'Header + {len(rows)-1} rows (including Mean)')
print('Header:', rows[0])
print('Mean row:', rows[-1])
"
```

Expected: 17 rows (1 header + 15 data + 1 mean). All score cells populated. F1 values in 0.80–0.97 range.

- [ ] **Step 4: Verify paper_results/table1.md**

```bash
cat paper_results/table1.md
```

Expected: Two markdown tables (1a and 1b) with 4-decimal-place scores.

- [ ] **Step 5: Smoke-test Streamlit UI**

```bash
streamlit run ui/app.py
```

Expected: Browser opens, text input works, pipeline produces structured card output. Kill with Ctrl+C after verifying.

- [ ] **Step 6: Fix any issues found**

If any verification step fails, debug and fix. Re-run `python eval/run_eval.py` to confirm.

- [ ] **Step 7: Final commit**

```bash
git add -A
git commit -m "feat: complete fidelity-agentic-stack — all 15 queries scored, Table 1 generated"
```

---

### Task 12: README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write README.md**

```markdown
# Fidelity Agentic Stack

Layer-wise fidelity evaluation of an MCP → A2A → A2UI pipeline, measuring BERTScore at each intermediate against ground-truth httpx documentation answers.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # set OPENAI_API_KEY

python scripts/fetch_docs.py     # download httpx docs
python mcp_server/index.py       # build FAISS index
python eval/run_eval.py          # run full evaluation
```

Results are written to `eval/results/` and `paper_results/table1.md`.

## Manual Inspection

```bash
streamlit run ui/app.py
```
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README"
```
