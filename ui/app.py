"""Streamlit UI for manual query inspection. Not part of the automated eval loop."""

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
st.title("MCP \u2192 A2A \u2192 A2UI Pipeline Inspector")

query = st.text_input("Enter a query about httpx:", placeholder="How do I set a timeout for an httpx request?")

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
        st.subheader("\u2082 \u2014 Structured Output")
        r2 = result["r2"]
        st.info(r2["summary"])
        st.markdown("\n".join(f"- {pt}" for pt in r2["key_points"]))
        if r2["code_example"]:
            st.code(r2["code_example"], language="python")
        st.caption(r2["source_ref"])

    with st.expander("Raw intermediates"):
        st.subheader("R\u2080 \u2014 Raw MCP chunks")
        st.text(result.get("r0", ""))
        st.subheader("R\u2081 \u2014 Synthesised answer")
        st.text(result.get("r1", ""))
        st.subheader("R\u2082 \u2014 Full JSON")
        st.json(result.get("r2", {}))
