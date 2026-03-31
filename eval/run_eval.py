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

    print(f"\n  \u0394\u2081 F1  = {d1_f1:+.4f} \u2014 A2A synthesis {label_f1(d1_f1)} F1")
    print(f"  \u0394\u2081 Recall = {d1_r:+.4f} \u2014 {label_recall(d1_r)}")
    print(f"  \u0394\u2082 F1  = {d2_f1:+.4f} \u2014 A2UI formatting {label_f1(d2_f1)} F1")
    print(f"  \u0394\u2082 Recall = {d2_r:+.4f} \u2014 {label_recall(d2_r)}")

    abs_d1 = abs(d1_f1)
    abs_d2 = abs(d2_f1)
    print(f"\n  Largest F1 delta: {'\u0394\u2081 (A2A)' if abs_d1 >= abs_d2 else '\u0394\u2082 (A2UI)'} at {max(abs_d1, abs_d2):.4f}")
    abs_d1r = abs(d1_r)
    abs_d2r = abs(d2_r)
    print(f"  Largest Recall delta: {'\u0394\u2081 (A2A)' if abs_d1r >= abs_d2r else '\u0394\u2082 (A2UI)'} at {max(abs_d1r, abs_d2r):.4f}")


if __name__ == "__main__":
    run()
