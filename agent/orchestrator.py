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
