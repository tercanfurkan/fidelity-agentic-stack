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
