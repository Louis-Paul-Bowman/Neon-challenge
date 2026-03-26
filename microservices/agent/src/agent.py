import json
import os

import requests as http
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://llm:11434")
MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")
BACKEND_URL = os.environ.get("BACKEND_URL", "http://backend:4000")

VESSEL_CODE = "32ebf047628f89ab"

SYSTEM_PROMPT = """You are an AI agent responding to mission control (NEON).
Every response MUST be a single JSON object with a "type" field. No other text outside the JSON.
Valid response formats:
- {"type": "enter_digits", "digits": "<string>"}  — use for numeric/keypad responses
- {"type": "speak_text", "text": "<string>"}       — use for voice responses, max 256 characters"""

# --- Tools -------------------------------------------------------------------

@tool
def eval_math_expression(expression: str) -> float:
    """Evaluate a mathematical expression containing numbers, +, -, *, /, %,
    parentheses, and Math.floor. Returns the numeric result."""
    resp = http.post(f"{BACKEND_URL}/eval", json={"expression": expression}, timeout=10)
    resp.raise_for_status()
    return resp.json()["result"]


TOOLS = [eval_math_expression]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}

# format="json" conflicts with Ollama's tool-calling API, so it is omitted here.
# The system prompt enforces JSON output for the final answer instead.
llm = ChatOllama(base_url=LLM_BASE_URL, model=MODEL).bind_tools(TOOLS)

# --- Handshake (Task A) ------------------------------------------------------

_HANDSHAKE_KEYWORDS = (
    "authorization code",
    "vessel",
    "handshake",
    "frequency",
    "credentials",
    "identification",
)


def _is_handshake(prompt: str) -> bool:
    lower = prompt.lower()
    return any(kw in lower for kw in _HANDSHAKE_KEYWORDS)

# --- Agent loop --------------------------------------------------------------

def process_prompt(prompt: str) -> dict:
    if _is_handshake(prompt):
        return {"type": "enter_digits", "digits": VESSEL_CODE}

    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]

    # Tool-calling loop: keep going until the LLM stops issuing tool calls.
    while True:
        response = llm.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            break

        for tc in response.tool_calls:
            result = TOOLS_BY_NAME[tc["name"]].invoke(tc["args"])
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    return json.loads(response.content)
