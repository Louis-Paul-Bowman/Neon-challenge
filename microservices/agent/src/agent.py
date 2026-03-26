import json
import os

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://llm:11434")
MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")

llm = ChatOllama(
    base_url=LLM_BASE_URL,
    model=MODEL,
    format="json",
)

VESSEL_CODE = "32ebf047628f89ab"

SYSTEM_PROMPT = """You are an AI agent responding to mission control (NEON).
Every response MUST be a single JSON object with a "type" field. No other text outside the JSON.
Valid response formats:
- {"type": "enter_digits", "digits": "<string>"}  — use for numeric/keypad responses
- {"type": "speak_text", "text": "<string>"}       — use for voice responses, max 256 characters"""

# Task A: Signal handshake — always the first checkpoint.
# Neon asks us to respond on a frequency or enter the vessel authorization code.
# This is deterministic and must never be delegated to the LLM.
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


def process_prompt(prompt: str) -> dict:
    if _is_handshake(prompt):
        return {"type": "enter_digits", "digits": VESSEL_CODE}

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])
    return json.loads(response.content)
