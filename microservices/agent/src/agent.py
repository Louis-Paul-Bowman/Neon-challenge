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

SYSTEM_PROMPT = """You are an AI agent responding to mission control (NEON).
Every response MUST be a single JSON object with a "type" field. No other text outside the JSON.
Valid response formats:
- {"type": "enter_digits", "digits": "<string>"}  — use for numeric/keypad responses
- {"type": "speak_text", "text": "<string>"}       — use for voice responses, max 256 characters"""


def process_prompt(prompt: str) -> dict:
    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])
    return json.loads(response.content)
