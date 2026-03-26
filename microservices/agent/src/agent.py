import json
import os
import logging

from requests import post
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

logging.basicConfig(level=logging.DEBUG, handlers=[logging.StreamHandler()])
logger = logging.getLogger("Agent")

MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")
BACKEND_URL = os.environ.get("BACKEND_URL", "http://backend:4000")

VESSEL_CODE = "32ebf047628f89ab"

# Used by the tool-calling loop — no format constraint (conflicts with tool use).
TOOL_SYSTEM_PROMPT = """You are an AI agent responding to mission control (NEON).
You have access to tools. Use them when needed, then stop calling tools once you have the answer."""

# Used by the formatting step — produces the final Neon-protocol JSON.
FORMAT_SYSTEM_PROMPT = """You are formatting a response for mission control (NEON).
Respond with ONLY a single JSON object — no other text before or after it.

Rules:
- Math / calculation results → {"type": "enter_digits", "digits": "<result as string>"}
  If the original prompt said "followed by the pound key", append # to the digits value.
- All other responses → {"type": "speak_text", "text": "<answer>"}  (max 256 characters)"""

# --- Tools -------------------------------------------------------------------


@tool
def eval_math_expression(expression: str) -> float:
    """Evaluate a mathematical expression. Pass the expression EXACTLY as given
    in the prompt, including any Math.floor(...) wrapping — do not simplify or
    strip any part of it. Supports numbers, +, -, *, /, %, (), and Math.floor."""
    logger.debug("eval_math_expression called with: %s", expression)
    resp = post(f"{BACKEND_URL}/eval", json={"expression": expression}, timeout=10)
    resp.raise_for_status()
    return resp.json()["result"]


TOOLS = [eval_math_expression]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}

llm = ChatAnthropic(model=MODEL).bind_tools(TOOLS)
formatter_llm = ChatAnthropic(model=MODEL)

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

    messages = [SystemMessage(content=TOOL_SYSTEM_PROMPT), HumanMessage(content=prompt)]

    # Tool-calling loop: keep going until the LLM stops issuing tool calls.
    while True:
        response = llm.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            break

        for tc in response.tool_calls:
            result = TOOLS_BY_NAME[tc["name"]].invoke(tc["args"])
            logger.debug("Tool %s returned: %s", tc["name"], result)
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    # Dedicated formatting step. Build a fresh human message with the original
    # prompt and any tool results — Claude requires the conversation to end with
    # a user message, so we cannot spread `messages` (ends with an AIMessage).
    tool_results = [m for m in messages if isinstance(m, ToolMessage)]
    context = f"Original prompt: {prompt}"
    if tool_results:
        results_str = "\n".join(f"- {m.content}" for m in tool_results)
        context += f"\n\nTool results:\n{results_str}"

    format_response = formatter_llm.invoke([
        SystemMessage(content=FORMAT_SYSTEM_PROMPT),
        HumanMessage(content=context),
    ])

    return json.loads(format_response.content)
