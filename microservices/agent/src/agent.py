import json
import os
import re
import uuid
import logging

from requests import post, get
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from Docs import PLAINTEXT_CV

logging.basicConfig(level=logging.DEBUG, handlers=[logging.StreamHandler()])
logger = logging.getLogger("Agent")

MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")
BACKEND_URL = os.environ.get("BACKEND_URL", "http://backend:4000")

VESSEL_CODE = "32ebf047628f89ab"

# --- Prompts -----------------------------------------------------------------

TOOL_SYSTEM_PROMPT = """You are an AI agent responding to mission control (NEON).
You have access to tools. Use them when needed, then stop calling tools once you have the answer."""

FORMAT_SYSTEM_PROMPT = """You are formatting a response for mission control (NEON).
Respond with ONLY a single JSON object — no other text before or after it.

Response format rules:
- Math / calculation results → {"type": "enter_digits", "digits": "<result as string>"}
  If the original prompt said "followed by the pound key", append # to the digits value.
- All other responses → {"type": "speak_text", "text": "<answer>"}

Length rules for speak_text (the prompt may specify constraints):
- "exactly N characters" → text must be exactly N characters
- "between X and Y characters" → text length must be in [X, Y]
- "at least N characters" → text length must be >= N
- "no more than N" / "at most N characters" → text length must be <= N
- Hard maximum: 256 characters regardless of other constraints
Craft your answer to naturally fit within the required length."""

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


@tool
def get_wikipedia_word(title: str, position: int) -> str:
    """Fetch the word at a given position (1-indexed) from a Wikipedia article summary.
    Use the article title exactly as mentioned in the prompt."""
    logger.debug(
        "get_wikipedia_word called with: title=%s position=%d", title, position
    )
    resp = get(
        f"{BACKEND_URL}/wiki", params={"title": title, "position": position}, timeout=10
    )
    resp.raise_for_status()
    return resp.json()["result"]


@tool
def get_cv() -> str:
    """Return the crew member's CV / biographical data. Use this to answer any
    questions about the crew member's background, experience, skills, or history."""
    logger.debug("get_cv called")
    return PLAINTEXT_CV


TOOLS = [eval_math_expression, get_wikipedia_word, get_cv]

# --- Agent -------------------------------------------------------------------

llm = ChatAnthropic(model=MODEL)
formatter_llm = ChatAnthropic(model=MODEL)

memory = MemorySaver()
agent = create_react_agent(llm, TOOLS, checkpointer=memory, prompt=TOOL_SYSTEM_PROMPT)

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


# --- Length coercion (Task D) ------------------------------------------------


def _parse_length_constraint(prompt: str) -> tuple[int | None, int | None]:
    """Return (min_len, max_len) parsed from the prompt. Either may be None."""
    lower = prompt.lower()
    exact = re.search(r"exactly\s+(\d+)\s+characters?", lower)
    if exact:
        n = int(exact.group(1))
        return n, n

    between = re.search(r"between\s+(\d+)\s+and\s+(\d+)\s+characters?", lower)
    if between:
        return int(between.group(1)), int(between.group(2))

    min_match = re.search(r"at\s+least\s+(\d+)\s+characters?", lower)
    max_match = re.search(
        r"(?:no\s+more\s+than|at\s+most)\s+(\d+)\s+characters?", lower
    )
    min_len = int(min_match.group(1)) if min_match else None
    max_len = int(max_match.group(1)) if max_match else None
    return min_len, max_len


def _coerce_length(text: str, min_len: int | None, max_len: int | None) -> str:
    """Truncate or space-pad text to satisfy length constraints."""
    hard_max = min(max_len, 256) if max_len is not None else 256
    if len(text) > hard_max:
        text = text[:hard_max]
    if min_len is not None and len(text) < min_len:
        text = text.ljust(min_len)
    return text


# --- Agent loop --------------------------------------------------------------


def process_prompt(prompt: str, thread_id: str | None = None) -> dict:
    if _is_handshake(prompt):
        return {"type": "enter_digits", "digits": VESSEL_CODE}

    # Each call without a thread_id gets an isolated session so it never
    # inherits history from other stateless calls.
    config = {"configurable": {"thread_id": thread_id or str(uuid.uuid4())}}

    state = agent.invoke({"messages": [HumanMessage(content=prompt)]}, config=config)

    # Extract tool results from the current turn (messages after the last
    # HumanMessage) to give the formatter accurate context.
    messages = state["messages"]
    last_human_idx = max(
        i for i, m in enumerate(messages) if isinstance(m, HumanMessage)
    )
    current_tool_results = [
        m for m in messages[last_human_idx:] if isinstance(m, ToolMessage)
    ]

    context = f"Original prompt: {prompt}"
    if current_tool_results:
        results_str = "\n".join(f"- {m.content}" for m in current_tool_results)
        context += f"\n\nTool results:\n{results_str}"

    format_response = formatter_llm.invoke(
        [
            SystemMessage(content=FORMAT_SYSTEM_PROMPT),
            HumanMessage(content=context),
        ],
        config=config,
    )

    result = json.loads(format_response.content)

    if result.get("type") == "speak_text":
        min_len, max_len = _parse_length_constraint(prompt)
        if min_len is not None or max_len is not None:
            result["text"] = _coerce_length(result["text"], min_len, max_len)
            logger.debug(
                "Coerced text length to %d (min=%s max=%s)",
                len(result["text"]),
                min_len,
                max_len,
            )

    # Store the final formatted answer back into the agent's thread so Task E
    # can recall words from previous Task D responses via the same thread_id.
    if thread_id and result.get("type") == "speak_text":
        agent.update_state(
            config,
            {"messages": [AIMessage(content=f"My response was: {result['text']}")]},
        )
        logger.debug("Stored formatted answer in thread %s", thread_id)

    return result
