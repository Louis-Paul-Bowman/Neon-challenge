import json
import os
import re
import uuid
import logging

from requests import post, get
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from schemas import PromptRequest
from Docs import PLAINTEXT_CV

logging.basicConfig(level=logging.DEBUG, handlers=[logging.StreamHandler()])
logger = logging.getLogger("Agent")

MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")
BACKEND_URL = os.environ.get("BACKEND_URL", "http://backend:4000")

VESSEL_CODE = "32ebf047628f89ab"

# --- System prompt -----------------------------------------------------------

SYSTEM_PROMPT = """You are an AI agent responding to mission control (NEON).

Use the available tools to answer each question, then reply with ONLY a single
JSON object — no other text before or after it.
Always respond with valid JSON only, do not include any explanation, reasoning, or text outside the JSON object.

Output format (pick one):
- Math / calculation result → {"type": "enter_digits", "digits": "<result as string>"}
  If the prompt says "followed by the pound key", append # to the digits value.
- All other answers → {"type": "speak_text", "text": "<answer>"}
Always respond with valid JSON only, do not include any explanation, reasoning, or text outside the JSON object.


Length rules for speak_text (obey any constraint stated in the prompt):
- "exactly N characters" → text must be exactly N characters
- "between X and Y characters" → text length must be in [X, Y]
- "at least N characters" → text length must be >= N
- "no more than N" / "at most N characters" → text length must be <= N
- Hard maximum: 256 characters regardless of other constraints
Craft your answer to naturally fit within the required length.

Memory: your previous JSON responses are visible in the conversation history.
If asked to recall a specific word from a previous answer, look at the "text"
field of the relevant earlier {"type": "speak_text", ...} message.
Always respond with valid JSON only, do not include any explanation, reasoning, or text outside the JSON object."""


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

memory = MemorySaver()
agent = create_react_agent(
    ChatAnthropic(model=MODEL),
    TOOLS,
    checkpointer=memory,
    prompt=SYSTEM_PROMPT,
)

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


# --- Input unscrambling ------------------------------------------------------


def decode_message(request: PromptRequest) -> str:
    if request.type == "challenge" and isinstance(request.prompt, list):
        words = sorted(request.prompt, key=lambda w: w.timestamp)
        return " ".join(w.word for w in words)
    return request.prompt


# --- JSON extraction / retry -------------------------------------------------

_REFORMAT_MSG = (
    "Your previous response was not valid JSON. "
    "Reply with ONLY a single JSON object in one of these two forms — no other text:\n"
    '  {"type": "enter_digits", "digits": "<string>"}\n'
    '  {"type": "speak_text", "text": "<string>"}'
)

_MAX_RETRIES = 2


def _extract_json(text: str) -> dict:
    """Parse JSON from text, tolerating a leading/trailing code fence."""
    text = text.strip()
    # Strip optional ```json ... ``` fences
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    return json.loads(text)


def _parse_json_with_retry(content: str, config: dict) -> dict:
    """Try to parse JSON from content; on failure nudge the agent and retry."""
    for attempt in range(_MAX_RETRIES + 1):
        try:
            return _extract_json(content)
        except (json.JSONDecodeError, ValueError):
            if attempt == _MAX_RETRIES:
                logger.error("All %d JSON parse attempts failed. Last response: %s", _MAX_RETRIES + 1, content)
                raise
            logger.warning("Attempt %d: invalid JSON, asking agent to reformat. Response was: %s", attempt + 1, content)
            state = agent.invoke({"messages": [HumanMessage(content=_REFORMAT_MSG)]}, config=config)
            content = state["messages"][-1].content
            logger.debug("Retry %d agent response: %s", attempt + 1, content)


# --- Agent loop --------------------------------------------------------------


def process_prompt(request: PromptRequest) -> dict:
    thread_id = request.thread_id
    prompt = decode_message(request)
    logger.debug("Unscrambled prompt: %s", prompt)

    if _is_handshake(prompt):
        return {"type": "enter_digits", "digits": VESSEL_CODE}

    # Each call without a thread_id gets an isolated session so it never
    # inherits history from other stateless calls.
    config = {"configurable": {"thread_id": thread_id or str(uuid.uuid4())}}

    state = agent.invoke({"messages": [HumanMessage(content=prompt)]}, config=config)

    # The agent's final AIMessage is already persisted in MemorySaver under
    # the thread_id, so Task E can recall it from conversation history.
    final_content = state["messages"][-1].content
    logger.debug("Agent raw response: %s", final_content)

    result = _parse_json_with_retry(final_content, config)

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

    return result
