# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an AI agent challenge project that connects to Neon via WebSocket and completes 5 distinct tasks. The agent is built as a 3-microservice system using LangChain (Node.js), with Docker orchestration.

## Architecture

Three microservices, each with its own Dockerfile, orchestrated by a top-level `docker-compose.yml`:

- **`microservices/agent/`** — LangChain-based AI agent. Handles WebSocket connection to Neon, input unscrambling, task routing, session memory, and response formatting.
- **`microservices/backend/`** — Express.js tool server. Exposes HTTP endpoints for math evaluation and Wikipedia lookups.
- **`microservices/llm/`** — Ollama LLM instance. Serves a lightweight model with tool-use and JSON output support. Fallback: OpenAI/Anthropic.

The agent calls backend tools over HTTP. The agent connects to Neon at `wss://neonhealth.software/agent-puzzle/challenge`.

## Communication Protocol

**Input** — scrambled JSON that must be reconstructed before processing:
```json
{"type":"challenge", "message": [{"word": "string", "timestamp": 0}]}
```
Reconstruct by sorting `message` array by ascending `timestamp`, then joining words with spaces.

**Output** — every response must be exactly one of these two JSON formats (no other text):
```json
{"type": "enter_digits", "digits": "<string>"}   // for numeric/keypad responses
{"type": "speak_text", "text": "<string>"}        // for voice responses, max 256 chars
```

## The 5 Tasks (Checkpoints)

Tasks arrive in this fixed order: **A → (B/C/D in random order) → E**

| Task | Description | Response |
|------|-------------|----------|
| A | Signal handshake — always respond with `{"type":"enter_digits","digits":"32ebf047628f89ab"}` | `enter_digits` |
| B | Evaluate a JS math expression (may use `Math.Floor`, `+`, `-`, `*`, `/`, `%`, `(`, `)`, numbers). Append `#` if prompt says "followed by the pound key". | `enter_digits` |
| C | Wikipedia lookup — fetch `/page/summary/{title}`, split extract into words, return word at position x (index x-1). | `speak_text` |
| D | CV/bio questions — answer from a hardcoded plaintext CV constant. Respect exact/min/max length constraints (wrong length aborts checkpoint). | `speak_text` |
| E | Recall a specific word from a previous Task D response. Requires in-memory session storage keyed by `thread_id` (UUID4). | `speak_text` |

## Key Implementation Notes

- **Math eval security**: Validate input strictly — only allow digits, `Math.Floor`, and the permitted operators before evaluating. Prevents RCE.
- **Wikipedia API**: `GET https://en.wikipedia.org/api/rest_v1/page/summary/{title}` — use the `extract` field, split on whitespace.
- **Length enforcement for Task D**: The agent prompt + a post-processing validator must coerce responses to required length (truncate/pad).
- **Session memory**: LangChain `ConversationBufferMemory` (or equivalent) keyed by `thread_id` so Task E can recall Task D words.
- **LLM model selection**: Must support tool use and structured JSON output. If local Ollama model is insufficient, switch to OpenAI/Anthropic in the agent config.

## Development Order (from Plan.txt)

Follow this sequence when building out features:
1. LLM microservice + Dockerfile + docker-compose
2. Agent microservice skeleton + Task A
3. Backend math eval endpoint + Task B
4. Backend Wikipedia endpoint + Task C
5. CV constant + Task D (with length enforcement)
6. Session memory + Task E
7. Input unscrambling pre-processor
8. WebSocket connection to Neon + end-to-end debugging

Each step includes a corresponding test in `tests/`.
