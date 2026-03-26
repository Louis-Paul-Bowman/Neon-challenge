"""WebSocket client that connects our agent to the Neon challenge endpoint."""

import json
import logging
import os
import uuid

import websocket

from agent import process_prompt
from schemas import PromptRequest

logger = logging.getLogger("NeonClient")

NEON_WS_URL = os.environ.get(
    "NEON_WS_URL", "wss://neonhealth.software/agent-puzzle/challenge"
)
DEFAULT_MAX_TURNS = int(os.environ.get("NEON_MAX_TURNS", "20"))


def run(max_turns: int = DEFAULT_MAX_TURNS) -> None:
    """Connect to Neon, process challenges, and send responses until done or max_turns reached."""

    thread_id = str(uuid.uuid4())
    turns = 0

    def on_open(ws: websocket.WebSocketApp) -> None:
        logger.info("Connected to Neon  thread_id=%s", thread_id)

    def on_message(ws: websocket.WebSocketApp, raw: str) -> None:
        nonlocal turns
        turns += 1
        logger.debug("Turn %d/%d — received: %s", turns, max_turns, raw)

        if turns > max_turns:
            logger.error("Max turns (%d) exceeded — closing connection.", max_turns)
            ws.close()
            return

        try:
            parsed = json.loads(raw)

            request = PromptRequest(**parsed, thread_id=thread_id)
            response = process_prompt(request)
        except Exception:
            logger.exception("Failed to process message: %s", raw)
            ws.close()
            return

        payload = json.dumps(response)
        logger.info("Turn %d — sending: %s", turns, payload)
        ws.send(payload)

    def on_error(ws: websocket.WebSocketApp, error: Exception) -> None:
        logger.error("WebSocket error: %s", error)

    def on_close(
        ws: websocket.WebSocketApp, close_status_code: int, close_msg: str
    ) -> None:
        logger.info(
            "Connection closed  status=%s  msg=%s  turns_used=%d",
            close_status_code,
            close_msg,
            turns,
        )

    ws_app = websocket.WebSocketApp(
        NEON_WS_URL,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )
    ws_app.run_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, handlers=[logging.StreamHandler()])
    run()
