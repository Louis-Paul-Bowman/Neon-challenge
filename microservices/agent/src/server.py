import os
import threading

import uvicorn
from fastapi import FastAPI
from agent import process_prompt
from neon_client import run as run_neon_client

from schemas import PromptRequest

app = FastAPI()


@app.post("/process")
def process(req: PromptRequest):
    return process_prompt(req)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    model = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")

    # When NEON_CONNECT=1 the WebSocket client runs alongside the HTTP server.
    if os.environ.get("NEON_CONNECT", "0") == "1":
        max_turns = int(os.environ.get("NEON_MAX_TURNS", "20"))
        print(f"Starting Neon WebSocket client  max_turns={max_turns}")
        t = threading.Thread(target=run_neon_client, kwargs={"max_turns": max_turns}, daemon=True)
        t.start()

    print(f"Agent microservice listening on port {port}  model: {model}")
    uvicorn.run(app, host="0.0.0.0", port=port)
