import os

import uvicorn
from fastapi import FastAPI
from agent import process_prompt

from schemas import PromptRequest

app = FastAPI()


@app.post("/process")
def process(req: PromptRequest):
    return process_prompt(req)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    model = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")
    print(f"Agent microservice listening on port {port}  model: {model}")
    uvicorn.run(app, host="0.0.0.0", port=port)
