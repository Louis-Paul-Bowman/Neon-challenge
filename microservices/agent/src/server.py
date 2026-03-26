import os

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from agent import process_prompt

app = FastAPI()


class PromptRequest(BaseModel):
    prompt: str
    thread_id: str | None = Field(default=None, description="UUID4 session ID for recall across turns")


@app.post("/process")
def process(req: PromptRequest):
    return process_prompt(req.prompt, thread_id=req.thread_id)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    model = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")
    print(f"Agent microservice listening on port {port}  model: {model}")
    uvicorn.run(app, host="0.0.0.0", port=port)
