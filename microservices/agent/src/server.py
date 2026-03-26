import os

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from agent import process_prompt

app = FastAPI()


class PromptRequest(BaseModel):
    prompt: str


@app.post("/process")
def process(req: PromptRequest):
    return process_prompt(req.prompt)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    llm_url = os.environ.get("LLM_BASE_URL", "http://llm:11434")
    model = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
    print(f"LLM: {llm_url}  model: {model}")
    uvicorn.run(app, host="0.0.0.0", port=port)
