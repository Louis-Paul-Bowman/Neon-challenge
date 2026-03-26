from schemas import PromptRequest


def decode_message(request: PromptRequest) -> str:
    if request.type == "challenge" and isinstance(request.prompt, list):
        words = sorted(request.prompt, key=lambda w: w.timestamp)
        return " ".join(w.word for w in words)
    return request.prompt
