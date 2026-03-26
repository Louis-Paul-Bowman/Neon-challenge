from requests import post
import random


def agent_unscramble_test() -> bool:

    original_prompt = "Speak the word Password directly."
    parts = [
        {"word": word, "timestamp": i} for i, word in enumerate(original_prompt.split())
    ]
    random.shuffle(parts)

    challenge_req = {"type": "challenge", "prompt": parts}
    print(challenge_req)

    req = post("http://localhost:3000/process", json=challenge_req)

    if not req.ok:
        print(req.json())
        return False

    try:
        resp = req.json()
        print(resp)
        result = resp["text"]
        type_ = resp["type"]
        return type_ == "speak_text" and str(result) == "Password"
    except:
        return False


if __name__ == "__main__":
    passed = agent_unscramble_test()
    print("[{}] Transmission unscramble test.".format("PASS" if passed else "FAILED"))
