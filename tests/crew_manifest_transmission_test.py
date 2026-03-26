import json
from requests import post


def crew_manifest_transmission_test(
    prompt: str, expected_result: str, min_chars: int = 1, max_chars: int = 256
) -> bool:

    if min_chars == max_chars:
        prompt += (
            f" Your entire JSON response must be exactly {min_chars} characters long."
        )
    else:
        prompt += f" Your entire JSON response must be between {min_chars} and {max_chars} characters long."

    req = post("http://localhost:3000/process", json={"prompt": prompt})

    if not req.ok:
        print(req.reason)
        try:
            print(req.json())
        finally:
            return False

    try:
        resp = req.json()
        print(resp)
        result_text = resp["text"]
        type_ = resp["type"]
        # Re-serialise exactly as the agent would produce it to measure true length
        response_json = json.dumps(resp, separators=(", ", ": "))
        response_len = len(response_json)
        length_ok = min_chars <= response_len <= max_chars
        return (
            type_ == "speak_text"
            and expected_result.lower() in result_text.lower()
            and length_ok
        )
    except:
        return False


if __name__ == "__main__":
    passed = all(
        (
            crew_manifest_transmission_test(
                prompt="Transmit where the crew member went to school.",
                expected_result="McGill University",
                min_chars=0,
                max_chars=126,
            ),
            crew_manifest_transmission_test(
                prompt="Speak which company the crew member co-founded.",
                expected_result="Arise Industries",
                min_chars=50,
                max_chars=50,
            ),
        )
    )
    print(
        "[{}] Crew manifest transmission test.".format("PASS" if passed else "FAILED")
    )
