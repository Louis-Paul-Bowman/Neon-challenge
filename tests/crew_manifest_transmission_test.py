from requests import post


def crew_manifest_transmission_test(
    prompt: str, expected_result: str, min_chars: int = 1, max_chars: int = 256
) -> bool:

    if min_chars == max_chars:
        prompt += (
            f" Use exactly {min_chars} characters in the result part of your response."
        )
    else:
        prompt += f" Use between {min_chars} and {max_chars} characters in the result part of your response."

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
        result = resp["text"]
        type_ = resp["type"]
        return (
            type_ == "speak_text"
            and expected_result.lower() in result.lower()
            and (len(result) >= min_chars and len(result) <= max_chars)
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
                min_chars=20,
                max_chars=20,
            ),
        )
    )
    print(
        "[{}] Crew manifest transmission test.".format("PASS" if passed else "FAILED")
    )
