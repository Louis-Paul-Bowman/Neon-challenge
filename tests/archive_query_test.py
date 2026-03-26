from requests import post

NEON_CODE = "32ebf047628f89ab"


def archive_query_test(prompt: str, expected_result: str) -> bool:

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
        return type_ == "speak_text" and result == expected_result
    except:
        return False


if __name__ == "__main__":
    passed = all(
        (
            archive_query_test(
                prompt="Speak the 8th word in the knowledge entry for Saturn",
                expected_result="Sun",
            ),
            archive_query_test(
                prompt="Speak the word in position 10 for the 'Large language model' article.",
                expected_result="trained",
            ),
            archive_query_test(
                prompt="Transmit the word in position 17 for the 'Diet Coke' article.",
                expected_result="soda",
            ),
        )
    )
    print("[{}] Archive query test.".format("PASS" if passed else "FAILED"))
