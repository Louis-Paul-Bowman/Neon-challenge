from requests import post


def computational_assessment_test(prompt: str, expected_result: str) -> bool:

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
        result = resp["digits"]
        type_ = resp["type"]
        return type_ == "enter_digits" and result == expected_result
    except:
        return False


if __name__ == "__main__":
    passed = all(
        (
            computational_assessment_test(
                prompt="Press Math.floor((7 * 3 + 2) / 5) .",
                expected_result="4",
            ),
            computational_assessment_test(
                prompt="Enter 3%2 .",
                expected_result="1",
            ),
            computational_assessment_test(
                prompt="Respond on Math.floor((7 * 3 + 2) / 5) followed by the # character with no space.",
                expected_result="4#",
            ),
        )
    )
    print("[{}] Computational assessment test.".format("PASS" if passed else "FAILED"))
