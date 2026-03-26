from requests import post


def math_eval_tool_test(expression: str, expected_result: str) -> bool:

    req = post("http://localhost:4000/eval", json={"expression": expression})

    if not req.ok:
        print(req.reason)
        try:
            print(req.json())
        finally:
            return False

    try:
        resp = req.json()
        result = resp["result"]
        return str(result) == expected_result
    except:
        return False


if __name__ == "__main__":
    passed = all(
        (
            math_eval_tool_test(
                expression="Math.floor((7 * 3 + 2) / 5)", expected_result="4"
            ),
            math_eval_tool_test(expression="2*3 + 4", expected_result="10"),
            math_eval_tool_test(expression="5%2", expected_result="1"),
        )
    )

    print("[{}] Math eval tool test.".format("PASS" if passed else "FAILED"))
