from requests import get


def wikipedia_tool_test(title: str, position: int, expected_result: str) -> bool:

    req = get(
        "http://localhost:4000/wiki", params={"title": title, "position": position}
    )

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
            wikipedia_tool_test(title="Saturn", position=8, expected_result="Sun"),
            wikipedia_tool_test(
                title="Large language model", position=10, expected_result="trained"
            ),
            wikipedia_tool_test(title="Diet Coke", position=17, expected_result="soda"),
        )
    )

    print("[{}] Wikipedia tool test.".format("PASS" if passed else "FAILED"))
