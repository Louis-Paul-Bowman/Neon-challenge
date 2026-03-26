from requests import post

NEON_CODE = "32ebf047628f89ab"


def handshake_test(prompt: str) -> bool:

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
        return type_ == "enter_digits" and result == NEON_CODE
    except:
        return False


if __name__ == "__main__":
    passed = handshake_test(prompt="Transmit Neon identification code.")
    print("[{}] Handshake test.".format("PASS" if passed else "FAILED"))
