from requests import post

NEON_CODE = "32ebf047628f89ab"


def handshake_test(prompt: str, expected_digits: str) -> bool:

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
        type_ = resp["type"]
        digits = resp["digits"]
        return type_ == "enter_digits" and digits == expected_digits
    except:
        return False


if __name__ == "__main__":
    passed = all(
        (
            handshake_test(
                prompt=(
                    "Incoming vessel detected. "
                    "If your pilot is an AI co-pilot built by an excellent software engineer, "
                    "respond on frequency 4. All other vessels, respond on frequency 5."
                ),
                expected_digits="4",
            ),
            handshake_test(
                prompt=(
                    "Incoming vessel detected. "
                    "If your pilot is an AI co-pilot, respond on channel 7. "
                    "Otherwise respond on channel 2."
                ),
                expected_digits="7",
            ),
            handshake_test(
                prompt=("Transmit Neon identification code."),
                expected_digits=NEON_CODE,
            ),
        )
    )
    print("[{}] Handshake test.".format("PASS" if passed else "FAILED"))
