from requests import post
from uuid import uuid4

from crew_manifest_transmission_test import crew_manifest_transmission_test


def transmission_verification_test() -> bool:
    """
    Depends on crew manifest test.
    """

    pre_test_passed = (
        crew_manifest_transmission_test(
            prompt="Speak which company the crew member co-founded.",
            expected_result="Arise Industries",
        ),
    )

    if not pre_test_passed:
        return False

    thread_id = str(uuid4())

    prompt = "Speak which company the crew member co-founded."

    req = post(
        "http://localhost:3000/process", json={"prompt": prompt, "thread_id": thread_id}
    )

    if not req.ok:
        print(req.reason)
        try:
            print(req.json())
        finally:
            return False

    try:
        resp = req.json()
        print(resp)
        result_1 = resp["text"]
    except:
        return False

    prompt = "Speak the name of the company we last talked about."

    req = post(
        "http://localhost:3000/process", json={"prompt": prompt, "thread_id": thread_id}
    )

    if not req.ok:
        print(req.reason)
        try:
            print(req.json())
        finally:
            return False

    try:
        resp = req.json()
        print(resp)
        result_2 = resp["text"]
        type_ = resp["type"]
        return type_ == "speak_text" and result_1.lower() in result_2.lower()
    except:
        return False


if __name__ == "__main__":
    passed = transmission_verification_test()
    print("[{}] Transmission verification test.".format("PASS" if passed else "FAILED"))
