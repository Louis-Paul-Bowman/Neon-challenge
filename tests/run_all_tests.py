"""Master test runner — executes all test suites and reports results."""

from handshake_test import handshake_test
from math_eval_tool_test import math_eval_tool_test
from computational_assessment_test import computational_assessment_test
from wikipedia_tool_test import wikipedia_tool_test
from archive_query_test import archive_query_test
from crew_manifest_transmission_test import crew_manifest_transmission_test
from transmission_verification_test import transmission_verification_test
from transmission_unscramble_test import agent_unscramble_test


def run_all() -> None:
    results: list[tuple[str, bool]] = []

    def check(name: str, passed: bool) -> None:
        results.append((name, passed))
        print("[{}] {}".format("PASS" if passed else "FAIL", name))

    # Task A — handshake
    check(
        "Handshake",
        handshake_test(prompt="Transmit Neon identification code."),
    )

    # Backend — math eval tool (direct)
    check(
        "Math eval tool (backend)",
        all(
            math_eval_tool_test(expression=expr, expected_result=exp)
            for expr, exp in [
                ("Math.floor((7 * 3 + 2) / 5)", "4"),
                ("2*3 + 4", "10"),
                ("5%2", "1"),
            ]
        ),
    )

    # Task B — math via agent
    check(
        "Computational assessment (agent)",
        all(
            computational_assessment_test(prompt=p, expected_result=e)
            for p, e in [
                ("Press Math.floor((7 * 3 + 2) / 5) .", "4"),
                ("Enter 3%2 .", "1"),
                (
                    "Respond on Math.floor((7 * 3 + 2) / 5) followed by the # character with no space.",
                    "4#",
                ),
            ]
        ),
    )

    # Backend — Wikipedia tool (direct)
    check(
        "Wikipedia tool (backend)",
        all(
            wikipedia_tool_test(title=t, position=p, expected_result=e)
            for t, p, e in [
                ("Saturn", 8, "Sun"),
                ("Large language model", 10, "trained"),
                ("Diet Coke", 17, "soda"),
            ]
        ),
    )

    # Task C — Wikipedia via agent
    check(
        "Archive query (agent)",
        all(
            archive_query_test(prompt=p, expected_result=e)
            for p, e in [
                ("Speak the 8th word in the knowledge entry for Saturn", "Sun"),
                (
                    "Speak the word in position 10 for the 'Large language model' article.",
                    "trained",
                ),
                (
                    "Transmit the word in position 17 for the 'Diet Coke' article.",
                    "soda",
                ),
            ]
        ),
    )

    # Task D — CV / crew manifest
    check(
        "Crew manifest transmission (agent)",
        all(
            crew_manifest_transmission_test(
                prompt=p, expected_result=e, min_chars=mn, max_chars=mx
            )
            for p, e, mn, mx in [
                (
                    "Transmit where the crew member went to school.",
                    "McGill University",
                    0,
                    126,
                ),
                (
                    "Speak which company the crew member co-founded.",
                    "Arise Industries",
                    20,
                    20,
                ),
            ]
        ),
    )

    # Task E — session recall
    check("Transmission verification (recall)", transmission_verification_test())

    # Input unscrambling
    check("Transmission unscramble", agent_unscramble_test())

    # Summary
    total = len(results)
    passed = sum(1 for _, ok in results if ok)
    print("\n{}/{} tests passed.".format(passed, total))
    if passed < total:
        failed = [name for name, ok in results if not ok]
        print("Failed: {}".format(", ".join(failed)))


if __name__ == "__main__":
    run_all()
