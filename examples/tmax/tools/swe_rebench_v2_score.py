"""SWE-rebench V2's reward for its pytest tasks: 1 iff every FAIL_TO_PASS test passed and no PASS_TO_PASS test failed.

The parser is SWE-rebench V2's parse_log_pytest (as Harbor's swe-rebench-v2 adapter vendors it): every log line that
starts with a status, the test id after it, with a failure's " - <message>" folded in. Test ids are matched exactly;
a PASS_TO_PASS test missing from the log is tolerated (it may not have run), a FAIL_TO_PASS one is not. Usage:

    python3 score.py <test log> <config.json with FAIL_TO_PASS and PASS_TO_PASS>

Writes /logs/verifier/reward.txt and /logs/verifier/score.json. Plain Python 3, for every task's interpreter.
"""

import json
import re
import sys

ANSI = re.compile(r"\x1b\[[0-9;]*m")
STATUSES = ("PASSED", "FAILED", "SKIPPED", "ERROR")


def parse(log):
    results = {}
    for line in ANSI.sub("", log).splitlines():
        for status in STATUSES:
            if line.startswith(status):
                if status == "FAILED":
                    line = line.replace(" - ", " ")
                parts = line.split()
                if len(parts) >= 2:
                    results[parts[1]] = parts[0]
                break
    return results


def grade(results, fail_to_pass, pass_to_pass):
    """The FAIL_TO_PASS tests that did not pass and the PASS_TO_PASS tests that failed or errored."""
    missed = [t for t in fail_to_pass if results.get(t) != "PASSED"]
    broken = [t for t in pass_to_pass if results.get(t) in ("FAILED", "ERROR")]
    return missed, broken


def main():
    with open(sys.argv[1], errors="replace") as f:
        results = parse(f.read())
    with open(sys.argv[2]) as f:
        config = json.load(f)
    missed, broken = grade(results, config["FAIL_TO_PASS"], config["PASS_TO_PASS"])
    reward = 0 if missed or broken else 1
    with open("/logs/verifier/reward.txt", "w") as f:
        f.write(f"{reward}\n")
    details = {
        "reward": reward,
        "parsed": len(results),
        "fail_to_pass": len(config["FAIL_TO_PASS"]),
        "pass_to_pass": len(config["PASS_TO_PASS"]),
        "missed": [[t, results.get(t)] for t in missed[:20]],
        "broken": [[t, results.get(t)] for t in broken[:20]],
    }
    with open("/logs/verifier/score.json", "w") as f:
        json.dump(details, f)
    print(f"reward {reward}: {len(missed)} FAIL_TO_PASS not passed, {len(broken)} PASS_TO_PASS broken")


if __name__ == "__main__":
    main()
