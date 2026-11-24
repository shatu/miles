"""R2E-Gym's reward: 1 iff every test of expected_output.json has its expected status in the pytest -rA log.

The parser is R2E-Gym's parse_log_pytest: the lines after "short test summary info", the test id without its file
("Class.test" or "test[param]"), and for failures and errors the id before " - <message>". It reproduces
expected_output_json from R2E's own fix-commit logs for the tasks kept. Plain Python 3, for every task's venv.
"""

import json
import re
import sys

ANSI = re.compile(r"\x1b\[[0-9;]*m")


def parse(log):
    status = {}
    log = ANSI.sub("", log)
    if "short test summary info" not in log:
        return status
    for line in log.split("short test summary info", 1)[1].strip().splitlines():
        for result in ("PASSED", "FAILED", "ERROR"):
            if line.startswith(result + " "):
                name = ".".join(line[len(result) + 1 :].split("::")[1:])
                if result != "PASSED":
                    name = name.split(" - ")[0]
                status[name.strip()] = result
                break
    return status


def main():
    with open(sys.argv[1], errors="replace") as f:
        actual = parse(f.read())
    with open(sys.argv[2]) as f:
        expected = {ANSI.sub("", k).strip(): v for k, v in json.load(f).items()}
    mismatched = sorted(k for k, v in expected.items() if actual.get(k) != v)
    reward = 0 if mismatched else 1
    with open("/logs/verifier/reward.txt", "w") as f:
        f.write(f"{reward}\n")
    details = {
        "reward": reward,
        "expected": len(expected),
        "parsed": len(actual),
        "mismatched": [[k, expected[k], actual.get(k)] for k in mismatched[:20]],
    }
    with open("/logs/verifier/score.json", "w") as f:
        json.dump(details, f)
    print(f"reward {reward}: {len(expected) - len(mismatched)}/{len(expected)} expected statuses matched")


if __name__ == "__main__":
    main()
