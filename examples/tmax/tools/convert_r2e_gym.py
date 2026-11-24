"""Convert R2E-Gym-Subset into a TMax task dataset (the layout of allenai/tmax-15k-open-instruct).

R2E-Gym (https://arxiv.org/abs/2504.07164) tasks are real commits from 10 Python repositories: the image holds the
repository at the commit before the fix (``/testbed``, its venv first on PATH) and R2E's hidden tests in
``/r2e_tests``; the reward is 1 iff R2E's test command reproduces every test status of the fix commit
(``expected_output_json``). The Subset is the split DeepSWE trained on: R2E-Gym-V1 minus sympy, matplotlib and moto,
and none of its repositories are in SWE-bench Verified.

Each task becomes ``<task_id>/`` with

* ``instruction.md``: the issue text the policy sees (also in the parquet's user turn).
* ``environment/setup.sh``: run by tmax_sandbox before the first command. It deletes ``/r2e_tests`` (the verifier
  brings its own copy at submit), deletes every git ref, remote and reflog entry and garbage-collects, so the fix
  commit and later history are gone from ``/testbed/.git``, and starts the shell in ``/testbed`` (where SWE-bench
  Verified's harness starts too).
* ``tests/``: ``test.sh`` (restores the repository's conftest.py / pytest.ini / tox.ini files and deletes added
  conftest.py files, so the policy cannot rewrite test outcomes through pytest hooks, then copies the hidden tests
  into ``/testbed/r2e_tests`` and runs ``run_tests.sh`` there, as R2E-Gym's own runtime does), ``score.py``
  (r2e_gym_score.py: R2E's log parser and reward),
  ``expected_output.json``, and the image's own ``r2e_tests/`` and ``/testbed/run_tests.sh``, read from its layers by
  fetch_image_files.py (``--image-files``). The dataset's ``test_file_names`` / ``test_file_codes`` miss files for
  some tasks (pyramid's tests have subdirectories, pillow's run through ``r2e_tests/unittest_custom_runner.py``), and
  the command differs by repository; a copy outside the container also keeps the policy from editing either.
* ``solution/``: ``files.json`` (the fix commit's non-test files) and ``meta.json``, for validate_swe_tasks.py; never
  uploaded to the sandbox.

The user turn is ``tmax_prompts.render_instance(<issue text>)``, the template of every TMax prompt and of
Vanillux2Agent's evaluation prompts, so an R2E task reads like a SWE-bench Verified task at evaluation.

Dropped (``drops.csv`` names the reason):

* ``inconsistent_reference``: R2E's own fix-commit test log does not reproduce ``expected_output_json`` under
  score.py's parser (flaky or environment-dependent tests); 70 tasks.
* ``leaky_statement``: the issue text is R2E's "Analysis / Key Parts from Commit Details" summary of the fix.
* ``empty_statement`` / ``bad_expected`` (an empty test name).
* ``no_image_files``: fetch_image_files.py could not read the image's tests or test command.

Writes ``<out>/data/train-00000-of-00001.parquet``, ``<out>/task-data.tar.gz``, ``<out>/manifest.csv``,
``<out>/drops.csv`` and ``<out>/README.md``. Usage:

    python examples/tmax/tools/fetch_image_files.py --images-from <R2E-Gym-Subset dir> --out <files> --gateway ...
    python examples/tmax/tools/convert_r2e_gym.py --source-dir <R2E-Gym-Subset dir> --image-files <files> \
        --out /weka/.../datasets/r2e-gym-subset-tmax
"""

import argparse
import csv
import glob
import io
import json
import os
import re
import sys
import tarfile
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tmax_prompts import TMAX_SYSTEM_PROMPT, render_instance  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r2e_gym_score import parse as parse_log  # noqa: E402

SOURCE = "R2E-Gym/R2E-Gym-Subset"
ENV_NAME = "swerl_vanillux_sandbox"
ANSI = re.compile(r"\x1b\[[0-9;]*m")
ISSUE = re.compile(r"\**\[ISSUE\]\**(.*?)(?:\**\[/ISSUE\]\**|\Z)", re.S)
LEAKY = re.compile(r"\*\*Analysis:?\*\*|Key Parts from Commit")

SETUP_SH = r"""#!/bin/bash
# R2E-Gym task setup (examples/tmax/tools/convert_r2e_gym.py), run once before the policy's first command.
set -euo pipefail
# The verifier uploads its own copy of the hidden tests at submit.
rm -rf /r2e_tests
cd /testbed
# The image cloned the whole repository and checked out the commit before the fix: drop every ref, remote and reflog
# entry, then the objects only they reached, so neither the fix nor any later commit is in .git. HEAD stays detached
# at the task's commit.
git for-each-ref --format='%(refname)' | while read -r ref; do git update-ref -d "$ref"; done
for remote in $(git remote); do git remote remove "$remote"; done
git reflog expire --expire=now --all
git gc --prune=now --quiet
printf '%s\n' /testbed > "$TMAX_CWD_FILE"
"""

TEST_SH = r"""#!/bin/bash
# R2E-Gym verifier (examples/tmax/tools/convert_r2e_gym.py): the image's test command (its /testbed/run_tests.sh, a
# copy the policy never saw) on a fresh copy of the hidden tests in /testbed/r2e_tests, as R2E-Gym's runtime runs
# them, scored by score.py against the fix commit's test statuses.
mkdir -p /logs/verifier
echo 0 > /logs/verifier/reward.txt
cd /testbed || exit 1
# The policy's edits to pytest's configuration do not count (a conftest.py hook can report every test as passed):
# restore the repository's conftest.py, pytest.ini and tox.ini files and delete conftest.py files it added.
git ls-files -z -- '*conftest.py' '*pytest.ini' '*tox.ini' | xargs -0 -r git checkout -q HEAD -- 2>/dev/null
git ls-files -z --others -- '*conftest.py' ':!.venv' | xargs -0 -r rm -f
rm -rf /testbed/r2e_tests
cp -r /tests/r2e_tests /testbed/r2e_tests
bash /tests/run_tests.sh > /logs/verifier/test_output.txt 2>&1
PY=/testbed/.venv/bin/python
[ -x "$PY" ] || PY=python
# score.py needs Python 3.6+: a venv too old to run it falls back to the system python3.
"$PY" /tests/score.py /logs/verifier/test_output.txt /tests/expected_output.json \
    || python3 /tests/score.py /logs/verifier/test_output.txt /tests/expected_output.json
"""

# score.py of every task, and the parser that checks each task's reference log with it.
SCORE_PY = (Path(__file__).resolve().parent / "r2e_gym_score.py").read_text()


def is_test_path(path: str) -> bool:
    """A repository test file (not part of the fix the policy must write)."""
    parts = path.split("/")
    name = parts[-1]
    return (
        any(p in ("test", "tests") for p in parts[:-1])
        or name.startswith("test_")
        or name.endswith("_test.py")
        or name == "conftest.py"
    )


def clean_statement(text: str) -> str:
    match = ISSUE.search(text)
    return (match.group(1) if match else text).strip()


def _strip_diff_prefix(side: dict | None) -> str | None:
    """A diff side's path ({"path": "a/x.py"}) without its a/ or b/ prefix; None for /dev/null."""
    path = (side or {}).get("path")
    if not path or path == "/dev/null":
        return None
    return path[2:] if path[:2] in ("a/", "b/") else path


def gold_files(parsed_commit: dict) -> dict:
    """The fix commit's non-test files: {"write": {path: content}, "delete": [path], "binary": [path]}."""
    write, delete, binary = {}, [], []
    for diff in parsed_commit["file_diffs"]:
        old, new = _strip_diff_prefix(diff.get("minus_file")), _strip_diff_prefix(diff.get("plus_file"))
        path = new or old
        if path is None or is_test_path(path):
            continue
        if diff.get("is_binary_file"):
            binary.append(path)
        elif new is None:
            delete.append(old)
        else:
            write[new] = diff.get("new_file_content") or ""
            if old and old != new:
                delete.append(old)
    return {"write": write, "delete": delete, "binary": binary}


def _add(tar: tarfile.TarFile, name: str, content: str | bytes, mode: int = 0o644) -> None:
    data = content if isinstance(content, bytes) else content.encode()
    info = tarfile.TarInfo(name)
    info.size, info.mode = len(data), mode
    tar.addfile(info, io.BytesIO(data))


def load_source(source_dir: str | None, cache_dir: str) -> pd.DataFrame:
    if not source_dir:
        from huggingface_hub import snapshot_download

        source_dir = snapshot_download(SOURCE, repo_type="dataset", cache_dir=cache_dir, allow_patterns=["data/*"])
    files = sorted(glob.glob(os.path.join(source_dir, "**", "*.parquet"), recursive=True))
    if not files:
        raise SystemExit(f"no parquet files under {source_dir}")
    return pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)


def image_files(root: Path, image: str) -> tuple[dict[str, bytes], bytes] | None:
    """The image's r2e_tests/ files and its run_tests.sh, as fetch_image_files.py stored them; None if missing."""
    repo, tag = image.rsplit(":", 1)
    base = root / repo / tag
    run_tests = base / "testbed" / "run_tests.sh"
    if not (base / "layers.json").exists() or not run_tests.exists() or not (base / "r2e_tests").is_dir():
        return None
    tests = {
        str(p.relative_to(base / "r2e_tests")): p.read_bytes() for p in (base / "r2e_tests").rglob("*") if p.is_file()
    }
    return tests, run_tests.read_bytes()


def convert(df: pd.DataFrame, out: Path, limit: int, files_root: Path) -> None:
    rows, manifest, drops = [], [], []
    (out / "data").mkdir(parents=True, exist_ok=True)
    tar_path = out / "task-data.tar.gz"
    with tarfile.open(tar_path.with_suffix(".tmp"), "w:gz") as tar:
        for r in df.itertuples():
            task_id = f"r2e_{r.repo_name}__{r.commit_hash[:12]}"
            expected = {ANSI.sub("", k).strip(): v for k, v in json.loads(r.expected_output_json).items()}
            execution = json.loads(r.execution_result_content)
            statement = clean_statement(r.problem_statement)
            reference = parse_log(execution["new_commit_res_stdout"])
            from_image = image_files(files_root, r.docker_image)
            reason = None
            if from_image is None:
                reason = "no_image_files"
            elif "" in expected:
                reason = "bad_expected"
            elif not statement:
                reason = "empty_statement"
            elif LEAKY.search(statement):
                reason = "leaky_statement"
            elif any(reference.get(k) != v for k, v in expected.items()):
                reason = "inconsistent_reference"
            if reason:
                drops.append({"task_id": task_id, "repo": r.repo_name, "image": r.docker_image, "reason": reason})
                continue
            parsed_commit = json.loads(r.parsed_commit_content)
            gold = gold_files(parsed_commit)
            tests, run_tests = from_image
            _add(tar, f"{task_id}/instruction.md", statement + "\n")
            _add(tar, f"{task_id}/environment/setup.sh", SETUP_SH, 0o755)
            _add(tar, f"{task_id}/tests/test.sh", TEST_SH, 0o755)
            _add(tar, f"{task_id}/tests/run_tests.sh", run_tests, 0o755)
            _add(tar, f"{task_id}/tests/score.py", SCORE_PY)
            _add(tar, f"{task_id}/tests/expected_output.json", json.dumps(expected, indent=1, ensure_ascii=False))
            for name, code in tests.items():
                _add(tar, f"{task_id}/tests/r2e_tests/{name}", code)
            _add(tar, f"{task_id}/solution/files.json", json.dumps(gold))
            meta = {
                "source": SOURCE,
                "repo": r.repo_name,
                "image": r.docker_image,
                "fix_commit": r.commit_hash,
                "base_commit": parsed_commit.get("old_commit_hash"),
            }
            _add(tar, f"{task_id}/solution/meta.json", json.dumps(meta))
            rows.append(
                {
                    "messages": [
                        {"role": "system", "content": TMAX_SYSTEM_PROMPT},
                        {"role": "user", "content": render_instance(statement)},
                    ],
                    "ground_truth": task_id,
                    "dataset": "passthrough",
                    "env_config": {"env_name": ENV_NAME, "image": r.docker_image, "task_id": task_id},
                    "source": "r2e_gym_subset",
                }
            )
            statuses = Counter(expected.values())
            manifest.append(
                {
                    "task_id": task_id,
                    "repo": r.repo_name,
                    "image": r.docker_image,
                    "fix_commit": r.commit_hash,
                    "test_files": len(tests),
                    "run_tests": run_tests.decode(errors="replace").strip(),
                    "expected_tests": len(expected),
                    "expected_failed": statuses["FAILED"],
                    "expected_error": statuses["ERROR"],
                    "gold_files": len(gold["write"]) + len(gold["delete"]),
                    "gold_binary": len(gold["binary"]),
                    "num_non_test_lines": r.num_non_test_lines,
                    "statement_chars": len(statement),
                }
            )
            if limit and len(rows) >= limit:
                break
    os.replace(tar_path.with_suffix(".tmp"), tar_path)
    pd.DataFrame(rows).to_parquet(out / "data" / "train-00000-of-00001.parquet", index=False)
    for name, records, fields in (
        ("manifest.csv", manifest, list(manifest[0]) if manifest else ["task_id"]),
        ("drops.csv", drops, ["task_id", "repo", "image", "reason"]),
    ):
        with open(out / name, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
    reasons = Counter(d["reason"] for d in drops)
    repos = Counter(m["repo"] for m in manifest)
    (out / "README.md").write_text(
        f"# R2E-Gym-Subset as TMax tasks\n\n"
        f"Converted from `{SOURCE}` by `examples/tmax/tools/convert_r2e_gym.py` (its docstring has the layout).\n\n"
        f"- {len(rows)} tasks kept of {len(df)}; dropped: {dict(reasons)}\n"
        f"- repositories: {dict(repos.most_common())}\n"
        f"- validate in sandboxes with `examples/tmax/tools/validate_swe_tasks.py` before training.\n"
    )
    print(f"kept {len(rows)} of {len(df)} tasks; dropped {dict(reasons)}; wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", required=True, help="output dataset directory")
    parser.add_argument("--source-dir", help=f"a local copy of {SOURCE} (default: download it)")
    parser.add_argument("--cache-dir", default=os.path.expanduser("~/.cache/huggingface/hub"))
    parser.add_argument(
        "--image-files", type=Path, required=True, help="fetch_image_files.py's output: each image's tests and command"
    )
    parser.add_argument("--limit", type=int, default=0, help="keep only the first N tasks (0 = all)")
    args = parser.parse_args()
    convert(load_source(args.source_dir, args.cache_dir), Path(args.out), args.limit, args.image_files)


if __name__ == "__main__":
    main()
