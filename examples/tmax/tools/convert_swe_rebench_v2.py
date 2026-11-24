"""Convert SWE-rebench V2's Python tasks into a TMax task dataset (the layout of allenai/tmax-15k-open-instruct).

SWE-rebench V2 (nebius/SWE-rebench-V2, CC-BY-4.0) tasks are merged GitHub PRs with real issues: the image holds the
repository at the PR's base commit in ``/<repository name>`` (its WORKDIR) with the project installed, and the reward
is 1 iff the PR's tests (``test_patch``) make every FAIL_TO_PASS test pass without failing a PASS_TO_PASS one. None of
its repositories are in SWE-bench Verified. Kept: Python tasks the dataset's LLM review labels clean (code ``A``) with
a complete statement of intent (4,994); every one is graded by ``parse_log_pytest``.

Each task becomes ``<task_id>/`` with

* ``instruction.md``: the issue text the policy sees (also in the parquet's user turn, the TMax prompt template).
* ``environment/setup.sh``: deletes ``/swebench_matterhorn`` (the image's copy of the task record, which includes the
  fix), drops every git ref, remote and reflog entry and garbage-collects (the image already scrubs them; this makes
  sure), and starts the shell in the repository.
* ``tests/``: ``test.sh`` (restores the repository's conftest.py / pytest.ini / tox.ini files from the base commit
  and deletes added conftest.py files, puts the files the PR's tests touch back to the base commit, applies
  ``test_patch.diff`` and runs the task's ``test_cmd``), ``score.py`` (swe_rebench_v2_score.py), ``config.json``
  (FAIL_TO_PASS, PASS_TO_PASS) and ``test_files.txt``.
* ``solution/``: ``fix.patch`` (the PR's non-test patch) and ``meta.json`` (``workdir``, ``hidden_paths``), for
  validate_swe_tasks.py; never uploaded to the sandbox.

Writes ``<out>/data/train-00000-of-00001.parquet``, ``<out>/task-data.tar.gz``, ``<out>/manifest.csv``,
``<out>/drops.csv`` and ``<out>/README.md``. Usage:

    python examples/tmax/tools/convert_swe_rebench_v2.py --source-dir <SWE-rebench-V2 dir> --out <dataset>
"""

import argparse
import csv
import glob
import json
import os
import re
import shlex
import sys
import tarfile
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tmax_prompts import TMAX_SYSTEM_PROMPT, render_instance  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_r2e_gym import _add  # noqa: E402

SOURCE = "nebius/SWE-rebench-V2"
ENV_NAME = "swerl_vanillux_sandbox"
HIDDEN_PATHS = ["/swebench_matterhorn"]
# SWE-bench Verified's repositories; SWE-rebench V2 has none of them, checked again here.
VERIFIED_REPOS = {
    "astropy/astropy",
    "django/django",
    "matplotlib/matplotlib",
    "mwaskom/seaborn",
    "pallets/flask",
    "psf/requests",
    "pydata/xarray",
    "pylint-dev/pylint",
    "pytest-dev/pytest",
    "scikit-learn/scikit-learn",
    "sphinx-doc/sphinx",
    "sympy/sympy",
}
SCORE_PY = (Path(__file__).resolve().parent / "swe_rebench_v2_score.py").read_text()
DIFF_PATHS = re.compile(r"^diff --git a/(\S+) b/(\S+)$", re.M)

SETUP_SH = r"""#!/bin/bash
# SWE-rebench V2 task setup (examples/tmax/tools/convert_swe_rebench_v2.py), run once before the first command.
set -euo pipefail
# The image's copy of the task record (with the fix and the tests) and the harness that came with it.
rm -rf /swebench_matterhorn
cd {workdir}
# The image already detached HEAD at the base commit and dropped every other ref; make sure, so no later commit of
# the repository stays in .git.
git for-each-ref --format='%(refname)' | while read -r ref; do git update-ref -d "$ref"; done
for remote in $(git remote); do git remote remove "$remote"; done
git reflog expire --expire=now --all
git gc --prune=now --quiet
printf '%s\n' {workdir} > "$TMAX_CWD_FILE"
"""

TEST_SH = r"""#!/bin/bash
# SWE-rebench V2 verifier (examples/tmax/tools/convert_swe_rebench_v2.py): the PR's tests on the policy's repository,
# scored by score.py (FAIL_TO_PASS all pass, no PASS_TO_PASS fails).
mkdir -p /logs/verifier
echo 0 > /logs/verifier/reward.txt
cd {workdir} || exit 1
BASE={base}
# The policy's edits to pytest's configuration do not count (a conftest.py hook can report every test as passed):
# the base commit's conftest.py, pytest.ini and tox.ini files come back, and conftest.py files it does not have go.
git ls-tree -r -z --name-only "$BASE" | grep -zE '(^|/)(conftest\.py|pytest\.ini|tox\.ini)$' \
    | xargs -0 -r git checkout -q "$BASE" --
git ls-files -z --cached --others -- '*conftest.py' ':!.venv' | while IFS= read -r -d '' f; do
    git cat-file -e "$BASE:$f" 2>/dev/null || rm -f "$f"
done
# The files the PR's tests touch go back to the base commit, so the test patch applies to them as written.
while IFS= read -r f; do
    if git cat-file -e "$BASE:$f" 2>/dev/null; then git checkout -q "$BASE" -- "$f"; else rm -f "$f"; fi
done < /tests/test_files.txt
if ! git apply /tests/test_patch.diff 2> /logs/verifier/test_patch.err; then
    cp /logs/verifier/test_patch.err /logs/verifier/test_output.txt
    exit 0
fi
{{
{test_cmd}
}} > /logs/verifier/test_output.txt 2>&1
python3 /tests/score.py /logs/verifier/test_output.txt /tests/config.json
"""


def load_source(source_dir: str | None, cache_dir: str) -> pd.DataFrame:
    if not source_dir:
        from huggingface_hub import snapshot_download

        source_dir = snapshot_download(SOURCE, repo_type="dataset", cache_dir=cache_dir, allow_patterns=["data/*"])
    files = sorted(glob.glob(os.path.join(source_dir, "**", "*.parquet"), recursive=True))
    if not files:
        raise SystemExit(f"no parquet files under {source_dir}")
    return pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)


def drop_reason(r) -> str | None:
    llm = (r.meta or {}).get("llm_metadata") or {}
    if r.language != "python":
        return "not_python"
    if llm.get("code") != "A" or llm.get("intent_completeness") != "complete":
        return "not_clean"
    if r.repo in VERIFIED_REPOS:
        return "verified_repo"
    if not (r.problem_statement or "").strip():
        return "empty_statement"
    if len(r.FAIL_TO_PASS) == 0:
        return "no_fail_to_pass"
    if (r.install_config or {}).get("log_parser") != "parse_log_pytest":
        return "other_log_parser"
    return None


def convert(df: pd.DataFrame, out: Path, limit: int) -> None:
    rows, manifest, drops = [], [], []
    (out / "data").mkdir(parents=True, exist_ok=True)
    tar_path = out / "task-data.tar.gz"
    with tarfile.open(tar_path.with_suffix(".tmp"), "w:gz") as tar:
        for r in df.itertuples():
            task_id = f"v2_{r.instance_id}"
            reason = drop_reason(r)
            if reason:
                if reason != "not_python":
                    drops.append({"task_id": task_id, "repo": r.repo, "reason": reason})
                continue
            workdir = "/" + r.repo.split("/", 1)[1]
            image = r.image_name.removeprefix("docker.io/")
            statement = r.problem_statement.strip()
            test_files = sorted({p for pair in DIFF_PATHS.findall(r.test_patch) for p in pair})
            test_cmd = r.install_config["test_cmd"]
            _add(tar, f"{task_id}/instruction.md", statement + "\n")
            _add(tar, f"{task_id}/environment/setup.sh", SETUP_SH.format(workdir=shlex.quote(workdir)), 0o755)
            _add(
                tar,
                f"{task_id}/tests/test.sh",
                TEST_SH.format(workdir=shlex.quote(workdir), base=shlex.quote(r.base_commit), test_cmd=test_cmd),
                0o755,
            )
            _add(tar, f"{task_id}/tests/test_patch.diff", r.test_patch)
            _add(tar, f"{task_id}/tests/test_files.txt", "".join(f + "\n" for f in test_files))
            _add(tar, f"{task_id}/tests/score.py", SCORE_PY)
            config = {"FAIL_TO_PASS": list(r.FAIL_TO_PASS), "PASS_TO_PASS": list(r.PASS_TO_PASS)}
            _add(tar, f"{task_id}/tests/config.json", json.dumps(config))
            _add(tar, f"{task_id}/solution/fix.patch", r.patch)
            meta = {
                "source": SOURCE,
                "instance_id": r.instance_id,
                "repo": r.repo,
                "image": image,
                "workdir": workdir,
                "hidden_paths": HIDDEN_PATHS,
                "base_commit": r.base_commit,
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
                    "env_config": {"env_name": ENV_NAME, "image": image, "task_id": task_id},
                    "source": "swe_rebench_v2",
                }
            )
            llm = r.meta["llm_metadata"]
            manifest.append(
                {
                    "task_id": task_id,
                    "repo": r.repo,
                    "image": image,
                    "workdir": workdir,
                    "fail_to_pass": len(r.FAIL_TO_PASS),
                    "pass_to_pass": len(r.PASS_TO_PASS),
                    "difficulty": llm.get("difficulty"),
                    "statement_chars": len(statement),
                    "created_at": r.created_at,
                }
            )
            if limit and len(rows) >= limit:
                break
    os.replace(tar_path.with_suffix(".tmp"), tar_path)
    pd.DataFrame(rows).to_parquet(out / "data" / "train-00000-of-00001.parquet", index=False)
    for name, records, fields in (
        ("manifest.csv", manifest, list(manifest[0]) if manifest else ["task_id"]),
        ("drops.csv", drops, ["task_id", "repo", "reason"]),
    ):
        with open(out / name, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
    reasons = Counter(d["reason"] for d in drops)
    repos = Counter(m["repo"] for m in manifest)
    (out / "README.md").write_text(
        f"# SWE-rebench V2 (Python, clean) as TMax tasks\n\n"
        f"Converted from `{SOURCE}` by `examples/tmax/tools/convert_swe_rebench_v2.py` (its docstring has the "
        f"layout).\n\n"
        f"- {len(rows)} Python tasks kept; dropped (Python only): {dict(reasons)}\n"
        f"- {len(repos)} repositories; largest: {dict(repos.most_common(10))}\n"
        f"- validate in sandboxes with `examples/tmax/tools/validate_swe_tasks.py` before training.\n"
    )
    print(f"kept {len(rows)} tasks from {len(repos)} repositories; dropped {dict(reasons)}; wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", required=True, help="output dataset directory")
    parser.add_argument("--source-dir", help=f"a local copy of {SOURCE} (default: download it)")
    parser.add_argument("--cache-dir", default=os.path.expanduser("~/.cache/huggingface/hub"))
    parser.add_argument("--limit", type=int, default=0, help="keep only the first N tasks (0 = all)")
    args = parser.parse_args()
    convert(load_source(args.source_dir, args.cache_dir), Path(args.out), args.limit)


if __name__ == "__main__":
    main()
