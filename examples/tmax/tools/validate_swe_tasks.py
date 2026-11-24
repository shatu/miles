"""Check SWE tasks in their sandboxes before training on them: the unchanged repository scores 0, the fix scores 1.

For every task of a TMax-format dataset whose task directories carry ``solution/`` (convert_r2e_gym.py,
convert_swe_rebench_v2.py), two fresh sandboxes through tmax_sandbox, configured by the training env vars
(``TMAX_SANDBOX_BACKEND=gateway`` with ``TMAX_GATEWAY_URL``, or podman with ``TMAX_DOCKER_HOST``):

1. noop: after the task's ``environment/setup.sh`` ran, check what it promises (``solution/meta.json``'s
   ``hidden_paths`` gone, by default ``/r2e_tests``; its ``fix_commit``, if any, unreachable; no git refs left; the
   shell starting in its ``workdir``, by default ``/testbed``), time a second ``git gc`` as an estimate of the
   setup's cost, and submit the repository as is: the reward must be 0.
2. gold: apply the fix (``solution/files.json``: the fix commit's non-test files; or ``solution/fix.patch``, applied
   with ``git apply``) in the workdir and submit: the reward must be 1.

One JSON line per task goes to ``--out`` (``ok`` = both rewards right and every check passed), with the start and
test times and score.py's mismatches; tasks already there are skipped, so a rerun resumes (``--retry-errors`` also
reruns tasks that hit an infrastructure error). ``--summary`` prints the counts of an existing file, and
``--export DIR`` writes a dataset of only the tasks that passed (the parquet rows and task directories), the one to
train on. Usage:

    TMAX_SANDBOX_BACKEND=gateway TMAX_GATEWAY_URL=http://127.0.0.1:24750 \
        python examples/tmax/tools/validate_swe_tasks.py <dataset_dir> --out results.jsonl --concurrency 32
"""

import argparse
import asyncio
import io
import json
import logging
import os
import random
import shlex
import sys
import tarfile
import time
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import tmax_sandbox  # noqa: E402

logger = logging.getLogger("validate_swe_tasks")

_PROBE = """
for p in {hidden}; do test -e "$p" && echo hidden_present=$p; done
cd {workdir} || exit 1
{fix_check}
echo refs=$(git for-each-ref | wc -l)
echo head=$(git rev-parse HEAD)
echo history=$(git rev-list --count HEAD)
echo git_mb=$(du -sm .git | cut -f1)
echo regc_s=$( {{ TIMEFORMAT=%R; time git gc --prune=now --quiet; }} 2>&1 )
"""


def make_sandbox(image: str, task_dir: str) -> tmax_sandbox.SandboxBase:
    if tmax_sandbox.backend_name() == "gateway":
        from tmax_gateway import TmaxGatewaySandbox

        return TmaxGatewaySandbox(image=image, task_dir=task_dir)
    return tmax_sandbox.TmaxSandbox(image=image, task_dir=task_dir)


def _workdir(meta: dict) -> str:
    return meta.get("workdir", "/testbed")


def _gold_tar(files: dict, workdir: str) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for path, content in files.items():
            data = content.encode()
            info = tarfile.TarInfo(f"{workdir.strip('/')}/{path}")
            info.size, info.mode = len(data), 0o644
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


async def _score_details(box: tmax_sandbox.SandboxBase) -> dict:
    """score.py's verdict and the head and tail of the test log, for a task whose reward came out wrong."""
    result = await box._exec("cat /logs/verifier/score.json 2>/dev/null")
    log = await box._exec(
        "head -c 1500 /logs/verifier/test_output.txt; echo; echo ...; tail -c 2500 /logs/verifier/test_output.txt"
    )
    try:
        details = json.loads(result.stdout)
    except ValueError:
        details = {"unparsed": result.stdout[-500:]}
    details["test_output"] = log.stdout
    return details


async def _noop(record: dict, image: str, task_dir: str, meta: dict) -> None:
    t0 = time.monotonic()
    async with make_sandbox(image, task_dir) as box:
        record["noop_start_s"] = round(time.monotonic() - t0, 1)
        fix = meta.get("fix_commit")
        fix_check = (
            f"git cat-file -e {shlex.quote(fix)}^{{commit}} 2>/dev/null && echo fix_commit=present || echo fix_commit=absent"
            if fix
            else ""
        )
        hidden = " ".join(shlex.quote(p) for p in meta.get("hidden_paths", ["/r2e_tests"]))
        probe = await box._exec(
            _PROBE.format(hidden=hidden, workdir=shlex.quote(_workdir(meta)), fix_check=fix_check), timeout_s=600
        )
        record["hidden_present"] = []
        for line in probe.stdout.splitlines():
            key, _, value = line.partition("=")
            if key == "hidden_present":
                record["hidden_present"].append(value)
            elif key and value:
                record[key] = value
        observation, _ = await box.bash("pwd")
        record["start_dir"] = observation.split("\n")[0]
        t1 = time.monotonic()
        record["noop_reward"] = await box.run_tests()
        record["noop_test_s"] = round(time.monotonic() - t1, 1)
        if record["noop_reward"] != 0:
            record["noop_score"] = await _score_details(box)


async def _gold(record: dict, image: str, task_dir: str, meta: dict, gold: dict | str) -> None:
    t0 = time.monotonic()
    workdir = shlex.quote(_workdir(meta))
    async with make_sandbox(image, task_dir) as box:
        record["gold_start_s"] = round(time.monotonic() - t0, 1)
        if isinstance(gold, str):  # a patch
            await box._put_archive(tmax_sandbox._tar_file("/tmp/.validate_fix.patch", gold, 0o644))
            applied = await box._exec(
                f"cd {workdir} && git apply /tmp/.validate_fix.patch; _rc=$?; rm -f /tmp/.validate_fix.patch; exit $_rc"
            )
            if applied.exit_code != 0:
                record["gold_apply_failed"] = applied.stderr[-500:]
        else:
            if gold["write"]:
                await box._put_archive(_gold_tar(gold["write"], _workdir(meta)))
            if gold["delete"]:
                await box._exec(f"cd {workdir} && rm -f -- " + " ".join(shlex.quote(p) for p in gold["delete"]))
        t1 = time.monotonic()
        record["gold_reward"] = await box.run_tests()
        record["gold_test_s"] = round(time.monotonic() - t1, 1)
        if record["gold_reward"] != 1:
            record["gold_score"] = await _score_details(box)


async def check_task(task_id: str, image: str, task_dir: str) -> dict:
    solution = Path(task_dir) / "solution"
    meta = json.loads((solution / "meta.json").read_text())
    patch = solution / "fix.patch"
    gold = patch.read_text() if patch.exists() else json.loads((solution / "files.json").read_text())
    record = {"task_id": task_id, "repo": meta.get("repo"), "image": image}
    steps = {
        "noop": lambda: _noop(record, image, task_dir, meta),
        "gold": lambda: _gold(record, image, task_dir, meta, gold),
    }
    for name, step in steps.items():
        try:
            await step()
        except Exception as e:  # an infrastructure failure: recorded, the task is retried with --retry-errors
            record[f"{name}_error"] = f"{type(e).__name__}: {e}"[:2000]
    checks = {
        "noop_reward": record.get("noop_reward") == 0,
        "gold_reward": record.get("gold_reward") == 1,
        "hidden_paths_gone": "refs" in record and not record.get("hidden_present"),
        "no_refs": record.get("refs") == "0",
        "start_dir": record.get("start_dir") == _workdir(meta),
    }
    if meta.get("fix_commit"):
        checks["fix_commit_gone"] = record.get("fix_commit") == "absent"
    record["failed_checks"] = [k for k, v in checks.items() if not v]
    record["ok"] = not record["failed_checks"]
    record["infra_error"] = any(k.endswith("_error") for k in record)
    return record


def load_tasks(dataset_dir: Path) -> list[tuple[str, str, str]]:
    rows = pd.read_parquet(next((dataset_dir / "data").glob("*.parquet")))
    task_root = dataset_dir / "task-data.tar.gz.extracted"
    if not task_root.is_dir():
        tmp = task_root.with_suffix(".tmp")
        with tarfile.open(dataset_dir / "task-data.tar.gz") as tar:
            tar.extractall(tmp)
        os.replace(tmp, task_root)
    return [(e["task_id"], e["image"], str(task_root / e["task_id"])) for e in rows.env_config]


def read_results(path: Path) -> dict[str, dict]:
    """The latest record of every task in a results file."""
    records = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        records[r["task_id"]] = r
    return records


def export(dataset_dir: Path, results: Path, out: Path) -> None:
    """Write the tasks that passed every check as a dataset of their own, in the same layout."""
    ok = {task_id for task_id, r in read_results(results).items() if r["ok"]}
    load_tasks(dataset_dir)  # extracts the task directories if needed
    rows = pd.read_parquet(next((dataset_dir / "data").glob("*.parquet")))
    rows = rows[rows.env_config.map(lambda e: e["task_id"] in ok)]
    (out / "data").mkdir(parents=True, exist_ok=True)
    rows.to_parquet(out / "data" / "train-00000-of-00001.parquet", index=False)
    with tarfile.open(out / "task-data.tar.gz.tmp", "w:gz") as tar:
        for env in rows.env_config:
            tar.add(dataset_dir / "task-data.tar.gz.extracted" / env["task_id"], arcname=env["task_id"])
    os.replace(out / "task-data.tar.gz.tmp", out / "task-data.tar.gz")
    (out / "README.md").write_text(
        f"# {dataset_dir.name}, validated\n\n{len(rows)} tasks of `{dataset_dir}` that passed every check of "
        f"`examples/tmax/tools/validate_swe_tasks.py` in `{results}` (the unchanged repository scores 0, the fix 1, "
        "hidden paths and future history gone, shell in the task's workdir).\n"
    )
    print(f"exported {len(rows)} tasks to {out}")


def summarize(path: Path) -> None:
    rs = list(read_results(path).values())
    print(f"{len(rs)} tasks: {sum(r['ok'] for r in rs)} ok, {sum(r['infra_error'] for r in rs)} with infra errors")
    print("failed checks:", dict(Counter(c for r in rs for c in r["failed_checks"])))
    by_repo = Counter((r["repo"], r["ok"]) for r in rs)
    for repo in sorted({r["repo"] for r in rs}):
        print(f"  {repo}: {by_repo[(repo, True)]} ok / {by_repo[(repo, True)] + by_repo[(repo, False)]}")
    frame = pd.DataFrame(rs)
    cols = [
        c for c in ("noop_start_s", "gold_start_s", "noop_test_s", "gold_test_s", "regc_s", "git_mb") if c in frame
    ]
    print(
        frame[cols].apply(pd.to_numeric, errors="coerce").describe(percentiles=[0.5, 0.9, 0.99]).round(1).to_string()
    )


async def main_async(args: argparse.Namespace) -> None:
    tasks = load_tasks(Path(args.dataset_dir))
    done = read_results(args.out) if args.out.exists() else {}
    if args.task_ids:
        wanted = set(args.task_ids.split(","))
        tasks = [t for t in tasks if t[0] in wanted]
    todo = [t for t in tasks if t[0] not in done or (args.retry_errors and done[t[0]]["infra_error"])]
    random.Random(args.seed).shuffle(todo)
    if args.limit:
        todo = todo[: args.limit]
    logger.info(
        f"{len(tasks)} tasks, {len(done)} already checked, checking {len(todo)} at concurrency {args.concurrency}"
    )
    semaphore, lock, counts = asyncio.Semaphore(args.concurrency), asyncio.Lock(), Counter()

    async def run(task):
        async with semaphore:
            record = await check_task(*task)
        async with lock:
            with open(args.out, "a") as f:
                f.write(json.dumps(record) + "\n")
            counts["ok" if record["ok"] else "infra" if record["infra_error"] else "bad"] += 1
            if sum(counts.values()) % 25 == 0 or not record["ok"]:
                logger.info(f"{dict(counts)} of {len(todo)}; {record['task_id']}: failed {record['failed_checks']}")

    await asyncio.gather(*(run(t) for t in todo))
    summarize(args.out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dataset_dir", help="a dataset written by convert_r2e_gym.py")
    parser.add_argument("--out", type=Path, required=True, help="JSONL of per-task results (appended, resumable)")
    parser.add_argument(
        "--concurrency", type=int, default=16, help="tasks checked at once (2 sandboxes each, in turn)"
    )
    parser.add_argument("--limit", type=int, default=0, help="check at most N more tasks (0 = all)")
    parser.add_argument("--task-ids", default="", help="comma-separated task ids to check")
    parser.add_argument("--seed", type=int, default=0, help="shuffle seed of the task order")
    parser.add_argument("--retry-errors", action="store_true", help="also recheck tasks that hit infra errors")
    parser.add_argument("--summary", action="store_true", help="only summarize --out")
    parser.add_argument("--export", type=Path, help="only write the tasks that passed in --out as a dataset here")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.summary:
        summarize(args.out)
    elif args.export:
        export(Path(args.dataset_dir), args.out, args.export)
    else:
        asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
