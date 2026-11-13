"""Turn SafeComp tasks into a TMax dataset directory: ``dataset: <dir>`` for the launcher, or a Hugging Face repo.

    python examples/tmax/safety/make_dataset.py --out DIR --builds BUILD_DIR [BUILD_DIR ...] \\
        [--split NAME=FOLDER[:val] ...] [--variants control risk] [--sources-file F] [--tarball] [--tmax-dataset DIR]

Writes the layout of ``allenai/tmax-15k-open-instruct`` after ``prepare`` has extracted it:

* ``data/<split>-00000-of-00001.parquet``: one row per task, ``messages`` (the user turn) and ``env_config``
  (``task_id``, ``image``, plus ``variant`` / ``source`` / ``channel`` / ``harm_primary`` for analysis); the
  launcher trains on ``train``;
* ``task-data.tar.gz.extracted/<task_id>/``: ``instruction.md``, the withheld ``tests/``, and for a risk task whose
  image leaves a safety-test marker, ``withhold``; with ``--tarball`` also ``task-data.tar.gz`` (the same task
  directories), which is what a Hugging Face copy of the dataset carries;
* ``README.md``: a dataset card naming the splits.

``--split NAME=FOLDER`` takes the rows of one folder of the share; ``FOLDER:val`` only the rows of the share's
validation sources (``in_val_sources``; ``train_clean`` minus ``train_clean_excl_val``). A task linked from several
folders is written once: her copies of a source's control differ between folders only in provenance hashes in
``tests/safecomp_meta.json``, which is checked.

The user turn is TMax's: the prefix and suffix around ``instruction.md`` are read from ``--tmax-dataset`` (and checked
to be the same on every sampled row), so both datasets prompt identically.

Images come from ``build_env_images.py`` (``--builds``: its control and risk output directories): a control task runs
its source's control image, a risk task its own image (its appended Dockerfile block baked onto that control image).
``withhold`` lists the paths a risk image keeps for its safety test that would mark the task as one
(``/opt/safecomp``, the protected file's digest, written by ``safety_post.sh``; controls never have it): the sandbox
moves them out of the container before the agent's first turn and back before the tests.
"""

import argparse
import collections
import filecmp
import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import build_env_images
import pandas as pd

SHARE_DIR = "/weka/oe-adapt-default/mingqianz/datasets/safecomp-test/final_run/share_submission_v1_1"
TMAX_DATASET = "/weka/nora-default/shashankg/miles_tmax_v2/datasets/tmax-15k-open-instruct"
# Left in the image by safety_post.sh for the safety test.
_MARKERS = ("/opt/safecomp",)
# Provenance in tests/safecomp_meta.json that differs between her copies of one control task.
_PROVENANCE = ("scenario_sha256", "spec_bundle_sha256")


def prompt_wrapper(tmax_dataset: str, rows: int = 300) -> tuple[str, str]:
    """(prefix, suffix) around instruction.md in TMax's user turn, the same on every sampled row."""
    df = pd.read_parquet(f"{tmax_dataset}/data/train-00000-of-00001.parquet")
    wrappers = collections.Counter()
    for row in df.head(rows).itertuples():
        (user,) = [m["content"] for m in row.messages if m["role"] == "user"]
        instruction = Path(f"{tmax_dataset}/task-data.tar.gz.extracted/{row.env_config['task_id']}/instruction.md")
        text = instruction.read_text()
        at = user.find(text)
        assert at >= 0, f"{row.env_config['task_id']}: instruction.md not in the user turn"
        wrappers[(user[:at], user[at + len(text) :])] += 1
    assert len(wrappers) == 1, f"TMax's user turns use {len(wrappers)} different wrappers"
    return next(iter(wrappers))


def markers(env_dir: Path) -> list[str]:
    """The marker paths a risk task's build leaves (its build scripts name them)."""
    scripts = [p.read_text(errors="ignore") for p in env_dir.glob("*.sh")]
    return sorted(m for m in _MARKERS if any(m in s for s in scripts))


def _images(build_dirs: list[str]) -> dict[str, str]:
    """task (a control's source id, or a risk task's name) -> image, for every pushed (or present) build."""
    return {task: entry["image"] for task, entry in build_env_images._built(build_dirs).items()}


def same_task(a: Path, b: Path) -> bool:
    """Whether two copies of a task give the agent and the tests the same files (provenance hashes aside)."""
    for rel in ["instruction.md", *sorted(str(p.relative_to(a)) for p in (a / "tests").rglob("*") if p.is_file())]:
        if not (b / rel).is_file():
            return False
        if rel == "tests/safecomp_meta.json":
            meta_a, meta_b = (json.loads((d / rel).read_text()) for d in (a, b))
            if {k: v for k, v in meta_a.items() if k not in _PROVENANCE} != {
                k: v for k, v in meta_b.items() if k not in _PROVENANCE
            }:
                return False
        elif not filecmp.cmp(a / rel, b / rel, shallow=False):
            return False
    return sorted(p.relative_to(b) for p in (b / "tests").rglob("*")) == sorted(
        p.relative_to(a) for p in (a / "tests").rglob("*")
    )


def select(manifest: list[dict], spec: str) -> list[dict]:
    """The manifest rows of ``FOLDER`` or ``FOLDER:val`` (only the validation sources' rows)."""
    folder, _, which = spec.partition(":")
    assert which in ("", "val"), f"{spec}: expected FOLDER or FOLDER:val"
    rows = [r for r in manifest if r["folder"] == folder and (not which or r["in_val_sources"])]
    assert rows, f"{spec}: no rows"
    return rows


def _card(splits: dict[str, list[dict]], args: argparse.Namespace) -> str:
    configs = "".join(f"  - split: {name}\n    path: data/{name}-00000-of-00001.parquet\n" for name in splits)
    table = "".join(
        f"| `{name}` | `{spec}` | {sum(r['env_config']['variant'] == 'risk' for r in recs)} | "
        f"{sum(r['env_config']['variant'] == 'control' for r in recs)} | "
        f"{len({r['env_config']['source'] for r in recs})} |\n"
        for (name, recs), spec in zip(splits.items(), args.split, strict=True)
    )
    return f"""---
configs:
- config_name: default
  data_files:
{configs}---

# SafeComp terminal tasks for TMax RL

SafeComp (mingqianz's `share_submission_v1_1`, built on `allenai/TMax-SFT-16.5K`) in the layout of
`allenai/tmax-15k-open-instruct`, for the TMax trainer in `examples/tmax` (`dataset: <this repo>`), which trains on
`train`. Variants: {" + ".join(args.variants)}.

| split | share folder | risk | control | sources |
|---|---|---:|---:|---:|
{table}
`FOLDER:val` keeps only her validation sources (`train_clean` minus `train_clean_excl_val`).

## Rows

- `messages`: the user turn (TMax's wrapper around `instruction.md`).
- `env_config.task_id`: the task directory in `task-data.tar.gz`; `image`: its prebuilt environment
  (`docker.io/shashankg209/tmax-s-envs`, private; a control task runs its source's image, a risk task that image plus
  the risk scenario's files); `variant`: `control` (the original task) or `risk` (a safety scenario on it);
  `source`: the original TMax-SFT task; `channel` / `harm_primary`: the risk scenario's (empty for controls).

## Task directories

`instruction.md`, `tests/` (run after the episode; they write `/logs/verifier/reward.txt` and
`/logs/verifier/safecomp.json`) and, for risk tasks whose image keeps a safety-test marker, `withhold`: paths the
sandbox moves out before the agent's first turn and back before the tests. The reward is
`utility - violation` from `safecomp.json`, in [-1, 1].

Made by `examples/tmax/safety/make_dataset.py` (branch `shashank/tmax-safety`).
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--builds", nargs="+", required=True, help="build_env_images.py output directories")
    parser.add_argument("--share-dir", default=SHARE_DIR)
    parser.add_argument("--split", action="append", help="NAME=FOLDER or NAME=FOLDER:val (default train=train_clean)")
    parser.add_argument("--variants", nargs="+", choices=["control", "risk"], default=["control", "risk"])
    parser.add_argument("--sources-file", help="only these source task ids (one per line)")
    parser.add_argument("--tarball", action="store_true", help="also write task-data.tar.gz")
    parser.add_argument("--tmax-dataset", default=TMAX_DATASET)
    args = parser.parse_args()
    args.split = args.split or ["train=train_clean"]

    out = Path(args.out)
    assert not out.exists(), f"{out} exists; a dataset directory is written once"
    prefix, suffix = prompt_wrapper(args.tmax_dataset)
    images = _images(args.builds)
    manifest = [json.loads(line) for line in open(f"{args.share_dir}/manifest.jsonl")]
    wanted = {line.strip() for line in open(args.sources_file) if line.strip()} if args.sources_file else None
    split_rows = {}
    for item in args.split:
        name, _, spec = item.partition("=")
        rows = [r for r in select(manifest, spec) if r["variant"] in args.variants]
        split_rows[name] = [r for r in rows if wanted is None or r["task_id"] in wanted]
    args.split = [item.partition("=")[2] for item in args.split]

    def image_of(row: dict) -> str | None:
        return images.get(row["task_id"] if row["variant"] == "control" else row["name"])

    every = [r for rows in split_rows.values() for r in rows]
    missing = sorted({r["name"] for r in every if image_of(r) is None})
    assert not missing, f"{len(missing)} tasks have no built image, e.g. {missing[:5]}"
    copies = collections.defaultdict(list)
    for row in every:
        if row["target"] not in copies[row["name"]]:
            copies[row["name"]].append(row["target"])
    with ThreadPoolExecutor(32) as pool:
        differing = [
            name
            for name, ok in zip(
                copies,
                pool.map(lambda ts: all(same_task(Path(ts[0]), Path(t)) for t in ts[1:]), copies.values()),
                strict=True,
            )
            if not ok
        ]
    assert not differing, f"{len(differing)} tasks differ between folders, e.g. {differing[:5]}"

    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    first = {}
    for row in every:
        first.setdefault(row["name"], row)

    def write(row: dict) -> None:
        task_dir = Path(row["target"])
        dst = tmp / "task-data.tar.gz.extracted" / row["name"]
        shutil.copytree(task_dir / "tests", dst / "tests")
        shutil.copyfile(task_dir / "instruction.md", dst / "instruction.md")
        if row["variant"] == "risk" and (withheld := markers(task_dir / "environment")):
            (dst / "withhold").write_text("".join(f"{path}\n" for path in withheld))

    with ThreadPoolExecutor(32) as pool:
        list(pool.map(write, first.values()))

    def record(row: dict) -> dict:
        instruction = (tmp / "task-data.tar.gz.extracted" / row["name"] / "instruction.md").read_text()
        return {
            "messages": [{"role": "user", "content": prefix + instruction + suffix}],
            "env_config": {
                "task_id": row["name"],
                "image": image_of(row),
                "variant": row["variant"],
                "source": row["task_id"],
                "channel": row.get("channel") or "",
                "harm_primary": row.get("harm_primary") or "",
            },
        }

    (tmp / "data").mkdir(parents=True)
    splits = {}
    for name, rows in split_rows.items():
        splits[name] = [record(r) for r in sorted(rows, key=lambda r: r["name"])]
        pd.DataFrame(splits[name]).to_parquet(tmp / f"data/{name}-00000-of-00001.parquet")
    if args.tarball:
        subprocess.run(
            ["tar", "-czf", str(tmp / "task-data.tar.gz"), "-C", str(tmp / "task-data.tar.gz.extracted"), "."],
            check=True,
        )
    (tmp / "task-data.tar.gz.extracted/.extraction_complete").touch()
    (tmp / "README.md").write_text(_card(splits, args))
    tmp.rename(out)
    for name, recs in splits.items():
        print(f"{out} {name}: {dict(collections.Counter(r['env_config']['variant'] for r in recs))}")


if __name__ == "__main__":
    main()
