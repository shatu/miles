"""Turn SafeComp tasks into a TMax dataset directory the launcher takes as ``dataset: <dir>``.

    python examples/tmax/safety/make_dataset.py --out DIR --builds BUILD_DIR [BUILD_DIR ...] \\
        [--folder train_clean] [--sources-file F] [--tmax-dataset DIR]

Writes the layout of ``allenai/tmax-15k-open-instruct`` after ``prepare`` has extracted it:

* ``data/train-00000-of-00001.parquet``: one row per task, ``messages`` (the user turn) and ``env_config``
  (``task_id``, ``image``, plus ``variant`` / ``source`` / ``channel`` / ``harm_primary`` for analysis);
* ``task-data.tar.gz.extracted/<task_id>/``: ``instruction.md``, the withheld ``tests/``, and for a risk task whose
  image leaves a safety-test marker, ``withhold``.

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
import json
import shutil
from pathlib import Path

import build_env_images
import pandas as pd

SHARE_DIR = "/weka/oe-adapt-default/mingqianz/datasets/safecomp-test/final_run/share_submission_v1_1"
TMAX_DATASET = "/weka/nora-default/shashankg/miles_tmax_v2/datasets/tmax-15k-open-instruct"
# Left in the image by safety_post.sh for the safety test.
_MARKERS = ("/opt/safecomp",)


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--builds", nargs="+", required=True, help="build_env_images.py output directories")
    parser.add_argument("--share-dir", default=SHARE_DIR)
    parser.add_argument("--folder", default="train_clean")
    parser.add_argument("--sources-file", help="only these source task ids (one per line)")
    parser.add_argument("--tmax-dataset", default=TMAX_DATASET)
    args = parser.parse_args()

    out = Path(args.out)
    assert not out.exists(), f"{out} exists; a dataset directory is written once"
    prefix, suffix = prompt_wrapper(args.tmax_dataset)
    images = _images(args.builds)
    manifest = [json.loads(line) for line in open(f"{args.share_dir}/manifest.jsonl")]
    rows = [r for r in manifest if r["folder"] == args.folder]
    if args.sources_file:
        wanted = {line.strip() for line in open(args.sources_file) if line.strip()}
        rows = [r for r in rows if r["task_id"] in wanted]

    def image_of(row: dict) -> str | None:
        return images.get(row["task_id"] if row["variant"] == "control" else row["name"])

    missing = sorted(r["name"] for r in rows if image_of(r) is None)
    assert not missing, f"{len(missing)} tasks have no built image, e.g. {missing[:5]}"

    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    records = []
    for row in sorted(rows, key=lambda r: r["name"]):
        task_dir = Path(row["target"])
        dst = tmp / "task-data.tar.gz.extracted" / row["name"]
        shutil.copytree(task_dir / "tests", dst / "tests")
        instruction = (task_dir / "instruction.md").read_text()
        (dst / "instruction.md").write_text(instruction)
        if row["variant"] == "risk" and (withheld := markers(task_dir / "environment")):
            (dst / "withhold").write_text("".join(f"{path}\n" for path in withheld))
        records.append(
            {
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
        )
    (tmp / "data").mkdir(parents=True)
    pd.DataFrame(records).to_parquet(tmp / "data/train-00000-of-00001.parquet")
    (tmp / "task-data.tar.gz.extracted/.extraction_complete").touch()
    counts = collections.Counter(r["env_config"]["variant"] for r in records)
    sources = len({r["env_config"]["source"] for r in records})
    (tmp / "README.md").write_text(
        f"SafeComp {args.folder} as a TMax dataset (examples/tmax/safety/make_dataset.py): {dict(counts)} tasks from "
        f"{sources} sources of {args.share_dir}; images from {' '.join(args.builds)}.\n"
    )
    tmp.rename(out)
    print(f"{out}: {dict(counts)}")


if __name__ == "__main__":
    main()
