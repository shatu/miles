"""Merge TMax task datasets (data/*.parquet + task-data.tar.gz each) into one, e.g. R2E-Gym and SWE-rebench V2.

The launcher's ``dataset:`` takes one directory, so a mixed run trains on a merged one: the parquet rows concatenated
(their natural proportions; the launcher shuffles prompts) and every task directory copied into one tarball. Task ids
must be unique across the inputs (convert_r2e_gym.py prefixes ``r2e_``, convert_swe_rebench_v2.py ``v2_``). Usage:

    python examples/tmax/tools/merge_task_datasets.py --out <merged> <dataset> <dataset> [...]
"""

import argparse
import os
import tarfile
from collections import Counter
from pathlib import Path

import pandas as pd


def merge(inputs: list[Path], out: Path) -> None:
    frames = [pd.read_parquet(next((d / "data").glob("*.parquet"))) for d in inputs]
    ids = [e["task_id"] for f in frames for e in f.env_config]
    duplicates = [i for i, n in Counter(ids).items() if n > 1]
    if duplicates:
        raise SystemExit(f"{len(duplicates)} task ids in more than one input, e.g. {duplicates[:3]}")
    (out / "data").mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_parquet(out / "data" / "train-00000-of-00001.parquet", index=False)
    wanted = set(ids)
    with tarfile.open(out / "task-data.tar.gz.tmp", "w:gz") as merged:
        for d in inputs:
            with tarfile.open(d / "task-data.tar.gz") as tar:
                for member in tar:
                    if member.name.split("/", 1)[0] in wanted:
                        merged.addfile(member, tar.extractfile(member) if member.isfile() else None)
    os.replace(out / "task-data.tar.gz.tmp", out / "task-data.tar.gz")
    counts = {d.name: len(f) for d, f in zip(inputs, frames, strict=True)}
    (out / "README.md").write_text(
        "# Merged TMax task datasets\n\n"
        f"Written by `examples/tmax/tools/merge_task_datasets.py` from: {counts} ({len(ids)} tasks).\n"
    )
    print(f"merged {counts} into {out} ({len(ids)} tasks)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("inputs", type=Path, nargs="+")
    args = parser.parse_args()
    merge(args.inputs, args.out)


if __name__ == "__main__":
    main()
