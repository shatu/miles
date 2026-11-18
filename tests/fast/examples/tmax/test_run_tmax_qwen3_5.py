"""The TMax launcher's GPU layout with SGLang OPD teacher servers (run_tmax_qwen3_5.py)."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "examples" / "tmax"))

import run_tmax_qwen3_5 as launcher  # noqa: E402


def _args(**kwargs):
    return launcher.ScriptArgs(model_name="Qwen3.5-4B-tmax-template", opd_teacher="tmax-9b", **kwargs)


def test_teachers_take_rollout_gpus_and_training_keeps_whole_nodes():
    args = _args(num_nodes=4, async_rollout_gpus=14, opd_teacher_gpus=2)
    assert args.opd_mode == "sglang", "a 9B teacher of a 4B student is served by SGLang"
    assert args.train_layout == (2, 8)


def test_teacher_and_rollout_gpus_must_leave_training_whole_nodes():
    with pytest.raises(AssertionError, match="whole nodes"):
        _ = _args(num_nodes=4, async_rollout_gpus=16, opd_teacher_gpus=2).train_layout


def test_on_one_node_training_gets_what_rollout_and_the_teachers_leave():
    assert _args(num_gpus_per_node=4, async_rollout_gpus=1, opd_teacher_gpus=2).train_layout == (1, 1)
    assert _args(num_gpus_per_node=8, opd_teacher_gpus=1).train_layout == (1, 7), "colocated"


def test_colocated_training_with_an_sglang_teacher_is_one_node_only():
    with pytest.raises(AssertionError, match="fully async"):
        _args(num_nodes=2, opd_teacher_gpus=1)


def test_teacher_servers_go_round_robin_over_the_rollout_nodes_last_first():
    assert launcher._opd_teacher_hosts(num_nodes=4, train_nodes=2, num_servers=2) == [3, 2]
    assert launcher._opd_teacher_hosts(num_nodes=4, train_nodes=2, num_servers=4) == [3, 2, 3, 2]
    assert launcher._opd_teacher_hosts(num_nodes=1, train_nodes=1, num_servers=2) == [0, 0]


def test_a_same_architecture_teacher_takes_no_gpu():
    args = launcher.ScriptArgs(opd_teacher="tmax-9b", num_nodes=4, async_rollout_gpus=16)
    assert args.opd_mode == "megatron" and args.teacher_server_gpus == 0 and args.train_layout == (2, 8)


def test_a_hub_recipe_is_downloaded_and_a_local_checkpoint_copied_once():
    assert launcher._fetch_command("allenai/tmax-9b", "/models/tmax-9b") == (
        "hf download allenai/tmax-9b --local-dir /models/tmax-9b"
    )
    copy = launcher._fetch_command("/runs/x/hf/step_79", "/models/step80")
    assert copy.startswith("if [ ! -d /models/step80 ]") and "cp -r /runs/x/hf/step_79 /models/step80.tmp" in copy


def test_the_opd_checkpoint_recipe_is_a_local_qwen35_9b():
    recipe = launcher._RECIPES["tmax-opd-9b-step80"]
    assert recipe.hf_repo.startswith("/") and recipe.hf_repo.endswith("/hf/step_79")
    assert recipe.megatron_model_type == "qwen3.5-9B" and recipe.enable_thinking


def _rollout_only_run(tmp_path, traced: dict[str, int], **kwargs):
    """A rollout_only run over tasks t0..t9 whose earlier attempts traced ``traced`` episodes per task."""
    dataset = tmp_path / "dataset"
    dataset.mkdir(parents=True)
    with open(dataset / "miles_prompts.jsonl", "w") as f:
        for i in range(10):
            f.write(json.dumps({"prompt": [], "metadata": {"task_id": f"t{i}", "image": "x"}}) + "\n")
    args = launcher.ScriptArgs(
        model_name="tmax-4b",
        dataset=str(dataset),
        rollout_only=True,
        trace_dir=str(tmp_path / "traces"),
        output_dir=str(tmp_path / "out"),
        exp_name="probe_20261114_000000",
        rollout_batch_size=2,
        n_samples_per_prompt=4,
        **kwargs,
    )
    run = Path(args.trace_run_dir)
    run.mkdir(parents=True)
    lines = [json.dumps({"task_id": task}) for task, n in traced.items() for _ in range(n)]
    (run / "traces-a.jsonl").write_text("\n".join(lines[: len(lines) // 2]) + "\n")
    (run / "traces-b.jsonl").write_text("\n".join(lines[len(lines) // 2 :]) + '\n{"task_id": "t9", "mess')
    return args


def _task_ids(path: str) -> list[str]:
    return [json.loads(line)["metadata"]["task_id"] for line in open(path)]


def test_a_rollout_only_run_samples_the_same_tasks_for_every_model_and_restart(tmp_path):
    fresh = _rollout_only_run(tmp_path / "a", {}, num_rollout=3)
    other_model = _rollout_only_run(tmp_path / "b", {}, num_rollout=3, rollout_seed=42)
    tasks = _task_ids(launcher._rollout_plan(fresh)[0])
    assert len(tasks) == 6 and tasks == _task_ids(launcher._rollout_plan(other_model)[0])
    reseeded = _rollout_only_run(tmp_path / "c", {}, num_rollout=3, rollout_seed=7)
    assert _task_ids(launcher._rollout_plan(reseeded)[0]) != tasks


def test_a_restarted_rollout_only_run_skips_the_fully_traced_tasks(tmp_path):
    planned = _task_ids(launcher._rollout_plan(_rollout_only_run(tmp_path / "a", {}, num_rollout=3))[0])
    # Two planned tasks done (over two hosts' files), one cut off mid-group, one done outside the sample, and the last
    # line cut off mid-write.
    traced = {planned[0]: 4, planned[1]: 4, planned[2]: 1}
    traced |= {next(f"t{i}" for i in range(10) if f"t{i}" not in planned): 4}
    args = _rollout_only_run(tmp_path / "b", traced, num_rollout=3)
    prompt_data, num_rollout = launcher._rollout_plan(args)
    assert _task_ids(prompt_data) == planned[2:]
    assert num_rollout == 2, "4 of the 6 tasks left, in batches of 2"
    assert prompt_data != args.prompt_data, "the dataset's prompt file is shared by runs and stays whole"


def test_a_fully_traced_rollout_only_run_has_nothing_left(tmp_path):
    planned = _task_ids(launcher._rollout_plan(_rollout_only_run(tmp_path / "a", {}, num_rollout=1))[0])
    args = _rollout_only_run(tmp_path / "b", {planned[0]: 4, planned[1]: 5}, num_rollout=1)
    assert launcher._rollout_plan(args)[1] == 0


def test_a_training_run_keeps_its_prompts_and_rollouts(tmp_path):
    args = _rollout_only_run(tmp_path, {"t0": 4}, num_rollout=3)
    args.rollout_only = False
    assert launcher._rollout_plan(args) == (args.prompt_data, 3)


def test_rollout_only_and_its_seed_are_fields_not_extra_args():
    with pytest.raises(AssertionError, match="set rollout_only"):
        launcher.ScriptArgs(extra_args="--lr 0 --debug-rollout-only")
    with pytest.raises(AssertionError, match="set rollout_seed"):
        launcher.ScriptArgs(extra_args="--rollout-seed 7")


def _split_dataset(tmp_path):
    """A dataset dir with train and two held-out splits of two tasks each."""
    import pandas as pd

    dataset = tmp_path / "dataset"
    (dataset / "data").mkdir(parents=True)
    for split in ("train", "validation", "id_test"):
        rows = [
            {
                "messages": [{"role": "user", "content": f"do {split} {i}"}],
                "env_config": {"task_id": f"{split}-{i}", "image": "x"},
            }
            for i in range(2)
        ]
        pd.DataFrame(rows).to_parquet(dataset / "data" / f"{split}-00000-of-00001.parquet")
    return dataset


def test_held_out_splits_get_their_own_prompt_file_and_train_keeps_its_name(tmp_path):
    dataset = _split_dataset(tmp_path)
    train = launcher.ScriptArgs(model_name="tmax-4b", dataset=str(dataset))
    held_out = launcher.ScriptArgs(model_name="tmax-4b", dataset=str(dataset), dataset_split="validation,id_test")
    assert train.prompt_data == f"{dataset}/miles_prompts.jsonl"
    assert held_out.prompt_data == f"{dataset}/miles_prompts_validation+id_test.jsonl"
    launcher._write_prompt_data(train)
    launcher._write_prompt_data(held_out)
    assert _task_ids(train.prompt_data) == ["train-0", "train-1"]
    assert _task_ids(held_out.prompt_data) == ["validation-0", "validation-1", "id_test-0", "id_test-1"]
    assert sorted(p.name for p in dataset.iterdir() if p.is_file()) == [
        "miles_prompts.jsonl",
        "miles_prompts_validation+id_test.jsonl",
    ], "no temporary file is left behind"
