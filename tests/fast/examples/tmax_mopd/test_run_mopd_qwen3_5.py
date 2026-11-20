"""The multi-teacher OPD launcher: GPU layout, teacher servers, the prompt mix and the train command
(run_mopd_qwen3_5.py)."""

import json
import shlex
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[4]
_EXAMPLE = _REPO / "examples" / "tmax_mopd"
sys.path[:0] = [str(_EXAMPLE), str(_REPO / "examples" / "tmax"), str(_REPO / "examples" / "tmax_search")]

import opd_teachers  # noqa: E402
import run_mopd_qwen3_5 as launcher  # noqa: E402

from tests.fast.launch_scripts.py_harness import (  # noqa: E402
    call_entrypoint,
    freeze_environment,
    import_launch_script,
    install_command_recorder,
)

# The search launcher's default Prime-RL source, which the search side's environment is read from.
_SOURCE = Path(launcher.ScriptArgs.prime_rl_source)


def test_prod_layout_two_training_nodes_and_one_server_per_teacher_on_the_rollout_nodes():
    args = launcher.ScriptArgs(num_nodes=4, async_rollout_gpus=14)
    assert args.train_layout == (2, 8)
    assert [domain for domain, _ in args.teacher_servers] == ["terminal", "search"]
    assert args.teacher_servers[0][1].endswith("/tmax-4b")
    assert args.teacher_servers[1][1].endswith("/search-4b-bcrl-v11-step279")
    # The terminal teacher on the last node, the search teacher on the one before.
    assert opd_teachers.hosts(4, args.train_layout[0], len(args.teacher_servers)) == [3, 2]


def test_one_node_smoke_layout():
    assert launcher.ScriptArgs(num_gpus_per_node=4, async_rollout_gpus=1).train_layout == (1, 1)
    with pytest.raises(AssertionError, match="whole nodes"):
        _ = launcher.ScriptArgs(num_nodes=4, async_rollout_gpus=16).train_layout


def test_each_domain_keeps_its_budget_and_the_session_server_caps_at_the_longer():
    args = launcher.ScriptArgs()
    assert args.budgets == {
        "terminal": {"max_seq_len": 67584, "max_tokens": 16384},
        "search": {"max_seq_len": 131072, "max_tokens": 32768},
    }
    assert args.max_seq_len == 131072


def test_a_teacher_must_render_as_the_student():
    with pytest.raises(AssertionError, match="enable_thinking"):
        launcher.ScriptArgs(terminal_teacher="tmax-2b")  # the 2B checkpoints render without thinking


def test_the_terminal_and_search_sides_keep_their_launchers_defaults():
    args = launcher.ScriptArgs()
    assert args.terminal.max_steps == launcher.tmax_launcher.ScriptArgs.max_steps
    assert args.search.prime_rl_config == launcher.search_launcher.ScriptArgs.prime_rl_config
    assert args.start_failure == "drop", "the TMax OPD runs': a fleet failure drops the group"


@pytest.mark.parametrize(
    ("terminal", "search", "fraction", "copies"),
    [(14622, 1131, 0.5, 13), (10, 3, 0.5, 3), (10, 3, 0.2, 1), (10, 100, 0.5, 1)],
)
def test_search_questions_are_repeated_to_reach_their_share(terminal, search, fraction, copies):
    mixed = launcher.mix_prompts(["t"] * terminal, ["s"] * search, fraction)
    assert mixed.count("t") == terminal and mixed.count("s") == search * copies


def test_prompt_rows_name_their_domain_and_teacher(tmp_path):
    args = launcher.ScriptArgs(data_dir=str(tmp_path), search_fraction=0.5)
    for path, rows in (
        (
            args.terminal.prompt_data,
            [{"prompt": [], "metadata": {"task_id": f"t{i}", "image": "img"}} for i in range(4)],
        ),
        (args.search.prompt_data, [{"prompt": [], "metadata": {"id": "q0", "question": "?", "answer": "!"}}]),
    ):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows))
    launcher._write_prompt_data(args)
    rows = [json.loads(line) for line in Path(args.prompt_data).read_text().splitlines()]
    assert [r["metadata"]["domain"] for r in rows] == ["terminal"] * 4 + ["search"] * 4
    assert all(r["metadata"]["opd_teacher"] == r["metadata"]["domain"] for r in rows)
    assert rows[0]["metadata"]["task_id"] == "t0" and rows[-1]["metadata"]["answer"] == "!"


@pytest.mark.skipif(not _SOURCE.is_dir(), reason="needs the pinned Prime-RL source on weka")
def test_pure_opd_train_command_with_both_domains(monkeypatch, tmp_path):
    freeze_environment(monkeypatch)
    recording = install_command_recorder(monkeypatch)
    started = []
    monkeypatch.setattr(opd_teachers, "start", lambda servers, **kwargs: started.append((servers, kwargs)))
    module = import_launch_script(_EXAMPLE / "run_mopd_qwen3_5.py")
    fields = {
        "terminal_gateway_url": "http://127.0.0.1:24750",
        "search_gateway_url": "http://gateway:1212",
        "num_nodes": 1,
        "num_gpus_per_node": 4,
        "async_rollout_gpus": 1,
        "trace_dir": "/traces",
        "exp_name": "mopd",
    }
    call_entrypoint(module, "execute", fields, tmp_path)
    (submit,) = [shlex.split(c) for c in recording.commands if "ray job submit" in c]
    (runtime_env,) = [t.removeprefix("--runtime-env-json=") for t in submit if t.startswith("--runtime-env-json=")]
    env = json.loads(runtime_env)["env_vars"]

    def flag(name):
        return submit[submit.index(name) + 1]

    assert flag("--custom-generate-function-path") == "generate.generate_with_teacher"
    assert flag("--custom-agent-function-path") == "mopd_agent_function.run"
    assert flag("--custom-rm-path") == "mopd_reward.reward_func"
    assert flag("--dynamic-sampling-filter-path") == "mopd_metrics.missing_reward_filter"
    assert flag("--custom-reward-post-process-path") == "generate.opd_pure_rewards"
    assert flag("--opd-type") == "sglang" and flag("--policy-loss-type") == "dppo_tv"
    assert flag("--max-seq-len") == "131072" and flag("--rollout-max-response-len") == "32768"
    assert "--fp32-lm-head" in submit and "--sglang-enable-fp32-lm-head" in submit
    assert flag("--num-gpus-per-node") == "2", "the two teacher servers hold the node's other GPUs"
    assert flag("--actor-num-gpus-per-node") == "1" and flag("--rollout-num-gpus") == "1"

    assert json.loads(env["MOPD_BUDGETS"])["terminal"]["max_seq_len"] == 67584
    assert env["TMAX_GATEWAY_URL"] == "http://127.0.0.1:24750" and env["TMAX_START_FAILURE"] == "drop"
    assert env["TMAX_TRACE_DIR"] == "/traces/mopd/terminal" and env["SEARCH_TRACE_DIR"] == "/traces/mopd/search"
    assert json.loads(env["SEARCH_ENV_ARGS"])["judge_server_url"] == "http://gateway:1212/v1"
    assert env["TMAX_OPD_TEACHER_URLS_FILE"].endswith("/mopd/opd-teachers.json")
    assert str(_SOURCE / "environments/search_agent") in env["PYTHONPATH"].split(":")

    ((servers, kwargs),) = started
    assert [domain for domain, _ in servers] == ["terminal", "search"]
    assert kwargs["context_length"] == 131072 + 2048 and kwargs["server_args"] == ("--enable-fp32-lm-head",)
    assert kwargs["urls_file"] == env["TMAX_OPD_TEACHER_URLS_FILE"]
