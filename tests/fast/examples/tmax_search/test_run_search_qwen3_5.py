"""The search launcher's train command: Prime-RL's recipe as Miles flags (run_search_qwen3_5.py)."""

import json
import shlex
import sys
from pathlib import Path

import pytest

_EXAMPLE = Path(__file__).resolve().parents[4] / "examples" / "tmax_search"
sys.path.insert(0, str(_EXAMPLE))

from tests.fast.launch_scripts.py_harness import (  # noqa: E402
    call_entrypoint,
    freeze_environment,
    import_launch_script,
    install_command_recorder,
)

# The launcher's default source: the 2026-11-18 sealed copy, whose experiment is search-agent-bc-rl-1131-fp32-lr5e-7.
_SOURCE = Path(
    "/weka/oe-adapt-default/shashankg/miles_search/prime-rl-sources/"
    "ebe1abbae42ec6587ff36647fca2b007583abd4501e45e0be15a37a3e1495160/source"
)
pytestmark = pytest.mark.skipif(not _SOURCE.is_dir(), reason="needs the pinned Prime-RL source on weka")


def _record(monkeypatch, tmp_path, **overrides) -> tuple[list[str], dict]:
    freeze_environment(monkeypatch)
    recording = install_command_recorder(monkeypatch)
    launcher = import_launch_script(_EXAMPLE / "run_search_qwen3_5.py")
    fields = {"gateway_url": "http://gateway:1212", "num_nodes": 1, "num_gpus_per_node": 8, "async_rollout_gpus": 4}
    call_entrypoint(launcher, "execute", fields | overrides, tmp_path)
    (submit,) = [shlex.split(c) for c in recording.commands if "ray job submit" in c]
    (runtime_env,) = [t.removeprefix("--runtime-env-json=") for t in submit if t.startswith("--runtime-env-json=")]
    return submit, json.loads(runtime_env)["env_vars"]


def _flag(argv: list[str], name: str) -> str:
    return argv[argv.index(name) + 1]


def test_prime_rl_recipe_flags(monkeypatch, tmp_path):
    argv, env = _record(monkeypatch, tmp_path)
    assert _flag(argv, "--policy-loss-type") == "dppo_tv" and _flag(argv, "--dppo-tv-threshold") == "0.2"
    assert _flag(argv, "--lr") == "5e-07" and _flag(argv, "--adam-eps") == "1e-12"
    assert _flag(argv, "--adam-beta1") == "0.95" and _flag(argv, "--adam-beta2") == "0.95"
    # group_token_mean on the sample-mean reducer, and Prime-RL's masking: no zero-variance filter
    assert _flag(argv, "--custom-convert-samples-to-train-data-path") == "search_loss.convert_samples_to_train_data"
    assert "--calculate-per-token-loss" not in argv
    assert _flag(argv, "--dynamic-sampling-filter-path") == "search_metrics.missing_reward_filter"
    assert _flag(argv, "--rollout-batch-size") == "8" and _flag(argv, "--n-samples-per-prompt") == "32"
    assert _flag(argv, "--max-seq-len") == "131072" and _flag(argv, "--rollout-max-response-len") == "32768"
    assert "--fp32-lm-head" in argv and "--sglang-enable-fp32-lm-head" in argv

    resolved = json.loads(env["SEARCH_ENV_ARGS"])
    assert resolved["search_server_url"] == "http://gateway:1212/search"
    assert resolved["judge_server_url"] == "http://gateway:1212/v1"
    assert resolved["max_tool_calls"] == 150 and resolved["max_turns"] == 151
    assert resolved["max_total_completion_tokens"] == 131072
    assert resolved["final_warning"].startswith("IMPORTANT: This was your last allowed tool call.")
    assert str(_SOURCE / "environments/search_agent") in env["PYTHONPATH"].split(":")
    assert resolved["dataset"].endswith("search_agents/bc-rl-subset/train.jsonl")
    assert resolved["local_search_model_path"] == "localsearch:bc-rl-v1-no-hf"
    assert resolved["penalize_truncated_final_turn"]
    assert _flag(argv, "--hf-checkpoint").endswith("/Qwen3.5-4B-tmax-template")  # hamishivi/Qwen3.5-4B


def test_bf16_head_and_per_token_loss(monkeypatch, tmp_path):
    argv, _ = _record(monkeypatch, tmp_path, fp32_lm_head=False, group_token_mean=False)
    assert "--fp32-lm-head" not in argv and "--sglang-enable-fp32-lm-head" not in argv
    assert "--calculate-per-token-loss" in argv and "--custom-convert-samples-to-train-data-path" not in argv
