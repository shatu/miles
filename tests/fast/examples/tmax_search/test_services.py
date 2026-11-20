"""The search services' Beaker tasks (examples/tmax_search/beaker/services.py): priorities and minimum runtimes."""

import sys

import pytest

from tests.fast.examples.tmax_search import conftest_paths

sys.path.insert(0, str(conftest_paths.REPO / "examples" / "tmax_search" / "beaker"))

import services  # noqa: E402

pytestmark = conftest_paths.needs_prime_rl_source


def _tasks(**context):
    source = conftest_paths.PRIME_RL_SOURCE
    return services.service_tasks(
        prime_rl_source=str(source),
        prime_rl_run_yaml=str(source / conftest_paths.PRIME_RL_CONFIG.rsplit("/", 1)[0] / "run.yaml"),
        run_dir="/weka/run",
        state_dir="/weka/run/state",
        identity="test-run",
        hf_token_secret="user_HF_TOKEN",
        context={"priority": "high", "min_runtime": "8h", **context},
        replicas={"terminal": 2, "localsearch": 2},
        nvme_cache_dir="/nvme/test",
    )


def test_the_tasks_take_the_runs_priority_and_only_the_judge_its_minimum_runtime():
    contexts = {task["name"]: task["context"] for task in _tasks()}
    assert {c["priority"] for c in contexts.values()} == {"high"}
    assert contexts["judge"]["minRuntime"] == "8h"
    assert all(c["minRuntime"] == "0h" for name, c in contexts.items() if name != "judge")


def test_a_task_can_get_its_own_priority():
    """The judge's Holmes GPU at urgent: at high, an immediate-priority job preempted it (01M7VJDZ5AG2X0PHKFDEZ2JC5T)."""
    contexts = {task["name"]: task["context"] for task in _tasks(task_priority={"judge": "urgent"})}
    assert contexts["judge"]["priority"] == "urgent"
    assert {c["priority"] for name, c in contexts.items() if name != "judge"} == {"high"}


def test_a_priority_for_an_unknown_task_is_an_error():
    with pytest.raises(AssertionError, match="names no service task"):
        _tasks(task_priority={"juge": "urgent"})
