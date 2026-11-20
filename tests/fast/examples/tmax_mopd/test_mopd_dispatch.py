"""Multi-teacher OPD's per-domain episode loop, reward and metrics (mopd_agent_function, mopd_reward, mopd_metrics)."""

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.fast.examples.tmax_search import conftest_paths

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "examples" / "tmax_mopd"))

pytestmark = conftest_paths.needs_prime_rl_source

if conftest_paths.PRIME_RL_SOURCE.is_dir():
    import mopd_agent_function
    import mopd_metrics
    import mopd_reward

from miles.utils.types import Sample  # noqa: E402

ARGS = SimpleNamespace(
    reward_key=None,
    rollout_batch_size=2,
    n_samples_per_prompt=2,
    wandb_always_use_train_step=False,
    custom_reward_post_process_path="generate.opd_pure_rewards",
)
BUDGETS = {
    "terminal": {"max_seq_len": 67584, "max_tokens": 16384},
    "search": {"max_seq_len": 131072, "max_tokens": 32768},
}


def test_each_episode_runs_its_domains_loop_with_its_budget(monkeypatch):
    calls = []

    def agent(domain):
        async def run(base_url, prompt, request_kwargs, metadata, **kwargs):
            calls.append((domain, request_kwargs, metadata))
            return {"domain": domain}

        return run

    monkeypatch.setattr(mopd_agent_function, "_AGENTS", {d: agent(d) for d in BUDGETS})
    monkeypatch.setenv("MOPD_BUDGETS", json.dumps(BUDGETS))
    mopd_agent_function.budgets.cache_clear()
    for domain in ("terminal", "search"):
        # Miles passes its job-wide budget and per-turn cap; the domain's replace them.
        metadata = {"domain": domain, "max_seq_len": 131072, "task_id": "x"}
        result = asyncio.run(mopd_agent_function.run("http://s", [], {"max_tokens": 32768, "top_p": 1.0}, metadata))
        assert result == {"domain": domain}
        _, request_kwargs, seen = calls[-1]
        assert seen["max_seq_len"] == BUDGETS[domain]["max_seq_len"] and seen["task_id"] == "x"
        assert request_kwargs == {"max_tokens": BUDGETS[domain]["max_tokens"], "top_p": 1.0}
    assert [c[0] for c in calls] == ["terminal", "search"]
    mopd_agent_function.budgets.cache_clear()


def test_the_abort_hook_is_the_terminal_agents():
    """Miles looks for ``abort`` next to the agent function; TMax's stops its sandboxes' episodes."""
    assert mopd_agent_function.abort is mopd_agent_function.tmax_agent_function.abort


def test_each_sample_gets_its_domains_reward(monkeypatch):
    def fixed(value):
        async def reward(args, sample):
            return value

        return reward

    rewards = {"terminal": fixed(1.0), "search": fixed(0.5)}
    monkeypatch.setattr(mopd_reward, "_REWARDS", rewards)
    samples = [Sample(metadata={"domain": "search"}), Sample(metadata={"domain": "terminal"})]
    assert asyncio.run(mopd_reward.reward_func(ARGS, samples)) == [0.5, 1.0]
    assert asyncio.run(mopd_reward.reward_func(ARGS, samples[1])) == 1.0


def _terminal_group(index, rewards):
    agent = {"turns": 3, "tool_calls": 3, "end_reason": "submitted", "submitted": 1.0}
    return [
        Sample(
            group_index=index, reward=r, response_length=10, metadata={"domain": "terminal", "agent_metrics": agent}
        )
        for r in rewards
    ]


def _search_group(index, rewards):
    agent = {"turns": 5, "tool_calls": 4, "stop_condition": "no_tools"}
    return [
        Sample(
            group_index=index,
            reward=r,
            response_length=20,
            metadata={
                "domain": "search",
                "agent_metrics": agent,
                "judge": {"valid": r is not None, "invoked": True},
                "reward_components": (
                    None
                    if r is None
                    else {
                        "judge_answer": r,
                        "tool_stderr_penalty_reward": 0.0,
                        "missing_final_answer_penalty_reward": 0.0,
                    }
                ),
            },
        )
        for r in rewards
    ]


def test_metrics_per_domain(monkeypatch):
    mopd_metrics.rollout_metrics(ARGS, 0, [])  # start empty
    terminal, search, judge_failed = (
        _terminal_group(0, [1.0, 0.0]),
        _search_group(1, [1.0, 1.0]),
        _search_group(2, [1.0, None]),
    )
    assert mopd_metrics.missing_reward_filter(ARGS, terminal).keep
    assert mopd_metrics.missing_reward_filter(ARGS, search).keep, "pure OPD keeps uniform groups"
    assert not mopd_metrics.missing_reward_filter(ARGS, judge_failed).keep

    metrics = mopd_metrics.rollout_metrics(ARGS, 4, terminal + search)
    assert metrics["rollout/terminal/agent/episodes"] == 2
    assert metrics["rollout/search/agent/episodes"] == 4, "the dropped group is described too"
    assert metrics["rollout/terminal/unfiltered/solve_rate_mean"] == pytest.approx(0.5)
    assert metrics["rollout/search/judge/invalid_rate"] == pytest.approx(1 / 4)
    assert metrics["rollout/terminal/trained/fraction"] == metrics["rollout/search/trained/fraction"] == 0.5
    assert metrics["rollout/trained/samples_total"] == 5 * 2 * 2, "every domain's samples, logged once"
    assert not any(k.endswith("/trained/samples_total") and k != "rollout/trained/samples_total" for k in metrics)
    assert metrics["rollout/search/trained/response_len_solved_mean"] == 20
