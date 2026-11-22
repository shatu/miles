"""Multi-teacher OPD's episode loop (``--custom-agent-function-path mopd_agent_function.run``): each prompt's domain's.

A prompt's ``metadata["domain"]`` (run_mopd_qwen3_5's prompt data) picks the loop: ``terminal`` runs TMax's
(tmax_agent_function.run, a task in a sandbox, the reward from its tests), ``search`` the search port's
(search_agent_function.run, which leaves the reward to the judge in mopd_reward). Each domain keeps its own token
budget: ``MOPD_BUDGETS`` (the launcher's) gives its episode budget (``max_seq_len``, the prompt included) and per-turn
cap (``max_tokens``), which replace Miles' job-wide ``--max-seq-len`` and ``--rollout-max-response-len``.
"""

import json
import os
from functools import cache
from typing import Any

import search_agent_function
import tmax_agent_function

# Miles calls the agent module's ``abort`` when over-sampling ends a step early: TMax's stops its sandboxes' episodes.
# The search loop has none and runs its episodes out.
from tmax_agent_function import abort  # noqa: F401

_AGENTS = {"terminal": tmax_agent_function.run, "search": search_agent_function.run}


async def run(
    base_url: str,
    prompt: Any,
    request_kwargs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any] | None:
    domain = metadata["domain"]
    budget = budgets()[domain]
    metadata = {**metadata, "max_seq_len": budget["max_seq_len"]}
    request_kwargs = {**(request_kwargs or {}), "max_tokens": budget["max_tokens"]}
    return await _AGENTS[domain](base_url, prompt, request_kwargs, metadata, **kwargs)


@cache
def budgets() -> dict[str, dict[str, int]]:
    """``{domain: {"max_seq_len": ..., "max_tokens": ...}}``, from ``MOPD_BUDGETS``."""
    return json.loads(os.environ["MOPD_BUDGETS"])
