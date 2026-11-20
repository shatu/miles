"""Multi-teacher OPD's task reward (``--custom-rm-path mopd_reward.reward_func``): each prompt's domain's.

``terminal``: the sandbox's test verdict, which the episode left in ``metadata["reward"]`` (generate.reward_func);
``search``: the judge and Prime-RL's penalties (search_reward.reward_func). Pure OPD trains on neither: the reward is
logged (``rollout/raw_reward`` and the per-domain metrics) and a missing one (``None``) drops its group.
"""

import asyncio

import generate
import search_reward

from miles.utils.types import Sample

_REWARDS = {"terminal": generate.reward_func, "search": search_reward.reward_func}


async def reward_func(args, samples: Sample | list[Sample], **kwargs) -> float | None | list[float | None]:
    if isinstance(samples, list):
        return list(await asyncio.gather(*(_reward(args, s) for s in samples)))
    return await _reward(args, samples)


async def _reward(args, sample: Sample) -> float | None:
    return await _REWARDS[sample.metadata["domain"]](args, sample)
