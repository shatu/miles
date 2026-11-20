---
title: "Multi-teacher OPD: one student from the terminal and search agents"
description: "Multi-teacher on-policy distillation of the terminal and search agents into one student, each episode scored by its domain's teacher."
# Generated from examples/tmax_mopd/README.md by scripts/tools/sync_example_docs.py. Edit that README, not this file.
---
Distills the TMax terminal agent and the search agent into one Qwen3.5 student with on-policy distillation from
two teachers (MOPD). The student (default `hamishivi/Qwen3.5-4B`) trains on a mix of both agents' prompts: the
[TMax](/examples/tmax) terminal tasks, each run in a sandbox, and the [search port](/examples/tmax-search)'s
questions, each run against Prime-RL's search services. Each episode is scored token by token by its own domain's
teacher:

| Domain | Prompts | Episode loop | Task reward (logged) | Teacher (default) |
|---|---|---|---|---|
| `terminal` | `allenai/tmax-15k-open-instruct`, 14.6k tasks | `tmax_agent_function.run` | the task's tests | `allenai/tmax-4b` |
| `search` | the Prime-RL experiment's questions (bc-rl-1131: 1,131) | `search_agent_function.run` | the judge, minus penalties | the search port's 4B run, step 280 |

Training is pure OPD: the advantage of each response token is `-opd_kl_coef * (log p_student - log p_teacher)`,
and the task rewards are computed and logged but not trained on.

## How it works

- **Prompt mix** (`prepare`). The launcher writes both launchers' prompt files (`run_tmax_qwen3_5.prepare_tasks`,
  `run_search_qwen3_5.write_prompt_data`) and mixes them. Every row is tagged with `metadata.domain` and
  `metadata.opd_teacher`. Each terminal task appears once. Each search question is repeated in whole copies until
  `search_fraction` of the rows are search rows (0.5 by default: 13 copies, 50.2%). `--rollout-shuffle` then draws
  batches from the mix at random.
- **Episodes** (`mopd_agent_function.run`). Each episode runs its domain's loop with that domain's budget, both
  passed in `MOPD_BUDGETS`:
  - terminal: 64k tokens per episode and 16k per turn, as in the 4B TMax runs;
  - search: 128k and 32k, as in Prime-RL.

  Miles' `--max-seq-len` (the session server's cap) is the longer of the two. The module re-exports TMax's `abort`,
  which Miles calls when over-sampling ends a step early.
- **Teachers** (`examples/tmax/opd_teachers.py`). Before the job starts, the launcher starts one SGLang server per
  teacher GPU (`terminal_teacher_gpus`, `search_teacher_gpus`) on the rollout nodes, with fp32 logits
  (`--enable-fp32-lm-head`). It lists the servers by name in `opd-teachers.json`. `generate.generate_with_teacher`
  sends each sample to the least busy server of the teacher named in its `metadata.opd_teacher`.
- **Search rewards come later.** Search samples reach the teacher before they have a reward: Miles runs the judge
  (`--custom-rm-path`) after the generate function. The wrapper therefore scores every sample except aborted ones and
  those whose agent reported the reward missing (`reward: None`, which only the terminal agent reports).
- **Reward and filter.** `mopd_reward` applies each domain's reward (terminal: the sandbox verdict; search: the judge).
  `mopd_metrics.missing_reward_filter` drops a group whose reward is missing: a sandbox the fleet lost, or an invalid
  judgment. It keeps uniform groups, which carry the full OPD signal.
- **Metrics.** `mopd_metrics` logs each domain's own metrics under `rollout/terminal/*` (as `tmax_metrics`) and
  `rollout/search/*` (as `search_metrics`), plus each domain's share of the trained samples
  (`rollout/<domain>/trained/fraction`). Miles' own `rollout/raw_reward` averages over both domains.
- **Training** follows the TMax OPD runs (`prod_4node_gateway_opd_9b_to_4b.yaml`): DPPO with threshold 0.1 anchored
  on the rollout log-probs, per-token loss, AdamW at lr 1e-6, fully async with staleness 4. Like the search recipe,
  it uses fp32 logits on both sides (`--fp32-lm-head`, `--sglang-enable-fp32-lm-head`).
- **Beaker** (`examples/tmax/beaker/submit.py`). One experiment runs both environments:
  - `gateway_registry` makes the head start a LiteRegistry gateway in front of the podman fleet for the terminal
    sandboxes;
  - `search_services` adds Prime-RL's services as tasks: the judge on a Holmes GPU, and the search and terminal
    replicas on Jupiter CPUs.

  A config that sets `sandbox_backend` keeps it alongside the search services.

## Configs

| Config | Layout | Use |
|---|---|---|
| `configs/smoke_mopd_4b.yaml` | 1 Holmes node: 1 training GPU, 1 rollout engine and 2 teacher GPUs, plus the judge's GPU; 3 steps of 4 x 4 | smoke |
| `configs/prod_4node_mopd_4b.yaml` | 4 Holmes nodes: 16 training GPUs, 14 rollout engines and 2 teacher GPUs; 8 x 32 per step, 500 steps | production |

```bash
python examples/tmax/beaker/submit.py examples/tmax_mopd/configs/smoke_mopd_4b.yaml
```

Local (with the services and a fleet gateway already running):

```bash
python examples/tmax_mopd/run_mopd_qwen3_5.py prepare examples/tmax_mopd/configs/smoke_mopd_4b.yaml
TMAX_GATEWAY_URL=... SEARCH_GATEWAY_URL=... \
    python examples/tmax_mopd/run_mopd_qwen3_5.py execute examples/tmax_mopd/configs/smoke_mopd_4b.yaml
```

## Open questions

- **Loss weighting.** The per-token loss weighs each token equally, so long search episodes (up to 128k tokens)
  outweigh terminal episodes (up to 64k). The search recipe's `group_token_mean` would weigh each group equally
  instead.
- **Environment drift for the search teacher.** The search teacher was trained on Prime-RL's bc-rl-v1-1 environment.
  The default search prompts are bc-rl-1131's, a subset served from the no-HF corpus. The teacher scores the
  student's episodes in an environment close to its own, but not the same one. `prime_rl_config` switches it.
- **Teacher capacity.** Each teacher has one server, and each scores about half the samples. The terminal teacher in
  the 9B-to-4B run had two servers for all of them, and was a 9B. Raise `*_teacher_gpus` if
  `perf/rollout_time` shows the teachers lagging.
