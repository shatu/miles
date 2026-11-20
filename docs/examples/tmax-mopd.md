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
| `search` | the Prime-RL experiment's questions (default bc-rl-v1-1, the teacher's own: 1,215) | `search_agent_function.run` | the judge, minus penalties | the search port's 4B run, step 280 |

Training is pure OPD: the advantage of each response token is `-opd_kl_coef * (log p_student - log p_teacher)`,
and the task rewards are computed and logged but not trained on.

## How it works

- **Prompt mix** (`prepare`). The launcher writes both launchers' prompt files (`run_tmax_qwen3_5.prepare_tasks`,
  `run_search_qwen3_5.write_prompt_data`) and mixes them. Every row is tagged with `metadata.domain` and
  `metadata.opd_teacher`. Each terminal task appears once. Each search question is repeated in whole copies until
  `search_fraction` of the rows are search rows (0.5 by default: 12 copies of bc-rl-v1-1's questions, 50.0%). `--rollout-shuffle` then draws
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
- **Training** uses the TMax OPD runs' optimizer and loss settings, not their teachers (both teachers here are 4B):
  DPPO with threshold 0.1 anchored on the rollout log-probs, per-token loss, AdamW at lr 1e-6, fully async with
  staleness 4. Like the search recipe, it uses fp32 logits on both sides (`--fp32-lm-head`,
  `--sglang-enable-fp32-lm-head`).
- **Beaker** (`examples/tmax/beaker/submit.py`). One experiment runs both environments:
  - `gateway_registry` makes the head start a LiteRegistry gateway in front of the podman fleet for the terminal
    sandboxes;
  - `search_services` adds Prime-RL's services as tasks: the judge on a Holmes GPU, and the search and terminal
    replicas on Jupiter CPUs.

  A config that sets `sandbox_backend` keeps it alongside the search services.

## Configs

| Config | Layout | Use |
|---|---|---|
| `configs/smoke_mopd_4b.yaml` | 1 Holmes node: 1 training GPU, 1 rollout engine and 2 teacher GPUs, plus the judge's GPU; 3 steps of 4 x 4 on bc-rl-1131 | smoke (passed) |
| `configs/prod_4node_mopd_4b.yaml` | 4 Holmes nodes: 16 training GPUs, 13 rollout engines, 1 terminal and 2 search teacher GPUs; 8 x 32 per step on bc-rl-v1-1, 500 steps | production |

```bash
python examples/tmax/beaker/submit.py examples/tmax_mopd/configs/smoke_mopd_4b.yaml
```

Local (with the services and a fleet gateway already running):

```bash
python examples/tmax_mopd/run_mopd_qwen3_5.py prepare examples/tmax_mopd/configs/smoke_mopd_4b.yaml
TMAX_GATEWAY_URL=... SEARCH_GATEWAY_URL=... \
    python examples/tmax_mopd/run_mopd_qwen3_5.py execute examples/tmax_mopd/configs/smoke_mopd_4b.yaml
```

## Smoke

`smoke_mopd_4b`, experiment `01M7VBC414YCQY28AEBWMQHH6Q` (2026-11-20, bc-rl-1131, one server per teacher): 3 steps,
every task exited 0. Every trained sample carried its own domain's teacher log-probs, one per response token.

| Step | Trained batch | OPD reverse KL | Train/rollout log-prob gap | Grad norm |
|---|---|---|---|---|
| 0 | 16 search (all judged correct) | 0.058 | 0.0067 | 5.0 |
| 1 | 12 search (4 correct) + 4 terminal | 0.078 | 0.0075 | 3.5 |
| 2 | 8 search (4 correct) + 8 terminal | 0.134 | 0.0154 | 3.7 |

Fully async fills the first batches with the groups that finish first, here search groups. The judge scored 29
episodes, all validly, 21 of them correct.

What the smoke's samples show, per domain:

| Domain | Trained tokens per sample | Scored tokens per sample (tool output included) | Per-token reverse KL |
|---|---|---|---|
| terminal | 12.2k | ~23.5k | 0.206 |
| search | 8.8k | ~40k | 0.061 |

- **Loss weighting.** Tool output is masked out of the loss, so search samples train on fewer tokens than terminal
  ones, and at a smaller KL. Per sample, terminal carries about 4.7 times the OPD signal. That gap is real: the search
  teacher is the student's own weights after 280 RL steps.
- **Teacher load.** Search samples are mostly tool output, which the teacher scores. The search teacher therefore
  does about 60% of the scoring, and the prod config gives it two servers to the terminal teacher's one.

## Open questions

- **The mix.** If the logged search reward (`rollout/search/judge/judge_answer_mean`) falls below where the teacher's
  run started, raise `search_fraction`. Watch `rollout/<domain>/trained/fraction` too: slow terminal groups are likelier
  to be dropped as stale, which skews training toward search.
- **Environment.** bc-rl-1131 (the newer Prime-RL experiment, on a corpus without Hugging Face pages) would want a
  search teacher trained there. The teacher saw bc-rl-v1-1's questions about twice in RL, so the student also learns
  memorized answers; only held-out evals show real gains.
- **Teacher capacity.** The server count comes from an estimate, not a measurement. Check how busy each teacher is in
  its `opd-teacher-<i>.log` early in a run. Teachers serving bf16 logits would be faster; in the search smokes, bf16
  rather than fp32 moved log-probs by only about 0.003.
- **Search evaluation.** The Miles search port does not run Prime-RL's frozen BrowseComp-Plus validation. Judging the
  merged model needs a search eval of its HF snapshots next to TBlite and TB2.1.
