"""Multi-teacher on-policy distillation (MOPD): one Qwen3.5 student from the TMax terminal agent and the search agent.

=====================

Distills two teachers into one student (default hamishivi/Qwen3.5-4B) on a mix of both agents' prompts: the TMax
terminal tasks (examples/tmax, a task in a sandbox) and the search port's questions (examples/tmax_search, Prime-RL's
search environment). Every episode runs its domain's loop (mopd_agent_function), and its domain's teacher scores every
response token on its own SGLang servers (generate.generate_with_teacher, opd_teachers): by default allenai/tmax-4b for
terminal episodes and the search port's 4B run at step 280 for search episodes. Pure OPD: the advantage is
-opd_kl_coef * (log p_student - log p_teacher) per token; each domain's task reward (the sandbox's tests, the search
judge) is computed and logged, not trained on, and a missing one drops its group.

Each domain keeps its recipe's episode budget and per-turn cap (terminal: 64k and 16k as the 4B TMax runs, search:
128k and 32k as Prime-RL's); training is the TMax OPD runs' (DPPO 0.1, per-token loss, AdamW 1e-6), with fp32 logits
on both sides as the search recipe.

Requires what both recipes need: a podman fleet's gateway for the terminal sandboxes (TMAX_GATEWAY_URL, which
beaker/tmax_beaker_entry.sh exports for a config's beaker.gateway_registry) and the search, terminal and judge services
behind their gateway (SEARCH_GATEWAY_URL, exported for a config's beaker.search_services), plus the pinned Prime-RL
source (--prime-rl-source).

`prepare` writes both domains' prompts and their mix, downloads the student and the teachers and converts the student
to Megatron torch_dist. `execute` starts the teacher servers and submits the training job.

=====================

Args:
  --model-name: the student, a key of the TMax launcher's RECIPES (default: Qwen3.5-4B-tmax-template).
  --terminal-teacher / --search-teacher: each domain's teacher, a key of the same RECIPES (default: tmax-4b /
      search-4b-bcrl-v11-step279).
  --terminal-teacher-gpus / --search-teacher-gpus: SGLang servers of one GPU each per teacher, on the rollout nodes
      (default: 1 / 1). They come out of the GPUs async_rollout_gpus does not take.
  --search-fraction: the search questions' share of the prompts (default: 0.5); each question is repeated to reach it.
  --terminal-max-seq-len / --terminal-max-response-len: terminal episode budget and per-turn cap (default: 67584 /
      16384).
  --search-max-seq-len / --search-max-response-len: search episode budget and per-turn cap (default: 131072 / 32768).
  --opd-kl-coef: weight of the per-token reverse KL (default: 1.0).
  --rollout-batch-size / --n-samples-per-prompt: prompts x samples per step (default: 8 x 32).
  --async-rollout-gpus: rollout GPUs; training gets the rest less the teacher servers', in whole nodes (default: 14).
  Terminal sandboxes (as run_tmax_qwen3_5.py, with its defaults): --sandbox-backend, --terminal-gateway-url,
      --docker-host, --max-steps, --sandbox-start-concurrency, --sandbox-exec-concurrency, --sandboxes-per-node,
      --start-failure, --episode-timeout-s, --tool-call-timeout-s.
  Search environment (as run_search_qwen3_5.py): --prime-rl-source, --prime-rl-config (default: the default search
      teacher's own, bc-rl-v1-1 from the source its run pinned), --prime-rl-checkout, --prime-rl-weka-mount,
      --search-gateway-url.
  --trace-dir: episode traces under <trace-dir>/<exp-name>/{terminal,search} (default: off).
  The rest as run_tmax_qwen3_5.py.

=====================

    python examples/tmax_mopd/run_mopd_qwen3_5.py prepare examples/tmax_mopd/configs/smoke_mopd_4b.yaml
    python examples/tmax_mopd/run_mopd_qwen3_5.py execute examples/tmax_mopd/configs/smoke_mopd_4b.yaml
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import typer

_SCRIPT_DIR = Path(__file__).resolve().parent
_TMAX_DIR = _SCRIPT_DIR.parent / "tmax"
_SEARCH_DIR = _SCRIPT_DIR.parent / "tmax_search"
sys.path[:0] = [str(_TMAX_DIR), str(_SEARCH_DIR)]
import opd_teachers  # noqa: E402
import run_search_qwen3_5 as search_launcher  # noqa: E402
import run_tmax_qwen3_5 as tmax_launcher  # noqa: E402
from launch_common import apply_yaml_config, fetch_command, wandb_args  # noqa: E402
from search_config import ENV_ARGS_VAR  # noqa: E402

from miles.utils.external_utils import command_utils  # noqa: E402

DOMAINS = ("terminal", "search")
_TMAX = tmax_launcher.ScriptArgs  # its fields' defaults are the terminal agent's here too
_SEARCH = search_launcher.ScriptArgs

app = typer.Typer()


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    run_id: str = command_utils.create_run_id()
    model_name: tmax_launcher.MODEL_NAMES = "Qwen3.5-4B-tmax-template"
    terminal_teacher: tmax_launcher.MODEL_NAMES = "tmax-4b"
    search_teacher: tmax_launcher.MODEL_NAMES = "search-4b-bcrl-v11-step279"
    terminal_teacher_gpus: int = 1
    search_teacher_gpus: int = 1
    opd_kl_coef: float = 1.0
    opd_teacher_timeout_s: int = _TMAX.opd_teacher_timeout_s
    search_fraction: float = 0.5
    # Each domain's episode budget (the prompt included) and per-turn cap: the 4B TMax runs' (open-instruct's
    # --response_length 65536 + --max_prompt_token_length 2048, --per_turn_max_tokens 16384) and Prime-RL's.
    terminal_max_seq_len: int = 67584
    terminal_max_response_len: int = 16384
    search_max_seq_len: int = _SEARCH.max_seq_len
    search_max_response_len: int = _SEARCH.rollout_max_response_len
    # Terminal sandboxes, as run_tmax_qwen3_5.py
    sandbox_backend: Literal["podman", "gateway"] = "gateway"
    terminal_gateway_url: str = ""  # empty inherits TMAX_GATEWAY_URL
    docker_host: str = ""
    max_steps: int = _TMAX.max_steps
    sandbox_start_concurrency: int = _TMAX.sandbox_start_concurrency
    sandbox_exec_concurrency: int = _TMAX.sandbox_exec_concurrency
    sandboxes_per_node: int = _TMAX.sandboxes_per_node
    start_failure: Literal["zero", "drop"] = "drop"  # the OPD runs': the fleet's failure is not the policy's
    episode_timeout_s: int = _TMAX.episode_timeout_s
    tool_call_timeout_s: int = _TMAX.tool_call_timeout_s
    # Search environment, as run_search_qwen3_5.py. By default the default search teacher's own: Prime-RL's bc-rl-v1-1
    # (1,215 questions, the corpus with Hugging Face pages) from the source search_prod4node_4b_bcrl_v11_high8h pinned,
    # so the teacher scores episodes from the environment it was trained in.
    prime_rl_source: str = (
        "/weka/oe-adapt-default/shashankg/miles_search/prime-rl-sources/"
        "a5fcb924645d9bd8f326efd5087c8de6447c83fe17c6d07c36e99ee736d30a37/source"
    )
    prime_rl_config: str = "configs/experiments/search-agent-bc-rl-v1-1-fp32-lr5e-7/rl.toml"
    prime_rl_checkout: str = _SEARCH.prime_rl_checkout
    prime_rl_weka_mount: str = _SEARCH.prime_rl_weka_mount
    search_gateway_url: str = ""  # empty inherits SEARCH_GATEWAY_URL
    # Training: the TMax OPD runs' optimizer and loss settings (prod_4node_gateway_opd_9b_to_4b.yaml's), not their
    # teachers: both teachers here are 4B.
    num_gpus_per_node: int = 8
    rollout_batch_size: int = 8
    n_samples_per_prompt: int = 32
    num_rollout: int = 500
    tensor_model_parallel_size: int = 1
    context_parallel_size: int = 1
    max_tokens_per_gpu: int = 131072
    optimizer_cpu_offload: bool = False
    lr: float = 1e-6
    dppo_threshold: float = 0.1
    fp32_lm_head: bool = True
    async_rollout_gpus: int = 14
    max_weight_staleness: int = 4
    async_max_concurrent_samples: int = 512
    async_data_buffer_capacity_factor: float = 4.0
    sglang_mem_fraction_static: float = 0.8
    log_kl_to_base: bool = True
    save_interval: int = 20
    hf_snapshot_every: int = 20
    trace_dir: str = ""
    dump_training_data: bool = False
    session_server: Literal["v1", "v2"] = "v2"
    wandb_project: str = ""
    wandb_entity: str = ""
    exp_name: str = ""
    extra_args: str = ""
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"

    def __post_init__(self):
        assert self.hf_snapshot_every >= 0 and self.hf_snapshot_every % self.save_interval == 0, (
            f"hf_snapshot_every ({self.hf_snapshot_every}) must be 0 or a multiple of save_interval "
            f"({self.save_interval}): HF snapshots are written at saves"
        )
        assert 0 < self.search_fraction < 1, f"search_fraction {self.search_fraction}: both domains need prompts"
        assert self.async_rollout_gpus > 0, "MOPD trains fully async: the teacher servers sit beside the rollout"
        assert self.terminal_teacher_gpus > 0 and self.search_teacher_gpus > 0, "each teacher needs a server"
        student = tmax_launcher.RECIPES[self.model_name]
        for domain, teacher in self.teachers.items():
            # The teacher scores the student's tokens, rendered with the student's template settings.
            assert (
                tmax_launcher.RECIPES[teacher].enable_thinking == student.enable_thinking
            ), f"the {domain} teacher {teacher} renders with another enable_thinking than {self.model_name}"

    @property
    def recipe(self):
        return tmax_launcher.RECIPES[self.model_name]

    @property
    def teachers(self) -> dict[str, str]:
        """Each domain's teacher (a RECIPES key), as the prompts' ``metadata["opd_teacher"]`` names it."""
        return {"terminal": self.terminal_teacher, "search": self.search_teacher}

    @property
    def teacher_servers(self) -> list[tuple[str, str]]:
        """(domain, HF checkpoint) per SGLang teacher server, one GPU each."""
        gpus = {"terminal": self.terminal_teacher_gpus, "search": self.search_teacher_gpus}
        return [(d, f"{self.model_dir}/{self.teachers[d]}") for d in DOMAINS for _ in range(gpus[d])]

    @property
    def max_seq_len(self) -> int:
        """Miles' --max-seq-len, the session server's cap: the longer domain's budget."""
        return max(self.terminal_max_seq_len, self.search_max_seq_len)

    @property
    def budgets(self) -> dict[str, dict[str, int]]:
        """mopd_agent_function's MOPD_BUDGETS."""
        return {
            "terminal": {"max_seq_len": self.terminal_max_seq_len, "max_tokens": self.terminal_max_response_len},
            "search": {"max_seq_len": self.search_max_seq_len, "max_tokens": self.search_max_response_len},
        }

    @property
    def terminal(self) -> tmax_launcher.ScriptArgs:
        """The TMax launcher's view of the terminal side: its task data, prompts and agent settings."""
        return tmax_launcher.ScriptArgs(
            model_name=self.model_name,
            data_dir=self.data_dir,
            sandbox_backend=self.sandbox_backend,
            gateway_url=self.terminal_gateway_url,
            docker_host=self.docker_host,
            max_steps=self.max_steps,
            sandbox_start_concurrency=self.sandbox_start_concurrency,
            sandbox_exec_concurrency=self.sandbox_exec_concurrency,
            sandboxes_per_node=self.sandboxes_per_node,
            start_failure=self.start_failure,
            episode_timeout_s=self.episode_timeout_s,
            tool_call_timeout_s=self.tool_call_timeout_s,
        )

    @property
    def search(self) -> search_launcher.ScriptArgs:
        """The search launcher's view of the search side: its prompts and environment."""
        return search_launcher.ScriptArgs(
            model_name=self.model_name,
            data_dir=self.data_dir,
            prime_rl_source=self.prime_rl_source,
            prime_rl_config=self.prime_rl_config,
            prime_rl_checkout=self.prime_rl_checkout,
            prime_rl_weka_mount=self.prime_rl_weka_mount,
            gateway_url=self.search_gateway_url,
        )

    @property
    def checkpoint_dir(self) -> str:
        return f"{self.output_dir}/{self.exp_name or self.run_id}/checkpoints"

    @property
    def trace_run_dir(self) -> str:
        return f"{self.trace_dir}/{self.exp_name or self.run_id}" if self.trace_dir else ""

    def domain_trace_dir(self, domain: str) -> str:
        return f"{self.trace_run_dir}/{domain}" if self.trace_run_dir else ""

    @property
    def opd_teacher_urls_file(self) -> str:
        """Where opd_teachers lists the teacher servers for generate.generate_with_teacher."""
        return f"{self.output_dir}/{self.exp_name or self.run_id}/opd-teachers.json"

    @property
    def prompt_data(self) -> str:
        terminal, search = (Path(p).stem for p in (self.terminal.prompt_data, self.search.prompt_data))
        return f"{self.data_dir}/mopd/{terminal}+{search}_search{self.search_fraction:g}.jsonl"

    @property
    def train_layout(self) -> tuple[int, int]:
        """(actor nodes, actor GPUs per node): one split node, or whole nodes for each side, training first; the
        teacher servers' GPUs are the rollout side's."""
        train_gpus = self.num_nodes * self.num_gpus_per_node - self.async_rollout_gpus - len(self.teacher_servers)
        assert train_gpus > 0, (
            f"async_rollout_gpus={self.async_rollout_gpus} and {len(self.teacher_servers)} teacher GPUs leave no GPU "
            "for training"
        )
        if self.num_nodes == 1:
            return 1, train_gpus
        assert train_gpus % self.num_gpus_per_node == 0, (
            f"on {self.num_nodes} nodes, async_rollout_gpus + the teacher GPUs must be whole nodes of "
            f"{self.num_gpus_per_node} GPUs"
        )
        return train_gpus // self.num_gpus_per_node, self.num_gpus_per_node


@app.command()
@command_utils.dataclass_cli
def prepare(args: ScriptArgs):
    U = args.create_backend()
    tmax_launcher.prepare_tasks(U, args.terminal)
    search_launcher.write_prompt_data(args.search)
    _write_prompt_data(args)
    for name in (args.model_name, *args.teachers.values()):
        U.exec_command_cpu(fetch_command(tmax_launcher.RECIPES[name].hf_repo, f"{args.model_dir}/{name}"))
    # SGLang serves the teachers' HF checkpoints; only the student trains.
    U.convert_checkpoint(
        model_name=args.model_name,
        megatron_model_type=args.recipe.megatron_model_type,
        num_gpus_per_node=args.num_gpus_per_node,
        dir_dst=args.model_dir,
        hf_checkpoint=f"{args.model_dir}/{args.model_name}",
        megatron_path=args.megatron_path,
    )


def _write_prompt_data(args: ScriptArgs) -> None:
    """Both launchers' prompts, each row tagged with its domain (the episode loop, the reward and the metrics) and
    teacher (``opd_teacher``, the teacher server): every terminal task once, and each search question as often as
    search_fraction asks (mix_prompts)."""
    rows = {}
    for domain, path in (("terminal", args.terminal.prompt_data), ("search", args.search.prompt_data)):
        rows[domain] = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        for row in rows[domain]:
            row["metadata"] |= {"domain": domain, "opd_teacher": domain}
    mixed = mix_prompts(rows["terminal"], rows["search"], args.search_fraction)
    out = Path(args.prompt_data)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.writelines(json.dumps(row) + "\n" for row in mixed)
    share = sum(row["metadata"]["domain"] == "search" for row in mixed) / len(mixed)
    print(f"{out}: {len(mixed)} prompts, {share:.3f} search", flush=True)


def mix_prompts(terminal: list[dict], search: list[dict], search_fraction: float) -> list[dict]:
    """Every terminal prompt once, then the search prompts repeated (whole copies, at least one) so that about
    search_fraction of the rows are search's. --rollout-shuffle draws from the mix at random."""
    copies = max(1, round(search_fraction / (1 - search_fraction) * len(terminal) / len(search)))
    return terminal + search * copies


@app.command()
@command_utils.dataclass_cli
def execute(args: ScriptArgs):
    U = args.create_backend()
    terminal_env = args.terminal.agent_env(trace_dir=args.domain_trace_dir("terminal"))
    search_env = args.search.env_args()

    ckpt_args = (
        f"--hf-checkpoint {args.model_dir}/{args.model_name} "
        f"--ref-load {args.model_dir}/{args.model_name}_torch_dist "
        f"--load {args.checkpoint_dir} "
        f"--save {args.checkpoint_dir} "
        f"--save-interval {args.save_interval} "
    )
    if args.hf_snapshot_every:
        ckpt_args += f"--save-hf {args.checkpoint_dir}/hf/step_{{rollout_id}} "

    # Both recipes sample at temperature 1, top-p 1, no top-k. The per-turn cap and the episode budget given here are
    # the job-wide ones (the longer domain's); mopd_agent_function applies each domain's own.
    rollout_args = (
        f"--prompt-data {args.prompt_data} "
        "--input-key prompt "
        "--metadata-key metadata "
        "--rollout-shuffle "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        "--rollout-temperature 1.0 "
        "--rollout-top-p 1.0 "
        "--rollout-top-k -1 "
        f"--rollout-max-response-len {max(args.terminal_max_response_len, args.search_max_response_len)} "
        f"--max-seq-len {args.max_seq_len} "
        f"--global-batch-size {args.rollout_batch_size * args.n_samples_per_prompt} "
        "--balance-data "
    )

    # generate.generate_with_teacher: Miles' agentic generate, then each sample's teacher scores it.
    agent_args = (
        "--custom-generate-function-path generate.generate_with_teacher "
        "--custom-agent-function-path mopd_agent_function.run "
        "--custom-rm-path mopd_reward.reward_func "
        f"--use-session-server {args.session_server} "
        "--tito-model qwen35 "
        "--dynamic-sampling-filter-path mopd_metrics.missing_reward_filter "
        "--custom-rollout-log-function-path mopd_metrics.log_rollout_data "
    )

    perf_args = (
        f"--tensor-model-parallel-size {args.tensor_model_parallel_size} "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        f"--context-parallel-size {args.context_parallel_size} "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        f"--max-tokens-per-gpu {args.max_tokens_per_gpu} "
    )

    # Pure OPD (open-instruct's --opd_pure): every trained reward is 0 (generate.opd_pure_rewards), so the advantage
    # is the OPD term alone; raw_reward stays the domain's task reward. DPPO anchored on the rollout's log-probs.
    grpo_args = (
        "--advantage-estimator grpo "
        "--disable-grpo-std-normalization "
        "--use-rollout-logprobs "
        "--calculate-per-token-loss "
        "--entropy-coef 0.0 "
        "--observe-training-entropy "
        f"--policy-loss-type dppo_tv --dppo-tv-threshold {args.dppo_threshold} "
        f"--use-opd --opd-type sglang --opd-kl-coef {args.opd_kl_coef} "
        # generate.generate_with_teacher sends Miles' teacher request to the servers opd_teachers lists, by the
        # sample's teacher, in place of --rm-url.
        f"--rm-url http://127.0.0.1:{opd_teachers.PORT}/generate "
        "--custom-reward-post-process-path generate.opd_pure_rewards "
    )
    if args.log_kl_to_base:
        grpo_args += "--use-kl-loss --kl-loss-coef 0.0 --kl-loss-type k3 "

    optimizer_args = (
        "--optimizer adam "
        f"--lr {args.lr} "
        "--lr-decay-style constant "
        "--weight-decay 0.0 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.999 "
        "--seed 42 "
    )
    if args.optimizer_cpu_offload:
        optimizer_args += "--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer "

    # SGLang TP>1 produces garbage output for Qwen3.5 (sgl-project/sglang#21039), hence one GPU per engine.
    sglang_args = (
        "--rollout-num-gpus-per-engine 1 "
        f"--sglang-mem-fraction-static {args.sglang_mem_fraction_static} "
        "--sglang-reasoning-parser qwen3 "
        "--sglang-tool-call-parser qwen3_coder "
    )
    if args.fp32_lm_head:
        sglang_args += "--sglang-enable-fp32-lm-head "

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        f"--actor-num-nodes {args.train_layout[0]} "
        f"--actor-num-gpus-per-node {args.train_layout[1]} "
        # On one node, the GPUs Miles gets: the teacher servers hold the rest.
        f"--num-gpus-per-node {args.num_gpus_per_node - (len(args.teacher_servers) if args.num_nodes == 1 else 0)} "
        "--fully-async "
        f"--max-weight-staleness {args.max_weight_staleness} "
        f"--async-data-buffer-capacity-factor {args.async_data_buffer_capacity_factor} "
        f"--async-max-concurrent-samples {args.async_max_concurrent_samples} "
        "--pause-generation-mode in_place "
        "--update-weight-transfer-mode broadcast "
        f"--rollout-num-gpus {args.async_rollout_gpus} "
    )
    if args.fp32_lm_head:
        misc_args += "--fp32-lm-head "
    if args.num_nodes > 1:
        # The agent processes run on the head, where the entrypoint's sandbox gateway listens.
        misc_args += "--pin-rollout-manager-to-head "
    if args.dump_training_data:
        assert args.trace_dir, "dump_training_data writes next to the traces: set trace_dir"
        misc_args += (
            f"--save-debug-rollout-data {args.trace_run_dir}/rollout_data/{{rollout_id}}.pt "
            f"--save-debug-train-data {args.trace_run_dir}/train_data/{{rollout_id}}_{{rank}}.pt "
        )

    wandb = wandb_args(
        __file__, run_id=args.run_id, exp_name=args.exp_name, project=args.wandb_project, entity=args.wandb_entity
    )
    train_args = (
        f"{ckpt_args} {rollout_args} {agent_args} {optimizer_args} {grpo_args} {wandb} {perf_args} "
        f"{sglang_args} {misc_args} {args.extra_args} "
    )

    source = Path(args.prime_rl_source)
    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.recipe.megatron_model_type,
        megatron_path=args.megatron_path,
        train_script="train_async.py",
        extra_env_vars={
            "PYTHONPATH": ":".join(
                [
                    args.megatron_path,
                    str(_SCRIPT_DIR),
                    str(_TMAX_DIR),
                    str(_SEARCH_DIR),
                    str(command_utils.repo_base_dir),
                    str(source / "deps/literegistry/literegistry_tool_client/src"),  # literegistry_tool_client
                    str(source / "environments/search_agent"),  # search_agent.{protocol,tools,tool_metrics}
                ]
            ),
            **terminal_env,
            ENV_ARGS_VAR: json.dumps(search_env),
            "SEARCH_TRACE_DIR": args.domain_trace_dir("search"),
            "MOPD_BUDGETS": json.dumps(args.budgets),
            # read by generate.generate_with_teacher, in the rollout process
            "TMAX_OPD_TEACHER_TIMEOUT_S": str(args.opd_teacher_timeout_s),
            "TMAX_OPD_TEACHER_URLS_FILE": args.opd_teacher_urls_file,
        },
        before_ray_job_submit=lambda: _start_teachers(args),
    )


def _start_teachers(args: ScriptArgs) -> None:
    """Each domain's teacher on its SGLang servers, on the rollout nodes (opd_teachers), listed by domain."""
    opd_teachers.start(
        args.teacher_servers,
        tokenizer_path=f"{args.model_dir}/{args.model_name}",
        context_length=args.max_seq_len + 2048,
        num_nodes=args.num_nodes,
        train_nodes=args.train_layout[0],
        urls_file=args.opd_teacher_urls_file,
        # The search recipe's fp32 logits: the teacher's log-probs without bf16 rounding, as the student's.
        server_args=("--enable-fp32-lm-head",) if args.fp32_lm_head else (),
    )


if __name__ == "__main__":
    sys.argv = apply_yaml_config(sys.argv, ScriptArgs)
    app()
