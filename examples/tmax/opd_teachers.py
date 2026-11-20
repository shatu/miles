"""SGLang servers of the on-policy distillation teachers, which a launcher starts before it submits the training job.

``start`` serves each teacher on SGLang servers of one GPU each and lists them in a JSON file, ``{teacher name: [server
/generate URLs]}``, which generate.generate_with_teacher reads in the rollout process. One teacher scores every sample
(examples/tmax, --opd-teacher-mode sglang); with several, each sample names its own (multi-teacher OPD,
examples/tmax_mopd).
"""

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PORT = 30100  # the first teacher server on a node; the next ones count up


def start(
    servers: list[tuple[str, str]],
    *,
    tokenizer_path: str,
    context_length: int,
    num_nodes: int,
    train_nodes: int,
    urls_file: str,
    server_args: tuple[str, ...] = (),
) -> None:
    """Start one SGLang server of one GPU per ``(teacher name, HF checkpoint)`` in ``servers``, list them in
    ``urls_file`` and wait until every one answers: up to 30 min, after which the launch fails with the servers' log
    tails. Runs after the Ray cluster is up, before the job is submitted.

    Each server is a detached Ray actor that holds its GPU (Ray sets its CUDA_VISIBLE_DEVICES) and runs SGLang on it,
    on the last nodes in the order Miles' placement group sorts them (node IP): there the rollout side is, so the
    placement group, created next on the GPUs left, still gives training its whole nodes first.
    """
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    from miles.ray.placement_group import sort_key

    ray.init(address="auto", namespace="tmax_opd_teacher", logging_level="ERROR")
    nodes = sorted(
        (n for n in ray.nodes() if n["Alive"] and n["Resources"].get("GPU")),
        key=lambda n: sort_key((0, n["NodeManagerAddress"], 0)),
    )
    assert len(nodes) == num_nodes, f"expected {num_nodes} GPU nodes in the Ray cluster, found {len(nodes)}"
    placement = hosts(len(nodes), train_nodes, len(servers))
    run_dir = os.path.dirname(urls_file)
    os.makedirs(run_dir, exist_ok=True)
    ship_by_value()
    server = ray.remote(num_gpus=1, num_cpus=1)(_TeacherServer)
    teachers: dict[str, list[str]] = {}
    logs = []
    for i, ((name, checkpoint), node) in enumerate(zip(servers, placement, strict=True)):
        port = PORT + placement[:i].count(node)
        logs.append(f"{run_dir}/opd-teacher-{i}.log")
        actor = server.options(
            name=f"opd-teacher-{i}",
            lifetime="detached",
            scheduling_strategy=NodeAffinitySchedulingStrategy(nodes[node]["NodeID"], soft=False),
        ).remote()
        command = [
            "-m", "sglang.launch_server",
            "--model-path", checkpoint,
            # --language-only: the tmax checkpoints and Miles' HF snapshots ship no vision tower. SGLang still loads an
            # image processor for Qwen3.5, from the tokenizer path; the tmax checkpoints ship none, so the student's
            # stands in (the processor configs are identical across Qwen3.5 sizes, and the vocabulary is shared:
            # requests carry token ids).
            "--tokenizer-path", tokenizer_path, "--language-only",
            "--port", str(port), "--tp-size", "1", "--mem-fraction-static", "0.8",
            "--context-length", str(context_length),
            *server_args,
        ]  # fmt: skip
        host = ray.get(actor.start.remote(command, logs[-1]))
        teachers.setdefault(name, []).append(f"http://{host}:{port}/generate")
    with open(urls_file, "w") as f:
        json.dump(teachers, f)
    print(f"OPD teacher servers: {teachers}", flush=True)
    pending, deadline = {url for urls in teachers.values() for url in urls}, time.time() + 1800
    while pending and time.time() < deadline:
        for url in sorted(pending):
            try:
                urllib.request.urlopen(url.removesuffix("/generate") + "/health_generate", timeout=30)
                pending.discard(url)
            except OSError:
                pass
        if pending:
            time.sleep(5)
    if pending:
        for log in logs:
            print(f"==> {log}", Path(log).read_text()[-8000:], sep="\n", flush=True)
        raise RuntimeError(f"OPD teacher servers not healthy after 30 min: {sorted(pending)}")


def ship_by_value() -> None:
    """Have Ray pickle this module's actor class by value. By reference (Ray's default for an importable module), a
    worker imports ``opd_teachers``, which only the launcher's path has: the actor failed with ModuleNotFoundError
    (2026-11-20, smoke 01M7VA7ZBZ53AC8V9G3567G0QK). Defined in the launcher script (``__main__``), it shipped by value.
    """
    import ray.cloudpickle

    ray.cloudpickle.register_pickle_by_value(sys.modules[__name__])


def hosts(num_nodes: int, train_nodes: int, num_servers: int) -> list[int]:
    """The node (index in Miles' placement order) of each teacher server: round-robin over the rollout nodes, the
    last first; on one node, that node."""
    rollout_nodes = list(range(train_nodes, num_nodes)) if num_nodes > 1 else [0]
    return [rollout_nodes[-1 - i % len(rollout_nodes)] for i in range(num_servers)]


class _TeacherServer:
    """A Ray actor holding one GPU, on which it runs an SGLang teacher server. The server is the actor's child, so Ray
    stops it with the actor (detached: at ray stop, or when the node goes), and the GPU is never busy while Ray counts
    it free."""

    def start(self, command: list[str], log: str) -> str:
        import ray

        host = ray.util.get_node_ip_address()
        with open(log, "w") as out:
            self._server = subprocess.Popen(
                [sys.executable, *command, "--host", host], stdout=out, stderr=subprocess.STDOUT
            )
        return host
