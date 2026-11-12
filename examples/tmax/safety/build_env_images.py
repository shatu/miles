"""Build the SafeComp source tasks' environment images on Beaker and push them to Docker Hub.

    python examples/tmax/safety/build_env_images.py plan   --out-dir DIR [--folders train_clean ...] [--sources-file F]
    python examples/tmax/safety/build_env_images.py submit --out-dir DIR --shards 12
    python examples/tmax/safety/build_env_images.py run    --out-dir DIR --shard I --shards N       # inside the job

SafeComp's tasks (mingqianz's share, ``share_submission_v1_1``) are Harbor tasks: a Dockerfile per task, built at run
time. TMax sandboxes start from a prebuilt image instead. Every risk task's Dockerfile is its source task's control
Dockerfile plus an appended block that only copies files and runs ``safety_post.sh`` (checked over all 1,713
``train_clean`` risk tasks), so only the control environments are built here; ``make_dataset.py`` turns the appended
block into a setup the sandbox replays when the container starts.

``plan`` lists the control environments of the chosen folders (or of the sources in ``--sources-file``), tags each by a
hash of its files, so a source shared by several folders is built once and the tag says nothing about the task, and
writes ``plan.jsonl``. ``submit`` starts one Beaker experiment with one CPU task per shard. Sources sharing a
Dockerfile header (the 3.6 GB ``base_install.sh`` layer) land in as few shards as possible, so that layer is built and
pushed a few times rather than once per shard. ``run`` builds a shard with ``podman build`` on the unmodified task
directory, pushes it, removes it locally, and appends one JSON line per source to ``shard-<i>.jsonl``; a tag already in
the registry is skipped, so a rerun only retries what failed.
"""

import argparse
import base64
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SHARE_DIR = "/weka/oe-adapt-default/mingqianz/datasets/safecomp-test/final_run/share_submission_v1_1"
REPO = "docker.io/shashankg209/tmax-s-envs"
_SNAPSHOT = "/tmp/miles_snapshot"
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
_BUILD_ATTEMPTS = 3
_MANIFEST_ACCEPT = ", ".join(
    [
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    ]
)


def env_tag(env_dir: Path) -> str:
    """A content hash of the environment directory: relative path and bytes of every file, in path order."""
    digest = hashlib.sha256()
    for path in sorted(p for p in env_dir.rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(env_dir)).encode() + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()[:16]


def _dockerfile_header(env_dir: Path) -> str:
    """Everything before the task's own post_install step: the layers sources can share."""
    dockerfile = (env_dir / "Dockerfile").read_text()
    return hashlib.sha1(dockerfile.split("COPY post_install.sh")[0].encode()).hexdigest()[:8]


def _plan(args: argparse.Namespace) -> None:
    rows = [json.loads(line) for line in open(f"{args.share_dir}/manifest.jsonl")]
    controls = [r for r in rows if r["variant"] == "control" and r["folder"] in args.folders]
    if args.sources_file:
        wanted = {line.strip() for line in open(args.sources_file) if line.strip()}
        controls = [r for r in controls if r["task_id"] in wanted]
        missing = wanted - {r["task_id"] for r in controls}
        assert not missing, f"no control in {args.folders} for {sorted(missing)}"

    def one(row: dict) -> dict:
        env_dir = Path(row["target"]) / "environment"
        return {
            "source": row["task_id"],
            "env_dir": str(env_dir),
            "header": _dockerfile_header(env_dir),
            "tag": env_tag(env_dir),
        }

    with ThreadPoolExecutor(32) as pool:
        entries = list(pool.map(one, controls))
    by_tag: dict[str, dict] = {}
    for entry in entries:
        by_tag.setdefault(entry["tag"], entry)
    # A source in several folders must have one environment, or the dataset could not name one image for it.
    tags_per_source: dict[str, set] = {}
    for entry in entries:
        tags_per_source.setdefault(entry["source"], set()).add(entry["tag"])
    conflicts = {s: t for s, t in tags_per_source.items() if len(t) > 1}
    assert not conflicts, f"sources with differing environments across folders: {conflicts}"
    plan = sorted(by_tag.values(), key=lambda e: (e["header"], e["source"]))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "plan.jsonl", "w") as f:
        for entry in plan:
            f.write(json.dumps({**entry, "image": f"{args.repo}:{entry['tag']}"}) + "\n")
    print(f"{len(plan)} images from {len(controls)} control links ({sorted(set(args.folders))}) -> {out}/plan.jsonl")


def _shard(plan: list[dict], shard: int, shards: int) -> list[dict]:
    """Contiguous slices of the header-sorted plan: each header group spans as few shards as possible."""
    size = -(-len(plan) // shards)
    return plan[shard * size : (shard + 1) * size]


def _submit(args: argparse.Namespace) -> None:
    out = Path(args.out_dir)
    plan = [json.loads(line) for line in open(out / "plan.jsonl")]
    shards = min(args.shards, len(plan))
    commit = _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain", "--untracked-files=no", "--", "examples/tmax")
    assert not dirty, f"commit examples/tmax first: the job runs commit {commit[:10]}\n{dirty}"
    git_dir = _git("rev-parse", "--path-format=absolute", "--git-common-dir")
    snapshot = (
        "git config --global --add safe.directory '*' && "
        f"rm -rf {_SNAPSHOT} && git clone -q --shared --no-checkout {git_dir} {_SNAPSHOT} && "
        f"git -C {_SNAPSHOT} checkout -q {commit}"
    )
    here = _HERE.relative_to(_ROOT)
    tasks = []
    for shard in range(shards):
        script = (
            f"bash {_SNAPSHOT}/{here}/podman_builder_setup.sh && "
            f"python3 -u {_SNAPSHOT}/{here}/build_env_images.py run --out-dir {out} --shard {shard} --shards {shards} "
            f"--jobs {args.jobs}"
        )
        tasks.append(
            {
                "name": f"shard-{shard:03d}",
                "image": {"docker": "docker.io/library/ubuntu:24.04"},
                # ubuntu:24.04 has no git; the snapshot is cloned by a git-capable step first.
                "command": [
                    "bash",
                    "-c",
                    f"apt-get update -q && apt-get install -y -q git >/dev/null && {snapshot} && {script}",
                ],
                "datasets": [
                    {"mountPath": f"/weka/{b}", "source": {"weka": b}} for b in ("nora-default", "oe-adapt-default")
                ],
                "envVars": [
                    {"name": "BEAKER_ALLOW_SUBCONTAINERS", "value": "1"},
                    {"name": "BEAKER_SKIP_DOCKER_SOCKET", "value": "1"},
                    {"name": "DOCKERHUB_USERNAME", "value": args.dockerhub_username},
                    {"name": "DOCKER_PAT", "secret": args.docker_pat_secret},
                ],
                # No resources: a CPU request makes the scheduler attach a GPU (as the registry mirrors run).
                "context": {"priority": args.priority},
                "constraints": {"cluster": args.clusters},
            }
        )
    spec = {
        "version": "v2",
        "description": f"tmax-s-envs build: {len(plan)} images in {shards} shards -> {out}",
        "budget": args.budget,
        "tasks": tasks,
        "retry": {"allowedTaskRetries": args.retries},
    }
    name = f"tmax-s-envs-build-{time.strftime('%Y%m%d-%H%M%S')}"
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(spec, f, indent=1)
    subprocess.run(
        ["beaker", "experiment", "create", "--workspace", args.workspace, "--name", name, f.name], check=True
    )


def _git(*argv: str) -> str:
    return subprocess.run(["git", "-C", str(_ROOT), *argv], check=True, capture_output=True, text=True).stdout.strip()


def _registry_token(repository: str) -> str:
    """A pull token for docker.io/<repository>, with the job's login so private repositories are visible."""
    url = f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{repository}:pull"
    request = urllib.request.Request(url)
    login = f"{os.environ['DOCKERHUB_USERNAME']}:{os.environ['DOCKER_PAT']}".encode()
    request.add_header("Authorization", "Basic " + base64.b64encode(login).decode())
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)["token"]


def _pushed(image: str, token: str) -> bool:
    repository, tag = image.removeprefix("docker.io/").rsplit(":", 1)
    request = urllib.request.Request(f"https://registry-1.docker.io/v2/{repository}/manifests/{tag}", method="HEAD")
    request.add_header("Authorization", f"Bearer {token}")
    request.add_header("Accept", _MANIFEST_ACCEPT)
    try:
        with urllib.request.urlopen(request, timeout=30):
            return True
    except urllib.error.HTTPError as e:
        if e.code in (401, 404):  # 401: the repository does not exist yet
            return False
        raise


def _run(args: argparse.Namespace) -> None:
    out = Path(args.out_dir)
    plan = [json.loads(line) for line in open(out / "plan.jsonl")]
    entries = _shard(plan, args.shard, args.shards)
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    results = out / f"shard-{args.shard:03d}.jsonl"
    repository = entries[0]["image"].removeprefix("docker.io/").rsplit(":", 1)[0] if entries else ""
    print(f"shard {args.shard}/{args.shards}: {len(entries)} images, {args.jobs} at a time", flush=True)

    def one(entry: dict) -> dict:
        t0 = time.monotonic()
        record = {k: entry[k] for k in ("source", "tag", "image")}
        try:
            if _pushed(entry["image"], _registry_token(repository)):
                return {**record, "status": "exists", "seconds": 0.0}
        except Exception as e:  # an unreadable registry only costs a rebuild
            print(f"{entry['source']}: registry check failed ({e!r}); building", flush=True)
        log_path = logs / f"{entry['source']}.log"
        error = ""
        for attempt in range(_BUILD_ATTEMPTS):
            with open(log_path, "a") as log:
                log.write(f"=== attempt {attempt + 1} {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
                log.flush()
                build = subprocess.run(
                    ["podman", "build", "--network", "host", "-t", entry["image"], entry["env_dir"]],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                if build.returncode == 0:
                    with tempfile.NamedTemporaryFile("r") as digest_file:
                        push = subprocess.run(
                            ["podman", "push", "--digestfile", digest_file.name, entry["image"]],
                            stdout=log,
                            stderr=subprocess.STDOUT,
                        )
                        digest = Path(digest_file.name).read_text().strip()
                    subprocess.run(["podman", "rmi", entry["image"]], stdout=log, stderr=subprocess.STDOUT)
                    if push.returncode == 0:
                        return {
                            **record,
                            "status": "pushed",
                            "digest": digest,
                            "seconds": round(time.monotonic() - t0, 1),
                        }
                    error = f"push exited {push.returncode}"
                else:
                    error = f"build exited {build.returncode}"
            time.sleep(10 * (attempt + 1))
        return {
            **record,
            "status": "failed",
            "error": error,
            "log": str(log_path),
            "seconds": round(time.monotonic() - t0, 1),
        }

    failed = 0
    with ThreadPoolExecutor(args.jobs) as pool:
        for result in pool.map(one, entries):
            failed += result["status"] == "failed"
            print(json.dumps(result), flush=True)
            with open(results, "a") as f:
                f.write(json.dumps(result) + "\n")
    print(f"shard {args.shard}: done, {failed} failed", flush=True)
    sys.exit(1 if failed else 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["plan", "submit", "run"])
    parser.add_argument("--out-dir", required=True, help="plan.jsonl, shard-<i>.jsonl and build logs")
    parser.add_argument("--share-dir", default=SHARE_DIR)
    parser.add_argument("--folders", nargs="+", default=["train_clean", "id_test", "request_only_ood", "security_ood"])
    parser.add_argument("--sources-file", help="plan only these source task ids (one per line)")
    parser.add_argument("--repo", default=REPO)
    parser.add_argument("--shards", type=int, default=12)
    parser.add_argument("--shard", type=int)
    parser.add_argument("--jobs", type=int, default=4, help="concurrent builds per shard")
    parser.add_argument("--clusters", nargs="+", default=["ai2/jupiter", "ai2/saturn", "ai2/ceres"])
    parser.add_argument("--workspace", default="ai2/general-tool-use")
    parser.add_argument("--budget", default="ai2/oe-omai")
    parser.add_argument("--priority", default="urgent")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--dockerhub-username", default="shashankg209")
    parser.add_argument("--docker-pat-secret", default="shashankg_DOCKER_PAT")
    args = parser.parse_args()
    {"plan": _plan, "submit": _submit, "run": _run}[args.command](args)


if __name__ == "__main__":
    main()
