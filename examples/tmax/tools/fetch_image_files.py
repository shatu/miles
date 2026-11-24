"""Copy selected layers of many images (e.g. R2E-Gym's hidden tests and test command) without pulling the images.

R2E-Gym images add their hidden tests and their test command in layers of their own (``COPY r2e_tests /r2e_tests``,
``COPY run_tests.sh /testbed/run_tests.sh``), and the dataset's copies are incomplete for some tasks (pyramid's
tests have subdirectories, pillow's run through ``r2e_tests/unittest_custom_runner.py``). This reads each image's
manifest and config from a registry, finds the layers whose history line contains one of ``--layer`` (default: those
two COPYs), and extracts them under ``<out>/<repository>/<tag>/`` (``r2e_tests/...``, ``testbed/run_tests.sh``),
with ``layers.json`` naming the digests. Images already there are skipped.

Registries: the docker-mirror pull-through caches a LiteRegistry fleet registers (``--gateway``), or Docker Hub
itself, anonymously. Both run into Docker Hub's pull limit for manifests nobody cached yet (a mirror then answers
404). ``--via-sandbox`` instead starts each image still missing as a TMax sandbox (tmax_sandbox's env vars: the
fleet's replicas pull with credentials) and reads ``/r2e_tests`` and ``/testbed/run_tests.sh`` out of the container,
in chunks under the gateway's 512 KiB output cap; same layout, ``layers.json`` says ``"via": "sandbox"``. Usage:

    python examples/tmax/tools/fetch_image_files.py --images-from <R2E-Gym parquet dir> --out <dir> \
        --gateway http://127.0.0.1:24750 [--via-sandbox]
"""

import argparse
import asyncio
import base64
import glob
import io
import json
import os
import random
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

ACCEPT = ",".join(
    [
        "application/vnd.docker.distribution.manifest.v2+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.oci.image.index.v1+json",
    ]
)
DEFAULT_LAYERS = ("COPY r2e_tests /r2e_tests", "COPY run_tests.sh /testbed/run_tests.sh")
# The paths those layers add, read from a running container by --via-sandbox.
SANDBOX_PATHS = ("r2e_tests", "testbed/run_tests.sh")
_CHUNK_BYTES = 360 * 1024  # base64 of a chunk stays under the gateway's 512 KiB stdout cap


def _get(url: str, token: str | None = None, accept: str | None = None, timeout: float = 120) -> bytes:
    request = urllib.request.Request(url)
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    if accept:
        request.add_header("Accept", accept)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


class Registry:
    """A registry's /v2 API: a pull-through mirror (no auth) or Docker Hub (anonymous bearer tokens)."""

    def __init__(self, base: str):
        self.base = base.rstrip("/")
        self.hub = "registry-1.docker.io" in base

    def _token(self, repo: str) -> str | None:
        if not self.hub:
            return None
        url = f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{repo}:pull"
        return json.loads(_get(url))["token"]

    def files(self, image: str, needles: tuple[str, ...]) -> tuple[dict[str, bytes], dict]:
        repo, tag = image.rsplit(":", 1)
        token = self._token(repo)
        manifest = json.loads(_get(f"{self.base}/v2/{repo}/manifests/{tag}", token, ACCEPT))
        if "manifests" in manifest:  # an index: take linux/amd64
            digest = next(
                m["digest"] for m in manifest["manifests"] if m.get("platform", {}).get("architecture") == "amd64"
            )
            manifest = json.loads(_get(f"{self.base}/v2/{repo}/manifests/{digest}", token, ACCEPT))
        config = json.loads(_get(f"{self.base}/v2/{repo}/blobs/{manifest['config']['digest']}", token))
        history = [h for h in config["history"] if not h.get("empty_layer")]
        if len(history) != len(manifest["layers"]):
            raise RuntimeError(f"{image}: {len(history)} history entries for {len(manifest['layers'])} layers")
        files, found = {}, {}
        for needle in needles:
            hits = [i for i, h in enumerate(history) if needle in h.get("created_by", "")]
            if not hits:
                raise RuntimeError(f"{image}: no layer made by {needle!r}")
            layer = manifest["layers"][hits[-1]]
            found[needle] = layer["digest"]
            with tarfile.open(
                fileobj=io.BytesIO(_get(f"{self.base}/v2/{repo}/blobs/{layer['digest']}", token))
            ) as tar:
                for member in tar.getmembers():
                    if member.isfile() and not os.path.basename(member.name).startswith(".wh."):
                        files[member.name] = tar.extractfile(member).read()
        return files, found


async def files_via_sandbox(image: str, empty_task_dir: str) -> dict[str, bytes]:
    """SANDBOX_PATHS of a container started from ``image``, as {path: bytes}."""
    from validate_swe_tasks import make_sandbox

    archive = "/tmp/.fetch_image_files.tgz"
    async with make_sandbox(image, empty_task_dir) as box:
        paths = " ".join(SANDBOX_PATHS)
        packed = await box._exec(f"cd / && tar czf {archive} {paths} && stat -c %s {archive}", timeout_s=300)
        if packed.exit_code != 0:
            raise RuntimeError(f"tar failed ({packed.exit_code}): {packed.stderr[-500:]}")
        size, data = int(packed.stdout.split()[-1]), b""
        for offset in range(0, size, _CHUNK_BYTES):
            chunk = await box._exec(f"tail -c +{offset + 1} {archive} | head -c {_CHUNK_BYTES} | base64 -w0")
            data += base64.b64decode(chunk.stdout.strip())
    if len(data) != size:
        raise RuntimeError(f"read {len(data)} of {size} bytes")
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        return {m.name: tar.extractfile(m).read() for m in tar.getmembers() if m.isfile()}


def mirrors_from_gateway(gateway: str) -> list[str]:
    models = json.loads(_get(f"{gateway.rstrip('/')}/v1/models", timeout=30))["data"]
    return [
        f"http://{r['host']}:{r['port']}"
        for m in models
        if m.get("id") == "docker-mirror"
        for r in m.get("metadata") or []
    ]


def _from_registries(registries: list[Registry], image: str, needles) -> tuple[dict[str, bytes], dict]:
    last = None
    for registry in random.sample(registries, len(registries))[:3]:
        try:
            return registry.files(image, needles)
        except Exception as e:  # try another mirror
            last = e
    raise last


def _safe(path: str) -> str:
    parts = [p for p in Path(path).parts if p not in ("", "/", "..")]
    return os.path.join(*parts)


async def fetch_all(
    images: list[str], out: Path, registries: list[Registry], needles, concurrency: int, via_sandbox: bool
) -> None:
    semaphore, done, failed = asyncio.Semaphore(concurrency), 0, []
    empty_task_dir = tempfile.mkdtemp(prefix="fetch-image-files-")  # no environment/setup.sh: the image as it is

    async def fetch(image: str) -> None:
        nonlocal done
        repo, tag = image.rsplit(":", 1)
        dest = out / repo / tag
        if (dest / "layers.json").exists():
            return
        async with semaphore:
            try:
                if via_sandbox:
                    files, found = await files_via_sandbox(image, empty_task_dir), {"via": "sandbox"}
                else:
                    files, found = await asyncio.to_thread(_from_registries, registries, image, needles)
            except Exception as e:
                failed.append((image, repr(e)))
                return
        tmp = dest.with_name(dest.name + ".tmp")
        for name, data in files.items():
            path = tmp / _safe(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        (tmp / "layers.json").write_text(json.dumps({"image": image, "layers": found}))
        os.replace(tmp, dest)
        done += 1
        if done % 100 == 0:
            print(f"{done} images fetched, {len(failed)} failed", flush=True)

    await asyncio.gather(*(fetch(i) for i in images))
    print(f"fetched {done}, failed {len(failed)}, of {len(images)} images under {out}")
    for image, error in failed[:20]:
        print(f"  {image}: {error[:300]}")
    if failed:
        (out / "failed.json").write_text(json.dumps(failed, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--images-from", required=True, help="a parquet dir with a docker_image column (R2E-Gym)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gateway", help="a LiteRegistry gateway whose docker-mirror services to read through")
    parser.add_argument("--registry", action="append", default=[], help="registry base URL(s) (default: Docker Hub)")
    parser.add_argument("--layer", action="append", default=[], help="history substring of a layer to extract")
    parser.add_argument("--via-sandbox", action="store_true", help="read the files from running containers")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    files = sorted(glob.glob(os.path.join(args.images_from, "**", "*.parquet"), recursive=True))
    images = sorted(set(pd.concat(pd.read_parquet(f, columns=["docker_image"]) for f in files).docker_image))
    if args.limit:
        images = images[: args.limit]
    bases = args.registry + (mirrors_from_gateway(args.gateway) if args.gateway else [])
    registries = [Registry(b) for b in bases or ["https://registry-1.docker.io"]]
    print(f"{len(images)} images through {len(registries)} registries: {[r.base for r in registries]}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    needles = tuple(args.layer) or DEFAULT_LAYERS
    asyncio.run(fetch_all(images, args.out, registries, needles, args.concurrency, args.via_sandbox))


if __name__ == "__main__":
    main()
