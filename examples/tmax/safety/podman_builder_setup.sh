#!/bin/bash
# Brings podman up inside a Beaker job (BEAKER_ALLOW_SUBCONTAINERS=1) for build_env_images.py, configured as the
# training jobs' sandbox daemons are (examples/tmax/beaker/tmax_beaker_entry.sh, step 1), and logs in to Docker Hub
# with the user's Beaker secret. Runs on stock ubuntu:24.04, whose apt podman is 4.9 like the training jobs'.
#
#   DOCKERHUB_USERNAME, DOCKER_PAT   the push login (DOCKER_PAT is a Beaker secret; never traced)
set -euo pipefail
PODMAN_DIR="$(cd "$(dirname "$0")/../beaker/podman" && pwd)"

apt-get update -q
DEBIAN_FRONTEND=noninteractive apt-get install -y -q --no-install-recommends \
    podman crun conmon netavark passt uidmap fuse-overlayfs golang-github-containers-common \
    python3 ca-certificates
mkdir -p /etc/containers/registries.conf.d
cp "$PODMAN_DIR/containers.conf" /etc/containers/containers.conf
cp "$PODMAN_DIR/policy.json" /etc/containers/policy.json
cp "$PODMAN_DIR/10-unqualified-search-registries.conf" /etc/containers/registries.conf.d/
# podman 4.9 rejects a storage.conf without runroot/graphroot.
printf '[storage]\ndriver = "overlay"\nrunroot = "/run/containers/storage"\ngraphroot = "/var/lib/containers/storage"\n' \
    > /etc/containers/storage.conf
grep -q "^root:10000:11165536" /etc/subuid 2>/dev/null || echo "root:10000:11165536" >> /etc/subuid
grep -q "^root:10000:11165536" /etc/subgid 2>/dev/null || echo "root:10000:11165536" >> /etc/subgid
# The images are built for podman sandboxes whose containers.conf shares the host network; so are their builds.
podman info --format '{{.Host.OCIRuntime.Name}} {{.Store.GraphDriverName}} {{.Version.Version}}'

printf '%s' "$DOCKER_PAT" | podman login docker.io -u "$DOCKERHUB_USERNAME" --password-stdin
