#!/usr/bin/env bash
# Builds the TMax training image and, with --beaker NAME, uploads it to Beaker, where it stays until deleted.
#
# Upstream publishes dated dev images (radixark/miles:dev-YYYYMMDDHHMM) and deletes them after 14 days; the image all
# TMax runs used, dev-202610151741, was deleted on 2026-10-29. This rebuilds it: upstream's docker/Dockerfile at the
# Miles commit that built it (docker-build run 38779343791), with the SGLang and Megatron commits that image had. A
# W&B run's recorded pip freeze gave those (sglang 0.5.22.dev46+g7b6993a, megatron-core 0.19.0+9df207fd1).
# sglang-miles was force-pushed since, so SGLang is fetched by commit, not by branch. The Dockerfile leaves most
# Python packages unpinned; Dockerfile.pins puts back the versions that had moved (train_image_pins.txt).
#
#   bash examples/tmax/docker/build_train_image.sh                        # build tmax-miles-train:<date>
#   bash examples/tmax/docker/build_train_image.sh --beaker tmax-miles-train-20261029
set -euo pipefail

MILES_COMMIT=af4413b74f33f072883b6fbc760cccba3d56160d     # upstream main at the 2026-10-15 17:41 build
SGLANG_IMAGE_TAG=v0.5.21
SGLANG_COMMIT=7b6993a310d0dbb67929b51582d1729a6295e417
MEGATRON_COMMIT=9df207fd129a64d49ee7f96d4ca8eb477fda1a5d
WHEELS_TAG_X86=cu130-torch213-x86_64

tag=tmax-miles-train:$(date -u +%Y%m%d)
beaker_name=""
workspace=ai2/oe-agents-holmes
budget=ai2/oe-omai
while [ $# -gt 0 ]; do
    case "$1" in
        --tag) tag=$2; shift 2 ;;
        --beaker) beaker_name=$2; shift 2 ;;
        --workspace) workspace=$2; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

repo=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
context=$(mktemp -d)
trap 'rm -rf "$context"' EXIT
git -C "$repo" fetch -q https://github.com/radixark/miles.git "$MILES_COMMIT" 2> /dev/null || true
git -C "$repo" archive "$MILES_COMMIT" | tar -x -C "$context"
# The Dockerfile's `COPY wheel[s]/` matches nothing without a wheels/ directory: BuildKit skips it, podman fails.
mkdir "$context/wheels"

# SGLANG_BRANCH is what the Dockerfile fetches; GitHub serves a commit by its hash.
docker build "$context" -f "$context/docker/Dockerfile" -t "$tag-upstream" \
    --platform linux/amd64 \
    --build-arg TARGETARCH=amd64 \
    --build-arg SGLANG_IMAGE_TAG=$SGLANG_IMAGE_TAG \
    --build-arg SGLANG_BRANCH=$SGLANG_COMMIT \
    --build-arg SGLANG_COMMIT=$SGLANG_COMMIT \
    --build-arg MEGATRON_COMMIT=$MEGATRON_COMMIT \
    --build-arg WHEELS_TAG_X86=$WHEELS_TAG_X86 \
    --build-arg MILES_COMMIT=$MILES_COMMIT \
    --label tmax.miles_commit=$MILES_COMMIT \
    --label tmax.sglang_image_tag=$SGLANG_IMAGE_TAG \
    --label tmax.sglang_commit=$SGLANG_COMMIT \
    --label tmax.megatron_commit=$MEGATRON_COMMIT \
    --label tmax.wheels_tag=$WHEELS_TAG_X86
here=$(cd "$(dirname "$0")" && pwd)
docker build "$here" -f "$here/Dockerfile.pins" -t "$tag" --build-arg BASE="$tag-upstream"

if [ -n "$beaker_name" ]; then
    beaker image create --name "$beaker_name" --workspace "$workspace" --budget "$budget" \
        --description "TMax training image: miles $MILES_COMMIT, sglang $SGLANG_COMMIT, megatron $MEGATRON_COMMIT" \
        "$tag"
fi
