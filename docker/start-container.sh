#!/usr/bin/env bash
# Start a torch-helion development container from the Loom development image.
#
# Unlike Loom's launcher, this binds the local checkout into the container so
# the same working tree is shared with the host and with VS Code Dev Containers.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONTAINER_NAME="${TORCH_HELION_CONTAINER_NAME:-torch-helion-dev}"
SSH_PORT="${TORCH_HELION_SSH_PORT:-2223}"
AUTHORIZED_KEYS="${TORCH_HELION_AUTHORIZED_KEYS:-${HOME}/.ssh/authorized_keys}"
IMAGE="${TORCH_HELION_IMAGE:-ftod/loom_dev:latest}"
WORKSPACE_DIR="/workspace/$(basename "$REPO_ROOT")"

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

command -v docker >/dev/null 2>&1 || fail "docker is not installed"

if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
    fail "container already exists: $CONTAINER_NAME"
fi

HARDWARE_ARGS=()

if [[ -c /dev/tenstorrent/0 && -d /dev/hugepages-1G ]]; then
    echo "Tenstorrent hardware detected; attaching it to the container."
    HARDWARE_ARGS+=(
        --device /dev/tenstorrent/0:/dev/tenstorrent/0
        -v /dev/hugepages:/dev/hugepages
        -v /dev/hugepages-1G:/dev/hugepages-1G
    )
else
    echo "Tenstorrent device or 1G hugepages directory not found; starting without it."
fi

docker create \
    --name "$CONTAINER_NAME" \
    --hostname "$CONTAINER_NAME" \
    --init \
    --restart unless-stopped \
    --publish "127.0.0.1:${SSH_PORT}:22" \
    "${HARDWARE_ARGS[@]}" \
    --mount "type=bind,src=${REPO_ROOT},dst=${WORKSPACE_DIR}" \
    --workdir "$WORKSPACE_DIR" \
    "$IMAGE" >/dev/null

if [ -s "$AUTHORIZED_KEYS" ]; then
    docker cp "$AUTHORIZED_KEYS" "${CONTAINER_NAME}:/run/loom/authorized_keys" \
        || echo "WARNING: could not copy authorized keys; SSH login is disabled."
else
    echo "No authorized keys at $AUTHORIZED_KEYS; SSH login is disabled."
fi

docker start "$CONTAINER_NAME" >/dev/null

echo "torch-helion container started."
echo "Install the workspace: docker exec -it ${CONTAINER_NAME} bash install-docker.sh"
echo "Open a shell:          docker exec -it ${CONTAINER_NAME} bash"
echo "Connect over SSH:      ssh -A -p ${SSH_PORT} root@localhost"
