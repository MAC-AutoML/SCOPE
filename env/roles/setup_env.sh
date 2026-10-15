#!/usr/bin/env bash
# Create one isolated virtualenv: train, image, or geneval.
# Dependencies install into <repo>/.venvs/<role>. The default package index is PyPI.
set -Eeuo pipefail
ROOT="${SCOPE_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MODE="${1:-train}"
case "$MODE" in train|image|geneval) ;; *) exit 2 ;; esac
VENV="${ROOT}/.venvs/${MODE}"
INDEX="${SCOPE_PYPI_INDEX:-https://pypi.org/simple}"
export UV_PYTHON_INSTALL_MIRROR="${SCOPE_PYTHON_MIRROR:-https://github.com/astral-sh/python-build-standalone/releases/download}"
mkdir -p "${ROOT}/.venvs"
LOCKS=()
trap 'for lock in "${LOCKS[@]}"; do rmdir "$lock" 2>/dev/null || true; done' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
acquire_lock() {
  local deadline=$((SECONDS + 1800))
  while ! mkdir "$1" 2>/dev/null; do
    test "$SECONDS" -lt "$deadline" || { echo "Timed out waiting for $1; inspect its owner before removing it" >&2; exit 1; }
    sleep 2
  done
  LOCKS+=("$1")
}
export PATH="${ROOT}/.bootstrap/bin:${PATH}"
if ! command -v uv >/dev/null; then
  acquire_lock "${ROOT}/.venvs/bootstrap.install-lock"
  if ! test -x "${ROOT}/.bootstrap/bin/uv"; then
    python3 -m venv "${ROOT}/.bootstrap"
    "${ROOT}/.bootstrap/bin/python" -m pip install --index-url "$INDEX" uv
  fi
  rmdir "${ROOT}/.venvs/bootstrap.install-lock"
  LOCKS=()
fi
# Atomic mkdir coordinates workers even when the shared filesystem has no lockd.
acquire_lock "${VENV}.install-lock"
REQ="${ROOT}/env/roles/requirements-${MODE}.txt"
STAMP="$(sha256sum "$REQ" | cut -d ' ' -f 1)"
if test -f "${VENV}/.ready" && test "$(cat "${VENV}/.ready")" = "$STAMP"; then exit 0; fi
PYTHON="${SCOPE_PYTHON:-3.10}"
if test -z "${SCOPE_PYTHON:-}" && test -x "${VENV}/bin/python"; then
  PYTHON="${VENV}/bin/python"
fi
uv venv --allow-existing --python "$PYTHON" "$VENV"
if test "$MODE" = geneval; then
  uv pip install --python "${VENV}/bin/python" --index-url "$INDEX" \
    --find-links "${MMCV_WHEEL_INDEX:-https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/index.html}" \
    --only-binary mmcv-full -r "$REQ"
else
  uv pip install --python "${VENV}/bin/python" --index-url "$INDEX" -r "$REQ"
fi
uv pip check --python "${VENV}/bin/python"
if test "$MODE" = geneval; then
  bash "${ROOT}/env/roles/setup_native_libs.sh"
  export LD_LIBRARY_PATH="${ROOT}/.native/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
case "$MODE" in
  train) "${VENV}/bin/python" -c 'from trl import GRPOTrainer; from transformers import Qwen3ForCausalLM' ;;
  image) "${VENV}/bin/python" -c 'from diffusers import QwenImagePipeline' ;;
  geneval) "${VENV}/bin/python" "${ROOT}/src/geneval_backend.py" ;;
esac
printf '%s\n' "$STAMP" > "${VENV}/.ready"
