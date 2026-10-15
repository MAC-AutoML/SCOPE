#!/usr/bin/env bash
# Start the full eight-node training job. Run this on every node.
# All nodes share SCOPE_JOB_ID and SCOPE_NODE_IPS. SCOPE_NODE_RANK is 0..7.
set -Eeuo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export SCOPE_PROJECT_ROOT="$ROOT"
if test -z "${SCOPE_JOB_ID:-}" || test -z "${SCOPE_NODE_IPS:-}"; then
  echo "Set SCOPE_JOB_ID and SCOPE_NODE_IPS, then run this script on every node." >&2
  echo "SCOPE_NODE_RANK selects this node (0..7). See env/env.example.txt." >&2
  exit 2
fi
exec python3 "${ROOT}/src/launch.py"
