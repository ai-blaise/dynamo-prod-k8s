#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# Non-disruptive by default. MODE=apply is required to create resources.
MODE=${MODE:-render}
SERVER_DRY_RUN=${SERVER_DRY_RUN:-0}
ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
SCRIPT_DIR="$ROOT_DIR/deploy/production/scripts"
BASE_MANIFEST=${BASE_MANIFEST:-$ROOT_DIR/deploy/production/examples/deepseek-v32-nextn-optrt-swapabodd-nixl.yaml}
IMAGE=${IMAGE:-local/dynamo-trtllm-optrt-custom:optrt-2ef2472-kvarn-graph-20260606}
OUT=${OUT:-/tmp/ucx-nixl-transport-ab-$(date +%Y%m%dT%H%M%SZ)}
NAMESPACE=${NAMESPACE:-moriio-bench}
RUN_BENCH=${RUN_BENCH:-0}
URL=${URL:-}

case "$MODE" in
  render|apply) ;;
  *) echo "MODE must be render or apply" >&2; exit 2 ;;
esac

mkdir -p "$OUT"
echo "[ucx-nixl-ab] probing image=$IMAGE for UCX and NIXL runtime readiness"
python3 "$SCRIPT_DIR/probe_moriio_deps.py" \
  --image "$IMAGE" \
  --require UCX \
  --require NIXL \
  --out "$OUT/dependency-gate.json"

render_one() {
  local name=$1 transport=$2
  local dir="$OUT/$name"
  mkdir -p "$dir"
  echo "[ucx-nixl-ab] render name=$name transport=$transport mode=$MODE"
  python3 "$SCRIPT_DIR/moriio_transport_benchmark.py" \
    --manifest "$BASE_MANIFEST" \
    --image "$IMAGE" \
    --namespace "$NAMESPACE-$name" \
    --transports "$transport" \
    --mode "$MODE" \
    --out-dir "$dir" | tee "$dir/render-report.log"
  local manifest
  manifest=$(python3 - "$dir/report.json" <<'PY'
import json, sys
payload=json.load(open(sys.argv[1]))
for variant in payload.get("variants", []):
    if variant.get("status") in {"rendered", "applied"}:
        print(variant.get("manifest", ""))
        break
PY
)
  if [[ "$SERVER_DRY_RUN" == "1" && -n "$manifest" ]]; then
    echo "[ucx-nixl-ab] server dry-run $manifest"
    sudo -E /usr/local/bin/k3s kubectl -n dynamo-system apply --dry-run=server -f "$manifest"
  fi
}

render_one ucx-baseline UCX_BASELINE
render_one ucx-pinning UCX
render_one nixl-baseline NIXL_BASELINE
render_one nixl-pinning NIXL

if [[ "$RUN_BENCH" == "1" ]]; then
  if [[ -z "$URL" ]]; then
    echo "RUN_BENCH=1 requires URL=http://<frontend>:8000" >&2
    exit 2
  fi
  echo "[ucx-nixl-ab] benchmarking active frontend URL=$URL"
  URL="$URL" OUT="$OUT/openai-current" "$SCRIPT_DIR/run_moriio_openai_matrix.sh"
else
  cat <<INFO
[ucx-nixl-ab] Prepared UCX/NIXL variants only. No GPU workload was launched.
Next after GPUs are explicitly assigned:
  MODE=apply OUT=$OUT IMAGE=$IMAGE BASE_MANIFEST=$BASE_MANIFEST deploy/production/scripts/run_ucx_nixl_transport_ab.sh
Then, once the selected variant is healthy:
  URL=http://<frontend>:8000 OUT=$OUT/openai-<variant> deploy/production/scripts/run_moriio_openai_matrix.sh
INFO
fi
