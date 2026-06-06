#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# Render-only by default. Set MODE=apply only after GPUs are explicitly free.
MODE=${MODE:-render}
ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
SCRIPT_DIR="$ROOT_DIR/deploy/production/scripts"
BASE_MANIFEST=${BASE_MANIFEST:-$ROOT_DIR/deploy/production/examples/deepseek-v32-nextn-optrt-swapabodd-nixl.yaml}
OUT=${OUT:-/tmp/moriio-transport-ab-$(date +%Y%m%dT%H%M%SZ)}
NAMESPACE=${NAMESPACE:-moriio-bench}
PARENT_IMAGE=${PARENT_IMAGE:-local/dynamo-trtllm-optrt-custom:canonical-smc-r20-fullsrc-ls-kvarn2-nvlsfix-smcfi-multidecode-swapabodd-20260606}
NIXL_IMAGE=${NIXL_IMAGE:-local/dynamo-trtllm-optrt-custom:canonical-smc-r20-fullsrc-ls-kvarn2-nvlsfix-smcfi-multidecode-swapabodd-20260606}
MOONCAKE_IMAGE=${MOONCAKE_IMAGE:-local/dynamo-trtllm-optrt-custom:canonical-smc-r20-fullsrc-ls-kvarn2-nvlsfix-smcfi-multidecode-swapabodd-mooncake-20260606}
NATIVE_MORI_IMAGE=${NATIVE_MORI_IMAGE:-local/dynamo-trtllm-optrt-custom:canonical-smc-r20-fullsrc-ls-kvarn2-nvlsfix-smcfi-multidecode-swapabodd-mori-20260606}
RUN_BENCH=${RUN_BENCH:-0}
URL=${URL:-}

mkdir -p "$OUT"
case "$MODE" in
  render|apply) ;;
  *) echo "MODE must be render or apply" >&2; exit 2 ;;
esac

render_variant() {
  local name=$1 image=$2 transport=$3
  local dir="$OUT/$name"
  mkdir -p "$dir"
  echo "[moriio-ab] $name transport=$transport image=$image mode=$MODE"
  python3 "$SCRIPT_DIR/moriio_transport_benchmark.py"     --manifest "$BASE_MANIFEST"     --image "$image"     --namespace "$NAMESPACE-$name"     --transports "$transport"     --mode "$MODE"     --out-dir "$dir" | tee "$dir/render-report.log"
}

render_variant ucx-baseline "$PARENT_IMAGE" UCX_BASELINE
render_variant ucx-pinning "$PARENT_IMAGE" UCX
render_variant nixl-baseline "$NIXL_IMAGE" NIXL_BASELINE
render_variant nixl-pinning "$NIXL_IMAGE" NIXL
render_variant mooncake-pinning "$MOONCAKE_IMAGE" MOONCAKE || true
render_variant native-mori "$NATIVE_MORI_IMAGE" NATIVE_MORI || true

if [[ "$RUN_BENCH" == "1" ]]; then
  if [[ -z "$URL" ]]; then
    echo "RUN_BENCH=1 requires URL=http://<frontend>:8000" >&2
    exit 2
  fi
  echo "[moriio-ab] Running OpenAI benchmark matrix against current URL=$URL"
  URL="$URL" OUT="$OUT/openai-current" "$SCRIPT_DIR/run_moriio_openai_matrix.sh"
else
  cat <<INFO
[moriio-ab] Render complete. No GPU workload was launched.
To apply one rendered variant after GPUs are free, inspect $OUT/<variant>/report.json and run:
  MODE=apply OUT=$OUT BASE_MANIFEST=$BASE_MANIFEST deploy/production/scripts/run_moriio_transport_ab_matrix.sh
To benchmark the active frontend only after apply/readiness:
  URL=http://<frontend>:8000 OUT=$OUT/openai-<variant> deploy/production/scripts/run_moriio_openai_matrix.sh
INFO
fi
