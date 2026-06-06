#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

URL=${URL:?Set URL to the OpenAI-compatible frontend base URL, e.g. http://10.43.x.y:8000}
MODEL=${MODEL:-BlaiseAI/DeepSeek-V3.2-REAP-345B-SpinQuant-ActKV-NVFP4-NextN-Graft}
OUT=${OUT:-/tmp/moriio-openai-matrix-$(date +%Y%m%dT%H%M%SZ)}
REQUESTS=${REQUESTS:-16}
CONCURRENCY=${CONCURRENCY:-16}
MAX_TOKENS=${MAX_TOKENS:-256}
PROMPT_TOKENS=${PROMPT_TOKENS:-1000 8000 32000 64000 128000}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

mkdir -p "$OUT"
for tokens in $PROMPT_TOKENS; do
  echo "[moriio] benchmark tokens=$tokens out=$OUT/openai-${tokens}.json"
  python3 "$SCRIPT_DIR/moriio_openai_benchmark.py" \
    --url "$URL" \
    --model "$MODEL" \
    --requests "$REQUESTS" \
    --concurrency "$CONCURRENCY" \
    --prompt-tokens "$tokens" \
    --max-tokens "$MAX_TOKENS" \
    --abort-one \
    --sample-nvidia-smi \
    --out "$OUT/openai-${tokens}.json"
done
python3 - "$OUT" <<'PY'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
print("prompt_tokens,success,failed,aborted,ttft_ms_p50,tpot_ms_p50,tokens_per_sec_user_after_ft_p50")
for path in sorted(out.glob("openai-*.json"), key=lambda p: int(p.stem.split("-")[-1])):
    payload = json.loads(path.read_text())
    cfg = payload["config"]
    s = payload["summary"]
    print(
        f"{cfg.get('prompt_tokens_target')},{s.get('success')},{s.get('failed')},{s.get('aborted')},"
        f"{s.get('ttft_ms_p50')},{s.get('tpot_ms_p50')},{s.get('tokens_per_sec_user_after_ft_p50')}"
    )
PY
