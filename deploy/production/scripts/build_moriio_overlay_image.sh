#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/spencergarnets/moriio-agent-20260605T1451Z}
BASE_IMAGE=${BASE_IMAGE:-local/dynamo-trtllm-optrt-custom:canonical-smc-r20-cpfix-ucx-mpirpc-kvarn2-ls-mlp-cutedsl-20260605}
OUT_IMAGE=${OUT_IMAGE:-local/dynamo-trtllm-optrt-custom:moriio-ucx-kvarn-k2v2-optrt-cutedsl-overlay-20260606}
CTX=${CTX:-$ROOT/build/moriio-overlay-image}

rm -rf "$CTX"
mkdir -p "$CTX/dynamo/trtllm/request_handlers" "$CTX/dynamo/trtllm/utils" "$CTX/dynamo/trtllm/tests"
mkdir -p "$CTX/tensorrt_llm/serve" "$CTX/optrt-tests/unittest/disaggregated" "$CTX/optrt-tests/blaise" "$CTX/optrt-docs/blaise"
cp "$ROOT/dynamo/components/src/dynamo/trtllm/llm_engine.py" "$CTX/dynamo/trtllm/llm_engine.py"
cp "$ROOT/dynamo/components/src/dynamo/trtllm/request_handlers/handler_base.py" "$CTX/dynamo/trtllm/request_handlers/handler_base.py"
cp "$ROOT/dynamo/components/src/dynamo/trtllm/utils/moriio_pinning.py" "$CTX/dynamo/trtllm/utils/moriio_pinning.py"
cp "$ROOT/dynamo/components/src/dynamo/trtllm/tests/test_moriio_pinning.py" "$CTX/dynamo/trtllm/tests/test_moriio_pinning.py"
OPTRT_ROOT=${OPTRT_ROOT:-$ROOT/TensorRT-LLM-moriio-optrt-af17afd}
cp "$OPTRT_ROOT/tensorrt_llm/serve/moriio_pinning.py" "$CTX/tensorrt_llm/serve/moriio_pinning.py"
cp "$OPTRT_ROOT/tensorrt_llm/serve/openai_disagg_service.py" "$CTX/tensorrt_llm/serve/openai_disagg_service.py"
cp "$OPTRT_ROOT/tensorrt_llm/serve/openai_protocol.py" "$CTX/tensorrt_llm/serve/openai_protocol.py"
cp "$OPTRT_ROOT/tests/unittest/disaggregated/test_openai_disagg_service.py" "$CTX/optrt-tests/unittest/disaggregated/test_openai_disagg_service.py"
cp "$OPTRT_ROOT/tests/blaise/test_moriio_docs_contract.py" "$CTX/optrt-tests/blaise/test_moriio_docs_contract.py"
cp "$OPTRT_ROOT/docs/blaise/README.md" "$CTX/optrt-docs/blaise/README.md"
cp "$OPTRT_ROOT/docs/blaise/moriio_disagg.md" "$CTX/optrt-docs/blaise/moriio_disagg.md"
cat > "$CTX/Dockerfile" <<EOF
FROM ${BASE_IMAGE}
COPY dynamo/trtllm/llm_engine.py /opt/dynamo/venv/lib/python3.12/site-packages/dynamo/trtllm/llm_engine.py
COPY dynamo/trtllm/request_handlers/handler_base.py /opt/dynamo/venv/lib/python3.12/site-packages/dynamo/trtllm/request_handlers/handler_base.py
COPY dynamo/trtllm/utils/moriio_pinning.py /opt/dynamo/venv/lib/python3.12/site-packages/dynamo/trtllm/utils/moriio_pinning.py
COPY dynamo/trtllm/tests/test_moriio_pinning.py /opt/dynamo/venv/lib/python3.12/site-packages/dynamo/trtllm/tests/test_moriio_pinning.py
COPY dynamo/trtllm/llm_engine.py /workspace/components/src/dynamo/trtllm/llm_engine.py
COPY dynamo/trtllm/request_handlers/handler_base.py /workspace/components/src/dynamo/trtllm/request_handlers/handler_base.py
COPY dynamo/trtllm/utils/moriio_pinning.py /workspace/components/src/dynamo/trtllm/utils/moriio_pinning.py
COPY dynamo/trtllm/tests/test_moriio_pinning.py /workspace/components/src/dynamo/trtllm/tests/test_moriio_pinning.py
COPY tensorrt_llm/serve/moriio_pinning.py /opt/dynamo/venv/lib/python3.12/site-packages/tensorrt_llm/serve/moriio_pinning.py
COPY tensorrt_llm/serve/openai_disagg_service.py /opt/dynamo/venv/lib/python3.12/site-packages/tensorrt_llm/serve/openai_disagg_service.py
COPY tensorrt_llm/serve/openai_protocol.py /opt/dynamo/venv/lib/python3.12/site-packages/tensorrt_llm/serve/openai_protocol.py
COPY optrt-tests/unittest/disaggregated/test_openai_disagg_service.py /workspace/tests/unittest/disaggregated/test_openai_disagg_service.py
COPY optrt-tests/blaise/test_moriio_docs_contract.py /workspace/tests/blaise/test_moriio_docs_contract.py
COPY optrt-docs/blaise/README.md /workspace/docs/blaise/README.md
COPY optrt-docs/blaise/moriio_disagg.md /workspace/docs/blaise/moriio_disagg.md
RUN PYTHONPYCACHEPREFIX=/tmp/pycache /opt/dynamo/venv/bin/python3 -m py_compile \
  /opt/dynamo/venv/lib/python3.12/site-packages/dynamo/trtllm/utils/moriio_pinning.py \
  /opt/dynamo/venv/lib/python3.12/site-packages/dynamo/trtllm/llm_engine.py \
  /opt/dynamo/venv/lib/python3.12/site-packages/dynamo/trtllm/request_handlers/handler_base.py \
  /opt/dynamo/venv/lib/python3.12/site-packages/tensorrt_llm/serve/moriio_pinning.py \
  /opt/dynamo/venv/lib/python3.12/site-packages/tensorrt_llm/serve/openai_disagg_service.py \
  /opt/dynamo/venv/lib/python3.12/site-packages/tensorrt_llm/serve/openai_protocol.py
EOF

docker build -t "$OUT_IMAGE" "$CTX"
docker run --rm --entrypoint bash "$OUT_IMAGE" -lc '/opt/dynamo/venv/bin/python3 - <<PY
from dynamo.trtllm.utils.moriio_pinning import *
env={ENV_PINNING_REQUIRED:"1", ENV_BACKEND:"UCX", ENV_PRODUCER_ID:"p", ENV_CONSUMER_ID:"c"}
params={"request_type":"context_only","ctx_dp_rank":1,"ctx_info_endpoint":"ctx:1","opaque_state":"opaque"}
attach_moriio_pin(params, request={"id":"r"}, disagg_request_id=1, component="p", disagg_machine_id=1, force=True, env=env)
validate_moriio_pin(params, request={"id":"r","ctx_dp_rank":1,"ctx_info_endpoint":"ctx:1"}, decoded_disagg_request_id=1, env=env)
print("PASS moriio overlay import/pin smoke")
PY'
echo "$OUT_IMAGE"
