#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

ROOT=${ROOT:-/home/spencergarnets/moriio-agent-20260605T1451Z}
OPTRT_ROOT=${OPTRT_ROOT:-$ROOT/TensorRT-LLM-moriio-optrt-af17afd}
BASE_IMAGE=${BASE_IMAGE:-local/dynamo-trtllm-optrt-custom:moriio-ucx-kvarn-k2v2-optrt-cutedsl-overlay-20260606}
OUT_IMAGE=${OUT_IMAGE:-local/dynamo-trtllm-optrt-custom:moriio-nixl-kvarn-k2v2-optrt-cutedsl-overlay-20260606}
BUILD_DIR=${BUILD_DIR:-$ROOT/build/nixl-wrapper-config-probe}
IMAGE_CTX=${IMAGE_CTX:-$ROOT/build/moriio-nixl-wrapper-image}
TORCH_PREFIX=/opt/dynamo/venv/lib/python3.12/site-packages/torch
TORCH_CMAKE=$TORCH_PREFIX/share/cmake/Torch

docker run --rm --user root --entrypoint bash \
  -v "$ROOT:/moriio" \
  "$BASE_IMAGE" -lc "set -euo pipefail
cd /moriio/${OPTRT_ROOT#$ROOT/}
rm -rf /moriio/${BUILD_DIR#$ROOT/}
mkdir -p /moriio/${BUILD_DIR#$ROOT/}
export HOME=/root
export PYTHONPATH=/moriio/${BUILD_DIR#$ROOT/}/_deps/cutlass-src/python:\${PYTHONPATH:-}
cmake -S cpp -B /moriio/${BUILD_DIR#$ROOT/} -G Ninja \
  -DPython_EXECUTABLE=/usr/bin/python3 \
  -DTorch_DIR=$TORCH_CMAKE \
  -DCMAKE_PREFIX_PATH=$TORCH_PREFIX \
  -DTORCH_INSTALL_PREFIX=$TORCH_PREFIX \
  -DNIXL_ROOT=/opt/nvidia/nvda_nixl \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_PYT=OFF \
  -DBUILD_TESTS=OFF \
  -DBUILD_BENCHMARKS=OFF \
  -DBUILD_DEEP_EP=OFF \
  -DBUILD_DEEP_GEMM=OFF \
  -DBUILD_FLASH_MLA=OFF \
  -DBUILD_MICRO_BENCHMARKS=OFF \
  -DENABLE_MULTI_DEVICE=OFF \
  -DNVRTC_DYNAMIC_LINKING=ON \
  -DCMAKE_CUDA_ARCHITECTURES=100 \
  2>&1 | tee /moriio/${BUILD_DIR#$ROOT/}/configure.log
cmake --build /moriio/${BUILD_DIR#$ROOT/} --target tensorrt_llm_nixl_wrapper --parallel 1 \
  2>&1 | tee /moriio/${BUILD_DIR#$ROOT/}/build-nixl-wrapper.log
ls -lh /moriio/${BUILD_DIR#$ROOT/}/tensorrt_llm/executor/cache_transmission/nixl_utils/libtensorrt_llm_nixl_wrapper.so"

sudo chown -R "$(id -u):$(id -g)" "$BUILD_DIR"
mkdir -p "$IMAGE_CTX"
cp "$BUILD_DIR/tensorrt_llm/executor/cache_transmission/nixl_utils/libtensorrt_llm_nixl_wrapper.so" "$IMAGE_CTX/"
cat > "$IMAGE_CTX/Dockerfile" <<DOCKERFILE
FROM $BASE_IMAGE
USER root
COPY libtensorrt_llm_nixl_wrapper.so /opt/dynamo/venv/lib/python3.12/site-packages/tensorrt_llm/libs/libtensorrt_llm_nixl_wrapper.so
COPY libtensorrt_llm_nixl_wrapper.so /usr/local/lib/libtensorrt_llm_nixl_wrapper.so
RUN set -eux; \\
    mkdir -p /opt/dynamo/venv/lib/python3.12/site-packages/tensorrt_llm/libs/nixl; \\
    cp -a /opt/nvidia/nvda_nixl/lib64/*.so* /opt/dynamo/venv/lib/python3.12/site-packages/tensorrt_llm/libs/nixl/; \\
    if [ -d /opt/nvidia/nvda_nixl/lib64/plugins ]; then cp -a /opt/nvidia/nvda_nixl/lib64/plugins /opt/dynamo/venv/lib/python3.12/site-packages/tensorrt_llm/libs/nixl/; fi; \\
    if command -v patchelf >/dev/null 2>&1; then \\
      patchelf --set-rpath '\$ORIGIN/nixl:\$ORIGIN/nixl/plugins:/opt/nvidia/nvda_nixl/lib64:/opt/nvidia/nvda_nixl/lib64/plugins' /opt/dynamo/venv/lib/python3.12/site-packages/tensorrt_llm/libs/libtensorrt_llm_nixl_wrapper.so; \\
      patchelf --set-rpath '\$ORIGIN/nixl:\$ORIGIN/nixl/plugins:/opt/nvidia/nvda_nixl/lib64:/opt/nvidia/nvda_nixl/lib64/plugins' /usr/local/lib/libtensorrt_llm_nixl_wrapper.so; \\
    fi; \\
    ldconfig || true; \\
    test -f /opt/dynamo/venv/lib/python3.12/site-packages/tensorrt_llm/libs/libtensorrt_llm_nixl_wrapper.so; \\
    test -f /opt/dynamo/venv/lib/python3.12/site-packages/tensorrt_llm/libs/nixl/libnixl.so
DOCKERFILE

docker build -t "$OUT_IMAGE" "$IMAGE_CTX"
docker image inspect "$OUT_IMAGE" --format '{{.Id}} {{.Size}}'
