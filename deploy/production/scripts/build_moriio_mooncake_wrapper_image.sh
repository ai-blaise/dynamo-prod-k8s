#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

ROOT=${ROOT:-/home/spencergarnets/moriio-agent-20260605T1451Z}
OPTRT_ROOT=${OPTRT_ROOT:-$ROOT/TensorRT-LLM-fullsrc-nixl-538627d}
MOONCAKE_ROOT_SRC=${MOONCAKE_ROOT_SRC:-$ROOT/Mooncake}
BASE_IMAGE=${BASE_IMAGE:-local/dynamo-trtllm-optrt-custom:canonical-smc-r20-fullsrc-ls-kvarn2-nvlsfix-smcfi-hostreq-20260606}
OUT_IMAGE=${OUT_IMAGE:-local/dynamo-trtllm-optrt-custom:canonical-smc-r20-fullsrc-ls-kvarn2-nvlsfix-smcfi-hostreq-mooncake-20260606}
SDK_PREFIX=${SDK_PREFIX:-$ROOT/build/mooncake-sdk}
MOONCAKE_BUILD_DIR=${MOONCAKE_BUILD_DIR:-$ROOT/build/mooncake-transfer-engine}
TRTLLM_BUILD_DIR=${TRTLLM_BUILD_DIR:-$ROOT/build/mooncake-wrapper-trtllm}
IMAGE_CTX=${IMAGE_CTX:-$ROOT/build/moriio-mooncake-wrapper-image}
TORCH_PREFIX=/opt/dynamo/venv/lib/python3.12/site-packages/torch
TORCH_CMAKE=$TORCH_PREFIX/share/cmake/Torch

missing=()
for path in "$MOONCAKE_ROOT_SRC/mooncake-transfer-engine/include/transfer_engine_c.h" "$OPTRT_ROOT/cpp/tensorrt_llm/executor/cache_transmission/mooncake_utils/CMakeLists.txt"; do
  [[ -e "$path" ]] || missing+=("$path")
done
if ((${#missing[@]})); then
  printf 'Missing required source paths:\n' >&2
  printf '  %s\n' "${missing[@]}" >&2
  exit 2
fi

cat <<INFO
Mooncake wrapper build inputs:
  BASE_IMAGE=$BASE_IMAGE
  OUT_IMAGE=$OUT_IMAGE
  MOONCAKE_ROOT_SRC=$MOONCAKE_ROOT_SRC
  SDK_PREFIX=$SDK_PREFIX
  OPTRT_ROOT=$OPTRT_ROOT

This script is non-GPU. It may still be CPU/package heavy because Mooncake Transfer Engine
requires system dev packages and initialized submodules. If configure fails on yaml-cpp,
JsonCpp, GLOG, gflags, ibverbs, yalantinglibs, or CUDA headers, run Mooncake's dependency
installer in a maintenance window or provide a prebuilt SDK with:
  MOONCAKE_ROOT=<prefix containing include/transfer_engine_c.h and lib/libtransfer_engine.so>
INFO

if [[ -n "${MOONCAKE_ROOT:-}" ]]; then
  test -f "$MOONCAKE_ROOT/include/transfer_engine_c.h"
  test -f "$MOONCAKE_ROOT/lib/libtransfer_engine.so"
else
  docker run --rm --user root --entrypoint bash \
    -v "$ROOT:/moriio" \
    "$BASE_IMAGE" -lc "set -euo pipefail
cd /moriio/${MOONCAKE_ROOT_SRC#$ROOT/}
if [ -f .gitmodules ]; then git submodule update --init --recursive; fi
rm -rf /moriio/${MOONCAKE_BUILD_DIR#$ROOT/} /moriio/${SDK_PREFIX#$ROOT/}
mkdir -p /moriio/${MOONCAKE_BUILD_DIR#$ROOT/} /moriio/${SDK_PREFIX#$ROOT/}
cmake -S . -B /moriio/${MOONCAKE_BUILD_DIR#$ROOT/} -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/moriio/${SDK_PREFIX#$ROOT/} \
  -DBUILD_SHARED_LIBS=ON \
  -DWITH_TE=ON \
  -DWITH_STORE=OFF \
  -DWITH_STORE_RUST=OFF \
  -DWITH_RUST_EXAMPLE=OFF \
  -DWITH_P2P_STORE=OFF \
  -DWITH_EP=OFF \
  -DBUILD_UNIT_TESTS=OFF \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_BENCHMARK=OFF \
  -DUSE_CUDA=ON \
  2>&1 | tee /moriio/${MOONCAKE_BUILD_DIR#$ROOT/}/configure.log
cmake --build /moriio/${MOONCAKE_BUILD_DIR#$ROOT/} --target transfer_engine --parallel 1 \
  2>&1 | tee /moriio/${MOONCAKE_BUILD_DIR#$ROOT/}/build-transfer-engine.log
cmake --install /moriio/${MOONCAKE_BUILD_DIR#$ROOT/} --prefix /moriio/${SDK_PREFIX#$ROOT/}
test -f /moriio/${SDK_PREFIX#$ROOT/}/include/transfer_engine_c.h
test -f /moriio/${SDK_PREFIX#$ROOT/}/lib/libtransfer_engine.so"
  sudo chown -R "$(id -u):$(id -g)" "$MOONCAKE_BUILD_DIR" "$SDK_PREFIX"
  export MOONCAKE_ROOT="$SDK_PREFIX"
fi

mkdir -p "$TRTLLM_BUILD_DIR"
docker run --rm --user root --entrypoint bash \
  -v "$ROOT:/moriio" \
  "$BASE_IMAGE" -lc "set -euo pipefail
cd /moriio/${OPTRT_ROOT#$ROOT/}
rm -rf /moriio/${TRTLLM_BUILD_DIR#$ROOT/}
mkdir -p /moriio/${TRTLLM_BUILD_DIR#$ROOT/}
cmake -S cpp -B /moriio/${TRTLLM_BUILD_DIR#$ROOT/} -G Ninja \
  -DPython_EXECUTABLE=/usr/bin/python3 \
  -DTorch_DIR=$TORCH_CMAKE \
  -DCMAKE_PREFIX_PATH=$TORCH_PREFIX \
  -DTORCH_INSTALL_PREFIX=$TORCH_PREFIX \
  -DMOONCAKE_ROOT=/moriio/${MOONCAKE_ROOT#$ROOT/} \
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
  2>&1 | tee /moriio/${TRTLLM_BUILD_DIR#$ROOT/}/configure.log
cmake --build /moriio/${TRTLLM_BUILD_DIR#$ROOT/} --target tensorrt_llm_mooncake_wrapper --parallel 1 \
  2>&1 | tee /moriio/${TRTLLM_BUILD_DIR#$ROOT/}/build-mooncake-wrapper.log
ls -lh /moriio/${TRTLLM_BUILD_DIR#$ROOT/}/tensorrt_llm/executor/cache_transmission/mooncake_utils/libtensorrt_llm_mooncake_wrapper.so"

sudo chown -R "$(id -u):$(id -g)" "$TRTLLM_BUILD_DIR"
mkdir -p "$IMAGE_CTX"
cp "$TRTLLM_BUILD_DIR/tensorrt_llm/executor/cache_transmission/mooncake_utils/libtensorrt_llm_mooncake_wrapper.so" "$IMAGE_CTX/"
cp "$MOONCAKE_ROOT/lib/libtransfer_engine.so" "$IMAGE_CTX/"
cat > "$IMAGE_CTX/Dockerfile" <<DOCKERFILE
FROM $BASE_IMAGE
USER root
RUN mkdir -p /opt/dynamo/venv/lib/python3.12/site-packages/tensorrt_llm/libs/mooncake
COPY libtensorrt_llm_mooncake_wrapper.so /opt/dynamo/venv/lib/python3.12/site-packages/tensorrt_llm/libs/libtensorrt_llm_mooncake_wrapper.so
COPY libtensorrt_llm_mooncake_wrapper.so /usr/local/lib/libtensorrt_llm_mooncake_wrapper.so
COPY libtransfer_engine.so /opt/dynamo/venv/lib/python3.12/site-packages/tensorrt_llm/libs/mooncake/libtransfer_engine.so
COPY libtransfer_engine.so /usr/local/lib/libtransfer_engine.so
RUN set -eux; \
    if command -v patchelf >/dev/null 2>&1; then \
      patchelf --set-rpath '\$ORIGIN/mooncake:/usr/local/lib' /opt/dynamo/venv/lib/python3.12/site-packages/tensorrt_llm/libs/libtensorrt_llm_mooncake_wrapper.so; \
      patchelf --set-rpath '\$ORIGIN/mooncake:/usr/local/lib' /usr/local/lib/libtensorrt_llm_mooncake_wrapper.so; \
    fi; \
    ldconfig || true; \
    test -f /opt/dynamo/venv/lib/python3.12/site-packages/tensorrt_llm/libs/libtensorrt_llm_mooncake_wrapper.so; \
    test -f /opt/dynamo/venv/lib/python3.12/site-packages/tensorrt_llm/libs/mooncake/libtransfer_engine.so; \
    test -f /usr/local/lib/libtransfer_engine.so
DOCKERFILE

docker build -t "$OUT_IMAGE" "$IMAGE_CTX"
docker image inspect "$OUT_IMAGE" --format '{{.Id}} {{.Size}}'
