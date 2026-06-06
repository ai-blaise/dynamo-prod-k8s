# MORI-IO-style op-trt disaggregated KV handoff

This runbook documents the CUDA/B200 adaptation of MORI-IO semantics for the custom op-trt stack. It does not load AMD MoRI on B200. Instead, it preserves the MORI-IO producer/consumer contract, request pinning, write-mode default, and explicit backend selection over TensorRT-LLM cache transceiver.

## Parent base requirements

MORI-IO is a follow-on only. Do not promote it until the parent non-MORI image includes op-trt commit `4c62f74` for OpenAI disagg request pinning, `eebb2db` for context-response-authoritative pinning metadata, and `0ae9399` for MPI control RPC fanout, plus the committed LayerSplit CP-shrink/native fixes and the snapshot/checkpoint guard. Do not bypass `TRTLLM_SNAPSHOT_HOOKS` for live CUDA/distributed state. Current restored full-source parent image is `local/dynamo-trtllm-optrt-custom:canonical-smc-r20-fullsrc-ls-kvarn2-nvlsfix-smcfi-hostreq-20260606` with Docker/k3s digest `sha256:9f57cbe58c376fcb515faa3272e91855958e580627305816fa5bc8f8d8afe1f6`. The isolated NIXL derivative for A/B is `local/dynamo-trtllm-optrt-custom:canonical-smc-r20-fullsrc-ls-kvarn2-nvlsfix-smcfi-hostreq-nixl-20260606` with digest `sha256:c3acebccb6454ba3060337c744267bf59357c635d98d744d2b9875a6635b0bc9`, imported into k3s `k8s.io`. The previous smcfi NIXL derivative was `sha256:17944510ec58a574c909b2016d2463295cb4a5a5d744392a7a520642da2e3b43`. The earlier full-source parent image was `local/dynamo-trtllm-optrt-custom:canonical-smc-r20-fullsrc-ls-kvarn2-nvlsfix-20260606` with digest `sha256:6cdb3efe2d317e45dead468103a5854c729e53ff2a31cb9fb3cc5de7ce979f71`, and its NIXL derivative was `sha256:7970f957f8de955f00f1d2bf9687b97ec6c19268de700d41a34bb20011f6dd9f`. The older UCX MORI-style overlay remains `local/dynamo-trtllm-optrt-custom:moriio-ucx-kvarn-k2v2-optrt-cutedsl-overlay-20260606` with digest `sha256:5e7ad82cc34c2075d29505eee56c3f6a45a1eb4feacfc1a66393c92833a7d3f5`, built on `local/dynamo-trtllm-optrt-custom:canonical-smc-r20-cpfix-ucx-mpirpc-kvarn2-ls-mlp-cutedsl-20260605`.

## Contract

- Prefill owns context compute with TP2xCP2 LayerSplit CP and EP4. `cp_config.cp_type` must be `LAYERSPLIT`; HELIX is not a fallback.
- Decode owns steady-state generation with TP4xCP1, EP4, forced WarpDecode, and SMC-SD enabled.
- The default transfer mode is `write`, represented by `DYN_TRTLLM_MORIIO_TRANSFER_MODE=write`, because it is the MORI-IO mode that overlaps prefill compute and KV transfer.
- Read mode is allowed only with `DYN_TRTLLM_MORIIO_TRANSFER_MODE=read` and `DYN_TRTLLM_MORIIO_ENABLE_READ_MODE=1`.
- Request pinning is required with `DYN_TRTLLM_MORIIO_PINNING_REQUIRED=1`; the Dynamo TRT-LLM backend stamps `_moriio_pin` into the actual context response `disaggregated_params` and decode validates it before using the handoff. Actual context response metadata wins for `ctx_dp_rank`, `ctx_info_endpoint`, and opaque transfer state; static `server_info` is backfill only.
- The full-source parent image already carries TensorRT-LLM UCX and NIXL wrappers plus `/opt/nvidia/nvda_nixl`; the isolated NIXL derivative rebuilds the NIXL wrapper from the matching full-source tree and bundles NIXL runtime libs under the TensorRT-LLM package path for hermetic A/B. Direct UCX remains the functional baseline: use `DYN_TRTLLM_MORIIO_BACKEND=UCX`, `cache_transceiver_config.backend: UCX`, `TRTLLM_USE_UCX_KVCACHE=1`, and `UCX_CUDA_IPC_ENABLE_MNNVL=0`. NIXL A/B must use `DYN_TRTLLM_MORIIO_BACKEND=NIXL`, `cache_transceiver_config.backend: NIXL`, `TRTLLM_USE_NIXL_KVCACHE=1`, and `TRTLLM_NIXL_KVCACHE_BACKEND=UCX`. Do not claim UCX, NIXL, Mooncake, or native MORI wins without the same workload measured on B200. Mooncake still requires `libtensorrt_llm_mooncake_wrapper.so`; native MoRI requires `mori.io`.

## Validation

Run before applying any op-trt MORI-IO manifest:

```bash
python3 deploy/production/scripts/validate_moriio_optrt.py   deploy/production/examples/deepseek-v32-nextn-optrt.yaml
```

The validator checks LayerSplit/no-HELIX, forced WarpDecode/no fallback, SMC-SD, explicit UCX/NIXL/Mooncake transceiver selection, write/read-mode policy, dense MLA KVarN `kvarn_k2v2` with Indexer `fp4`, and request pinning envs. The default manifest validates the current explicit UCX overlay artifact.

## Status

Implemented: source-level request pinning hooks in Dynamo TRT-LLM, explicit deployment env/config, and manifest validation.
Validated: static manifest validation, Python compile/unit-level pinning helper checks, full-source NIXL wrapper image build, transport render probe for NIXL, and Kubernetes server dry-run for the full-source NIXL r20 variant.
Experimental: full write-mode concurrent decode preallocation in TensorRT-LLM C++ equivalent to vLLM MoRIIO `save_kv_layer`; current CUDA path uses TensorRT-LLM disaggregated transceiver semantics. Real NIXL is now a runnable candidate once GPUs are free, but it has not been E2E or performance validated. Mooncake and native MORI remain blocked by missing runtime/API dependencies.

## Pin Fields

The prefill worker attaches `_moriio_pin` inside `disaggregated_params`; decode removes it before constructing TensorRT-LLM dataclasses and validates it against decoded params. Required fields include:

- `transfer_id=trtllm:<disagg_request_id>`
- `client_request_id`
- `disagg_request_id`
- `transfer_mode`
- `cache_transceiver_backend`
- `producer_id`
- `consumer_id`

Set producer/consumer labels in the deployment envs for log observability:

```yaml
- name: DYN_TRTLLM_MORIIO_PRODUCER_ID
  value: optrt-prefill-layersplit
- name: DYN_TRTLLM_MORIIO_CONSUMER_ID
  value: optrt-decode-warpdecode-smc
```

## E2E Gate After Parent Rollout

Do not promote this manifest until the parent image/deployment is stable and committed. The transport harness renders/apply variants and probes runtime support; the streaming benchmark client records TTFT, TPOT/ITL, and tokens/sec/user after first token. Example after the frontend service is reachable:

```bash
python3 deploy/production/scripts/moriio_openai_benchmark.py \
  --url http://<frontend-service-ip>:8000 \
  --model BlaiseAI/DeepSeek-V3.2-REAP-345B-SpinQuant-ActKV-NVFP4-NextN-Graft \
  --requests 8 --concurrency 2 --max-tokens 256 --abort-one \
  --out /tmp/moriio-bench-ucx/openai-metrics.json
```

For isolated 001 validation:

1. Build or pull an image that includes the Dynamo pinning hooks in both `handler_base.py` and `llm_engine.py`.
2. Apply the manifest in an isolated namespace.
3. Send one short request, one long-prefill request, one retry, and one abort/cancel case.
4. Check logs for `PREFILL: attached MORI-IO pin` and `DECODE: validated MORI-IO pin`.
5. Check TRT-LLM args for `cache_transceiver_config.backend='UCX'` and `TRTLLM_USE_UCX_KVCACHE=1` in the current runnable overlay. Re-run the same test with NIXL/Mooncake only after their wrapper libraries are present.
6. Check no `HELIX` path is active, LayerSplit is active only on prefill, WarpDecode is forced on decode, and SMC draft traffic is present.
7. Compare ITL/tok/s/user after first token against the parent baseline before considering TTFT tradeoffs.

## Source-grounded Integration Map

Direct source audit as of June 6 08:12 UTC:

- vLLM native MORI-IO: `vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py` imports `mori.io` and fails availability when absent (lines 74-85). Its scheduler binds `request_id` to `transfer_id`, stores recv/save queues, and maps transfer IDs (lines 270-323). Write mode has decode allocate local blocks, then notify prefill with `transfer_id`, decode block IDs, and remote notify port (lines 426-450). `request_finished` defers producer block free, returns `remote_block_ids`/`remote_engine_id` for read mode, and frees via `finished_sending` or timeout (lines 558-627, 629-650). The toy proxy embeds producer/consumer ZMQ endpoints into `request_id`, creates one `transfer_id`, and dispatches prefill/decode concurrently in write mode (examples/disaggregated/.../moriio_toy_proxy_server.py lines 259-325).
- vLLM native MORI runtime API: `moriio_engine.py` imports `BackendType`, `EngineDesc`, `IOEngine`, `IOEngineConfig`, `MemoryDesc`, `PollCqMode`, `RdmaBackendConfig`, and `XgmiBackendConfig` from `mori.io` (lines 45-54). The writer queues per-layer `WriteTask`s, waits for decode remote allocation, synchronizes the CUDA event, builds/reuses sessions, and marks failed transfers done to avoid leaks (lines 64-190, 223-260). This is the minimum native API we are missing in the current B200 image.
- SGLang native MORI: `python/sglang/srt/disaggregation/mori/conn.py` imports `mori.cpp.TransferStatus` and the same `mori.io` primitives (lines 16-26). It tracks rooms/bootstrap addresses, handles multi-rank success/failure, cleans room mappings, sends metadata containing decode KV indices/state, and has explicit sender/receiver abort/clear paths (lines 490-541, 1459-1575). This confirms native MORI cannot run in our image until `mori.io`/`mori.cpp` are installed and CUDA/B200 compatible.
- SGLang NIXL: `python/sglang/srt/disaggregation/nixl/conn.py` requires `nixl._api`, validates the selected plugin, registers VRAM/DRAM memory, pre-builds transfer descriptor lists, and has heterogeneous TP handling (lines 247-288, 383-390, 516-560, 900-940). This supports NIXL as the closest feasible optimized path on B200 when TRT-LLM wrappers exist.
- SGLang Mooncake: `python/sglang/srt/disaggregation/mooncake/conn.py` still contains a TODO that NVLink transport is not bug-free for auxiliary sends (lines 829-833), shards transfer queues by destination sessions for early abort behavior (lines 1519-1524), and traces aborts (lines 1710-1713). In this TRT-LLM image it remains blocked by missing `libtensorrt_llm_mooncake_wrapper.so`, plus missing Mooncake Transfer Engine SDK headers/libs for wrapper build.
- TRT-LLM transport abstraction: `cpp/include/tensorrt_llm/executor/transferAgent.h` defines register/deregister memory, load/invalidate remote agents, submit transfers, notifications, and descriptor checks (lines 371-423). It dynamically loads `libtensorrt_llm_nixl_wrapper.so` or `libtensorrt_llm_mooncake_wrapper.so` by backend name (lines 461-480). `cacheTransceiver.cpp` chooses UCX/NIXL/Mooncake from explicit backend/envs (lines 72-104), constructs NIXL/Mooncake `AgentConnectionManager`s (lines 288-302), sends context KV asynchronously/layer-wise, receives decode KV async/sync, checks completion, and cancels requests through sender/receiver (lines 382-435, 655-827).
- TRT-LLM NIXL performance guard: `baseTransBuffer.cpp` warns dynamic buffers may fail with NIXL and recommends `cache_transceiver_config.max_tokens_in_buffer` covering maximum ISL (lines 148-158). The r20 validator requires `131072`. `cacheFormatter.cpp` and `mlaCacheFormatter.cpp` explain why non-preallocated cudaMallocAsync buffers require copies for UCX GPU-direct RDMA (lines 87-90 and 322-324), so the A/B must watch transfer bandwidth/copy overhead.
- Dynamo/op-trt pinning hooks: `components/src/dynamo/trtllm/utils/moriio_pinning.py` creates a JSON-safe pin with `client_request_id`, `disagg_request_id`, `transfer_id`, transfer mode/backend, producer/consumer IDs/topologies, machine ID, and a `remote_block_metadata_owner=trtllm-cache-transceiver-opaque-state` marker (lines 17-142). It rejects literal MORI/MORIIO on B200 without native runtime (lines 60-75), pops the pin before TRT dataclass construction (lines 169-173), validates router worker/machine ID (lines 196-229), and validates mode/backend/schema/handoff (lines 232-260+). `handler_base.py` and `llm_engine.py` attach pins at prefill and validate them at decode before importing KV (handler lines 516-660; engine lines 1017-1088).
- Deployment guardrails: `validate_moriio_optrt.py` rejects HELIX, enforces TP2xCP2 LayerSplit prefill and TP4xCP1 decode, dense MLA KVarN `kvarn_k2v2` with Indexer not KVarN, forced WarpDecode/no fallback, SMC, explicit transport backend, write/read policy, NIXL/UCX/Mooncake envs, and `UCX_CUDA_IPC_ENABLE_MNNVL=0` (lines 66-210).

Current implementation state: UCX and NIXL are runnable candidates through TRT-LLM cache transceiver plus MORI-style deterministic pinning. This is not native MORI-IO ownership yet because Python/Dynamo does not own remote block IDs, decode preallocation, per-layer writes, or wait/ready; TRT-LLM C++ owns those via opaque disaggregated params and cache transceiver state. Native MORI requires adding a TensorRT-LLM transfer backend or sidecar that exposes `mori.io`-equivalent memory descriptors/sessions/notifications, then extending the pin with native remote block/write metadata instead of `trtllm-cache-transceiver-opaque-state`.

## hostreq NIXL Candidate

Built without GPUs from op-trt `538627d93` and parent image `local/dynamo-trtllm-optrt-custom:canonical-smc-r20-fullsrc-ls-kvarn2-nvlsfix-smcfi-hostreq-20260606`:

```text
local/dynamo-trtllm-optrt-custom:canonical-smc-r20-fullsrc-ls-kvarn2-nvlsfix-smcfi-hostreq-nixl-20260606
sha256:c3acebccb6454ba3060337c744267bf59357c635d98d744d2b9875a6635b0bc9
```

It is imported into k3s containerd with `k3s ctr -n k8s.io images import`. Validate only, while canary owns GPUs:

```bash
python3 deploy/production/scripts/validate_moriio_optrt.py \
  deploy/production/examples/deepseek-v32-nextn-optrt-hostreq-nixl.yaml \
  --transport NIXL \
  --image local/dynamo-trtllm-optrt-custom:canonical-smc-r20-fullsrc-ls-kvarn2-nvlsfix-smcfi-hostreq-nixl-20260606

sudo -E /usr/local/bin/k3s kubectl -n dynamo-system apply --dry-run=server \
  -f deploy/production/examples/deepseek-v32-nextn-optrt-hostreq-nixl.yaml
```

When GPUs are explicitly free, the first real A/B apply command is:

```bash
sudo -E /usr/local/bin/k3s kubectl -n dynamo-system apply \
  -f deploy/production/examples/deepseek-v32-nextn-optrt-hostreq-nixl.yaml
```

Then benchmark with the same workload used for UCX baseline:

```bash
URL=http://<frontend>:8000 \
OUT=/tmp/moriio-bench-hostreq-nixl \
deploy/production/scripts/run_moriio_openai_matrix.sh
```

Required comparison: UCX baseline, UCX with MORI-style pinning, NIXL baseline, NIXL with MORI-style pinning, then Mooncake/native MORI only after their blockers are removed. Do not claim a winner until the 16-user 1k/8k/32k/64k/128k table includes TTFT, ITL/TPOT, tokens/sec/user after first token, failure/abort behavior, and KV transfer metrics.




## Current swapabodd Dependency Probe

The latest parent image seen in this workspace is `local/dynamo-trtllm-optrt-custom:canonical-smc-r20-fullsrc-ls-kvarn2-nvlsfix-smcfi-multidecode-swapabodd-20260606`, built from the SMC odd-draft SwapAB tuning fix. A non-GPU dependency probe shows the image is UCX/NIXL-runnable at the package/library level and still not native-MORI or Mooncake runnable:

- runnable: `UCX`, `NIXL`
- blocked: `MOONCAKE` because `libtensorrt_llm_mooncake_wrapper.so` and `libtransfer_engine.so` are absent
- blocked: `NATIVE_MORI` because `mori.io`, `mori.cpp.TransferStatus`, `libtensorrt_llm_mori_wrapper.so`, and the minimum `mori.io` IOEngine/MemoryDesc/RDMA API are absent

The checked-in current-image NIXL manifest is `deploy/production/examples/deepseek-v32-nextn-optrt-swapabodd-nixl.yaml`. It is only a prepared A/B candidate; do not apply it while the parent canary owns GPUs.

## Current multidecode Dependency Probe

The current parent image `local/dynamo-trtllm-optrt-custom:canonical-smc-r20-fullsrc-ls-kvarn2-nvlsfix-smcfi-multidecode-20260606` has been probed without GPUs. It contains `libtensorrt_llm_ucx_wrapper.so`, `libtensorrt_llm_nixl_wrapper.so`, NIXL Python packages, and `/opt/nvidia/nvda_nixl`, so UCX and NIXL are library-runnable candidates. It does not contain `mori.io`, `mori.cpp.TransferStatus`, `libtensorrt_llm_mori_wrapper.so`, `libtensorrt_llm_mooncake_wrapper.so`, or `libtransfer_engine.so`, so native MORI and Mooncake remain blocked.

The checked-in current-image NIXL manifest is `deploy/production/examples/deepseek-v32-nextn-optrt-multidecode-nixl.yaml`. Validate it with:

```bash
python3 deploy/production/scripts/validate_moriio_optrt.py \
  deploy/production/examples/deepseek-v32-nextn-optrt-multidecode-nixl.yaml \
  --transport NIXL \
  --image local/dynamo-trtllm-optrt-custom:canonical-smc-r20-fullsrc-ls-kvarn2-nvlsfix-smcfi-multidecode-20260606
```

Probe any candidate image before applying it:

```bash
deploy/production/scripts/probe_moriio_deps.py \
  --image local/dynamo-trtllm-optrt-custom:canonical-smc-r20-fullsrc-ls-kvarn2-nvlsfix-smcfi-multidecode-20260606 \
  --image local/dynamo-trtllm-optrt-custom:canonical-smc-r20-fullsrc-ls-kvarn2-nvlsfix-smcfi-multidecode-mooncake-20260606 \
  --image local/dynamo-trtllm-optrt-custom:canonical-smc-r20-fullsrc-ls-kvarn2-nvlsfix-smcfi-multidecode-mori-20260606
```

## Native MORI Code Ownership Plan

Native MORI is not implemented until the following file-level ownership is complete and validated. The current `_moriio_pin` is deliberately marked `remote_block_metadata_owner=trtllm-cache-transceiver-opaque-state`; native MORI must replace that with explicit MORI transfer ownership rather than rebranding UCX/NIXL.

TRT-LLM/op-trt C++ ownership:

- Add a MORI transfer wrapper under `cpp/tensorrt_llm/executor/cache_transmission/mori_utils/` with `CMakeLists.txt`, `transferAgent.cpp`, and any wrapper-local headers. It must expose the same `BaseAgentWrapper` entry points that `transferAgent.h` loads today for NIXL/Mooncake: register/deregister memory, load/invalidate remote agent, submit descriptors, notify, check transfers, and agent metadata.
- Extend `cpp/include/tensorrt_llm/executor/transferAgent.h` only if the current dynamic loader cannot map a new backend string to `libtensorrt_llm_mori_wrapper.so`. Keep NIXL/Mooncake loading behavior intact.
- Extend `cpp/tensorrt_llm/batch_manager/cacheTransceiver.cpp` and adjacent config parsing to accept an explicit `MORI` backend, construct the MORI `AgentConnectionManager`, and route send/recv/cancel through the same fail-closed paths used by NIXL/Mooncake. Do not silently fall back to UCX.
- Preserve LayerSplit CP/TP reassembly in the existing cache formatter paths. MORI must receive the same TP2xCP2 prefill to TP4xCP1 decode block mapping and must not reintroduce HELIX or owner-local assumptions.
- Add package/build plumbing so `scripts/build_wheel.py` and image packaging include `libtensorrt_llm_mori_wrapper.so` only when the native MORI SDK is present.

Dynamo and OpenAI service ownership:

- Extend `components/src/dynamo/trtllm/utils/moriio_pinning.py` and `tensorrt_llm/serve/moriio_pinning.py` with a `native_mori` payload containing `transfer_id`, `producer_engine_id`, `consumer_engine_id`, handshake endpoint, notify endpoint, remote allocation IDs, block IDs or peer write descriptors, layer/stride metadata, and cleanup generation.
- Keep the existing authoritative context-response behavior in `tensorrt_llm/serve/openai_disagg_service.py`: `ctx_dp_rank`, `ctx_info_endpoint`, and `encoded_opaque_state` from the real prefill response win; server_info is backfill only.
- Wire abort/cancel cleanup in `components/src/dynamo/trtllm/request_handlers/handler_base.py` and `components/src/dynamo/trtllm/llm_engine.py` so decode allocation is released and producer-side pending writes are cancelled on success, error, retry, and client abort.
- Add tests beside `components/src/dynamo/trtllm/tests/test_moriio_pinning.py` and `tests/unittest/disaggregated/test_openai_disagg_service.py` for stale transfer IDs, wrong consumer engine, retry with new transfer generation, abort cleanup, read-mode opt-in, and fail-closed unsupported backend.

Direct source behavior to mirror:

- vLLM `MoRIIOConnector`, `MoRIIOConnectorScheduler`, and `MoRIIOConnectorWorker` in `vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py`: request_id to transfer_id binding, save/recv queues, read-mode `remote_block_ids`/`remote_engine_id`, write-mode decode allocation followed by producer notification, and delayed producer free.
- vLLM `MoRIIOWriter`, `WriteTask`, `MoRIIOWrapper`, `RemoteAllocInfo`, and `MoRIIOAgentMetadata` in `moriio_engine.py`: `mori.io` IOEngine/session setup, CUDA event synchronization, per-layer write tasks, cached sessions, failure completion marking, and background progress.
- SGLang `MoriKVManager`, `MoriKVSender`, `MoriKVReceiver`, and bootstrap helpers in `python/sglang/srt/disaggregation/mori/conn.py`: room/bootstrap identity, sender/receiver metadata, multi-rank success/failure accounting, `send_metadata`, `clear`, and `abort`.

Native build dependencies still missing:

- Python/C++ package exports for `mori.io` and `mori.cpp.TransferStatus` with CUDA/B200 support. vLLM and SGLang import these symbols directly; the current hostreq and NIXL images do not contain them.
- Native runtime symbols equivalent to `BackendType`, `EngineDesc`, `IOEngine`, `IOEngineConfig`, `MemoryDesc`, `MemoryLocationType`, `PollCqMode`, `RdmaBackendConfig`, and `XgmiBackendConfig`.
- A vendoring plan that installs MORI under an isolated prefix such as `/opt/mori` and packages only `libtensorrt_llm_mori_wrapper.so` plus required runtime libraries into derivative images. UCX and NIXL envs must remain explicit and unaffected.

## Mooncake Build Ownership Plan

TRT-LLM already has Mooncake wrapper source at `cpp/tensorrt_llm/executor/cache_transmission/mooncake_utils/CMakeLists.txt`. It only builds when `MOONCAKE_ROOT` points at an SDK containing `include/transfer_engine_c.h` and `lib/libtransfer_engine.so`; the wrapper links `transfer_engine` and `CUDA::cudart` into `libtensorrt_llm_mooncake_wrapper.so`.

The Mooncake source clone at `/home/spencergarnets/moriio-agent-20260605T1451Z/Mooncake` provides the needed SDK pieces: `mooncake-transfer-engine/include/transfer_engine_c.h`, `mooncake-transfer-engine/include/CMakeLists.txt` install rules, and the `transfer_engine` target in `mooncake-transfer-engine/src/CMakeLists.txt`. It also requires system dependencies including yaml-cpp, JsonCpp, glog, gflags, ibverbs/libnuma, ASIO/yalantinglibs, and CUDA headers. The checked-in build harness is `deploy/production/scripts/build_moriio_mooncake_wrapper_image.sh`; it is non-GPU but may be CPU/package heavy and should be run only in a maintenance window or with a prebuilt SDK:

```bash
MOONCAKE_ROOT=/path/to/mooncake-sdk \
BASE_IMAGE=local/dynamo-trtllm-optrt-custom:canonical-smc-r20-fullsrc-ls-kvarn2-nvlsfix-smcfi-hostreq-20260606 \
OUT_IMAGE=local/dynamo-trtllm-optrt-custom:canonical-smc-r20-fullsrc-ls-kvarn2-nvlsfix-smcfi-hostreq-mooncake-20260606 \
  deploy/production/scripts/build_moriio_mooncake_wrapper_image.sh
```

Without `MOONCAKE_ROOT`, the script attempts a minimal Transfer Engine build in the base container with `BUILD_SHARED_LIBS=ON`, `WITH_TE=ON`, store/EP/examples/tests disabled, and `USE_CUDA=ON`. If configure fails on missing third-party packages, the exact unblocker is to install Mooncake dependencies or provide a prebuilt SDK; do not mark Mooncake runnable until `libtensorrt_llm_mooncake_wrapper.so` and `libtransfer_engine.so` are both present in the derivative image and `moriio_transport_benchmark.py` renders the `MOONCAKE` variant.

## A/B Matrix Driver

`deploy/production/scripts/run_moriio_transport_ab_matrix.sh` is render-only by default and launches no GPU work. It prepares UCX baseline, UCX+pinning, NIXL baseline, NIXL+pinning, Mooncake, and native MORI variants against the same r20 manifest. It intentionally reports Mooncake/native MORI as blocked when their wrapper/package is absent. After GPUs are explicitly free, use `MODE=apply` for one variant at a time, wait for readiness, then run the OpenAI matrix:

```bash
OUT=/tmp/moriio-transport-ab \
MODE=render \
  deploy/production/scripts/run_moriio_transport_ab_matrix.sh

URL=http://<frontend>:8000 \
OUT=/tmp/moriio-bench-<variant> \
  deploy/production/scripts/run_moriio_openai_matrix.sh
```

The default prompt sweep is `1000 8000 32000 64000 128000` at 16 requests and concurrency 16. Override with `PROMPT_TOKENS`, `REQUESTS`, `CONCURRENCY`, and `MAX_TOKENS` only when comparing every transport with the same values.


## Promotion Gate

Do not promote a backend by label alone. Before applying or benchmarking a new image tag, run the dependency gate with the backend being tested. The command must return zero for a backend to be considered runnable at the dependency level:

```bash
# UCX/NIXL should pass on current full-source canary images.
deploy/production/scripts/probe_moriio_deps.py \
  --image local/dynamo-trtllm-optrt-custom:<tag> \
  --require UCX \
  --require NIXL

# Native MORI is expected to fail until mori.io/mori.cpp and the TRT-LLM wrapper exist.
deploy/production/scripts/probe_moriio_deps.py \
  --image local/dynamo-trtllm-optrt-custom:<tag> \
  --require NATIVE_MORI

# Mooncake is expected to fail until libtransfer_engine.so and the TRT-LLM wrapper exist.
deploy/production/scripts/probe_moriio_deps.py \
  --image local/dynamo-trtllm-optrt-custom:<tag> \
  --require MOONCAKE
```

Passing this dependency gate is still not sufficient for promotion. A promotion candidate must also pass request-pinning, wait/ready, cleanup/abort, LayerSplit + dense MLA KVarN + HISA + WarpDecode + SMC E2E, and the 16-user `1000 8000 32000 64000 128000` benchmark against every runnable baseline.

## Failure Policy

Unsupported mode/backend combinations must fail closed. Do not set `DYN_TRTLLM_MORIIO_BACKEND=NIXL` unless `libtensorrt_llm_nixl_wrapper.so` exists in the image, and do not enable Mooncake unless `libtensorrt_llm_mooncake_wrapper.so` exists and passes the DeepSeek MLA + LayerSplit + SMC benchmark. Do not use `DYN_TRTLLM_MORIIO_BACKEND=moriio` on CUDA/B200 without a real `mori.io` package and CUDA-compatible path.

## Strict Routing Checks

Decode must reject a pinned handoff when the router-injected `worker_id` modulo 1024 does not match `_moriio_pin.disagg_machine_id`. This catches stale or cross-replica prefill results before TensorRT-LLM imports KV. Producer and consumer labels are also validated from env so a manifest cannot silently pair a LayerSplit prefill with an incompatible decode deployment.

## Long-context Benchmark Matrix

After the isolated namespace is healthy, run the same request count, concurrency, model, output length, and prompt family for every runnable transport. The benchmark script supports deterministic synthetic prompt lengths and `nvidia-smi` sampling so the 1k..128k requirement can be replayed without host tokenizer imports:

```bash
FRONTEND_URL=http://<moriio-frontend>:8000
MODEL=BlaiseAI/DeepSeek-V3.2-REAP-345B-SpinQuant-ActKV-NVFP4-NextN-Graft
OUT=/tmp/moriio-bench-cutedsl-ucx
mkdir -p "$OUT"
for TOKENS in 1000 8000 32000 64000 128000; do
  python3 deploy/production/scripts/moriio_openai_benchmark.py \
    --url "$FRONTEND_URL" \
    --model "$MODEL" \
    --requests 16 \
    --concurrency 16 \
    --prompt-tokens "$TOKENS" \
    --max-tokens 256 \
    --abort-one \
    --sample-nvidia-smi \
    --out "$OUT/openai-${TOKENS}.json"
done
```


Shortcut after the frontend URL is known:

```bash
URL=http://<moriio-frontend>:8000 \
OUT=/tmp/moriio-bench-<variant> \
deploy/production/scripts/run_moriio_openai_matrix.sh
```

Required report fields are TTFT, TPOT/ITL, tokens/sec/user after first token, aggregate streamed chunks, abort-trigger result, GPU utilization/memory/power samples, and the logs around `PREFILL: attached MORI-IO pin`, `DECODE: validated MORI-IO pin`, wait/ready, and release/abort cleanup. Do not compare UCX against NIXL/Mooncake until their TensorRT-LLM wrapper libraries are present and the transport render probe marks those variants runnable. NIXL is renderable with `local/dynamo-trtllm-optrt-custom:canonical-smc-r20-fullsrc-ls-kvarn2-nvlsfix-smcfi-hostreq-nixl-20260606` and its checked-in manifest `deploy/production/examples/deepseek-v32-nextn-optrt-hostreq-nixl.yaml`; Mooncake remains blocked.

## Optimization Notes

Keep write mode as the default because it preserves concurrent prefill/decode dispatch. The default overlay intentionally keeps `cache_transceiver_config.backend: UCX` as the only currently runnable baseline. NIXL, Mooncake, and native MORI require separate explicit manifests plus isolated ITL/correctness measurements after their runtime wrappers are present; do not infer expected wins from UCX-only data.

## Lifecycle Log Evidence

During isolated E2E, collect logs for all of these events:

- `PREFILL: attached MORI-IO pin`
- `DECODE: validated MORI-IO pin`
- `DECODE: tracking active MORI-IO handoff`
- `DECODE: released active MORI-IO handoff`
- abort/cancel logs that include `active MORI-IO handoff`

These logs prove deterministic request association and cleanup observation. They do not move KV ownership into Python; TensorRT-LLM still owns remote block allocation, wait/ready state, transfer completion, and GPU-memory release.

## KVarN Boundary

KVarN is dense MLA latent KV cache only (`mla_latent_kv_dtype` / `mla_latent_kv_amortize`). Do not configure KVarN as an Indexer storage dtype. The Indexer reads the dequantized latent / sparse-MLA view, matching `docs/blaise/kvarn.md` and `docs/blaise/indexer.md`. The MORI composition manifest configures dense MLA latent KVarN as `mla_latent_kv_dtype: kvarn_k2v2` with `mla_latent_kv_amortize: true` while leaving Indexer K as `indexer_k_dtype: fp4`. Do not claim E2E KVarN correctness until an isolated request proves dense side-pool P/D seeding and transfer.


## Topology Pinning

The manifest pins producer topology with `DYN_TRTLLM_MORIIO_PRODUCER_TOPOLOGY=tp2-cp2-pp1-layersplit-full-layer` and consumer topology with `DYN_TRTLLM_MORIIO_CONSUMER_TOPOLOGY=tp4-cp1-pp1-warpdecode-smc`. This explicitly captures the LayerSplit CP/TP/PP remap contract and prevents a pinned KV handoff from floating to an incompatible decode instance.


## NIXL Wrapper Artifact

The previous blocker `libtensorrt_llm_nixl_wrapper.so` has been removed for the isolated MORI workspace and for the current `hostreq` full-source parent A/B derivative. The wrapper was configured and built inside the existing CUDA/TRT-LLM container without `--gpus`, using `/opt/nvidia/nvda_nixl` from the image and the target `tensorrt_llm_nixl_wrapper` only. Reproduce the current derivative with:

```bash
ROOT=/home/spencergarnets/moriio-agent-20260605T1451Z \
OPTRT_ROOT=/home/spencergarnets/moriio-agent-20260605T1451Z/TensorRT-LLM-fullsrc-nixl-538627d \
BASE_IMAGE=local/dynamo-trtllm-optrt-custom:canonical-smc-r20-fullsrc-ls-kvarn2-nvlsfix-smcfi-hostreq-20260606 \
OUT_IMAGE=local/dynamo-trtllm-optrt-custom:canonical-smc-r20-fullsrc-ls-kvarn2-nvlsfix-smcfi-hostreq-nixl-20260606 \
BUILD_DIR=/home/spencergarnets/moriio-agent-20260605T1451Z/build/nixl-wrapper-fullsrc-538627d-hostreq \
IMAGE_CTX=/home/spencergarnets/moriio-agent-20260605T1451Z/build/moriio-hostreq-nixl-wrapper-image \
  deploy/production/scripts/build_moriio_nixl_wrapper_image.sh
```

Produced image:

```text
local/dynamo-trtllm-optrt-custom:canonical-smc-r20-fullsrc-ls-kvarn2-nvlsfix-smcfi-hostreq-nixl-20260606
sha256:c3acebccb6454ba3060337c744267bf59357c635d98d744d2b9875a6635b0bc9

legacy pre-smcfi full-source candidate:
local/dynamo-trtllm-optrt-custom:canonical-smc-r20-fullsrc-ls-kvarn2-nvlsfix-nixl-20260606
sha256:7970f957f8de955f00f1d2bf9687b97ec6c19268de700d41a34bb20011f6dd9f

legacy overlay candidate:
local/dynamo-trtllm-optrt-custom:moriio-nixl-kvarn-k2v2-optrt-cutedsl-overlay-20260606
sha256:dca8f7e0ffaadb1d646bd7214705013e135a9e814792689e8e1ddcfc09652484
```

The transport probe now renders both `UCX` and real TRT-LLM `NIXL` variants for the current `hostreq` derivative image. This is not performance proof: it only proves the wrapper and manifest are runnable candidates once GPUs are free. The NIXL C++ agent still honors `TRTLLM_NIXL_KVCACHE_BACKEND`; set it explicitly to `UCX` for the current B200 baseline unless a different NIXL plugin is intentionally tested.

## Remaining Wrapper/API Blockers

NIXL is no longer blocked at wrapper-build time in this workspace: TensorRT-LLM source loads it through `libtensorrt_llm_nixl_wrapper.so` from `cpp/include/tensorrt_llm/executor/transferAgent.h`, and the full-source derivative contains that `.so` plus NIXL runtime libs. It is still unproven until an isolated GPU E2E and 16-user benchmark run complete. Mooncake remains blocked because `libtensorrt_llm_mooncake_wrapper.so` is absent; TensorRT-LLM CMake additionally requires `MOONCAKE_ROOT` plus `transfer_engine` library/header (`cpp/tensorrt_llm/executor/cache_transmission/mooncake_utils/CMakeLists.txt`). Native MORI remains blocked because the image has no CUDA/B200-compatible `mori.io` package/API. `scripts/build_wheel.py` only packages transport wrappers if the built `.so` files exist, so Mooncake removal requires building the Mooncake transfer engine SDK first, then rebuilding the TRT-LLM wrapper and rerunning the same transport harness.
