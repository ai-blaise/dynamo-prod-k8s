# MORI-IO-style op-trt disaggregated KV handoff

This runbook documents the CUDA/B200 adaptation of MORI-IO semantics for the custom op-trt stack. It does not load AMD MoRI on B200. Instead, it preserves the MORI-IO producer/consumer contract, request pinning, write-mode default, and explicit backend selection over TensorRT-LLM cache transceiver.

## Parent base requirements

MORI-IO is a follow-on only. Do not promote it until the parent non-MORI image includes op-trt commit `4c62f74` for OpenAI disagg request pinning, `eebb2db` for context-response-authoritative pinning metadata, and `0ae9399` for MPI control RPC fanout, plus the committed LayerSplit CP-shrink/native fixes and the snapshot/checkpoint guard. Do not bypass `TRTLLM_SNAPSHOT_HOOKS` for live CUDA/distributed state. Current runnable MORI overlay image is `local/dynamo-trtllm-optrt-custom:moriio-ucx-kvarn-k2v2-optrt-cutedsl-overlay-20260606` with digest `sha256:5e7ad82cc34c2075d29505eee56c3f6a45a1eb4feacfc1a66393c92833a7d3f5`, built on `local/dynamo-trtllm-optrt-custom:canonical-smc-r20-cpfix-ucx-mpirpc-kvarn2-ls-mlp-cutedsl-20260605`.

## Contract

- Prefill owns context compute with TP2xCP2 LayerSplit CP and EP4. `cp_config.cp_type` must be `LAYERSPLIT`; HELIX is not a fallback.
- Decode owns steady-state generation with TP4xCP1, EP4, forced WarpDecode, and SMC-SD enabled.
- The default transfer mode is `write`, represented by `DYN_TRTLLM_MORIIO_TRANSFER_MODE=write`, because it is the MORI-IO mode that overlaps prefill compute and KV transfer.
- Read mode is allowed only with `DYN_TRTLLM_MORIIO_TRANSFER_MODE=read` and `DYN_TRTLLM_MORIIO_ENABLE_READ_MODE=1`.
- Request pinning is required with `DYN_TRTLLM_MORIIO_PINNING_REQUIRED=1`; the Dynamo TRT-LLM backend stamps `_moriio_pin` into the actual context response `disaggregated_params` and decode validates it before using the handoff. Actual context response metadata wins for `ctx_dp_rank`, `ctx_info_endpoint`, and opaque transfer state; static `server_info` is backfill only.
- On the current B200 image, the only runnable TensorRT-LLM cache transceiver wrapper is direct `UCX`: use `DYN_TRTLLM_MORIIO_BACKEND=UCX`, `cache_transceiver_config.backend: UCX`, `TRTLLM_USE_UCX_KVCACHE=1`, and `UCX_CUDA_IPC_ENABLE_MNNVL=0`. Do not claim this is equivalent to real NIXL or Mooncake. Real TRT-LLM NIXL is available only in the separate NIXL wrapper image listed below; real Mooncake requires `libtensorrt_llm_mooncake_wrapper.so`; native MoRI requires `mori.io`. The UCX interim overlay still lacks all non-UCX runtime proof and must fail closed for unavailable transports.

## Validation

Run before applying any op-trt MORI-IO manifest:

```bash
python3 deploy/production/scripts/validate_moriio_optrt.py   deploy/production/examples/deepseek-v32-nextn-optrt.yaml
```

The validator checks LayerSplit/no-HELIX, forced WarpDecode/no fallback, SMC-SD, explicit UCX/NIXL/Mooncake transceiver selection, write/read-mode policy, dense MLA KVarN `kvarn_k2v2` with Indexer `fp4`, and request pinning envs. The default manifest validates the current explicit UCX overlay artifact.

## Status

Implemented: source-level request pinning hooks in Dynamo TRT-LLM, explicit deployment env/config, and manifest validation.
Validated: static manifest validation and Python compile/unit-level pinning helper checks.
Experimental: full write-mode concurrent decode preallocation in TensorRT-LLM C++ equivalent to vLLM MoRIIO `save_kv_layer`; current CUDA path uses TensorRT-LLM disaggregated transceiver semantics over direct UCX because NIXL/Mooncake wrapper libraries are absent in the image.

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
for TOKENS in 1000 4000 16000 32000 64000 128000; do
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

Required report fields are TTFT, TPOT/ITL, tokens/sec/user after first token, aggregate streamed chunks, abort-trigger result, GPU utilization/memory/power samples, and the logs around `PREFILL: attached MORI-IO pin`, `DECODE: validated MORI-IO pin`, wait/ready, and release/abort cleanup. Do not compare UCX against NIXL/Mooncake until their TensorRT-LLM wrapper libraries are present and the transport render probe marks those variants runnable. NIXL is now renderable only with `local/dynamo-trtllm-optrt-custom:moriio-nixl-kvarn-k2v2-optrt-cutedsl-overlay-20260606`; Mooncake remains blocked.

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

The previous blocker `libtensorrt_llm_nixl_wrapper.so` has been removed for the isolated MORI workspace. The wrapper was configured and built inside the existing CUDA/TRT-LLM overlay container without `--gpus`, using `/opt/nvidia/nvda_nixl` from the image and the target `tensorrt_llm_nixl_wrapper` only. Reproduce with:

```bash
ROOT=/home/spencergarnets/moriio-agent-20260605T1451Z \
  deploy/production/scripts/build_moriio_nixl_wrapper_image.sh
```

Produced image:

```text
local/dynamo-trtllm-optrt-custom:moriio-nixl-kvarn-k2v2-optrt-cutedsl-overlay-20260606
sha256:dca8f7e0ffaadb1d646bd7214705013e135a9e814792689e8e1ddcfc09652484
```

The transport probe now renders both `UCX` and real TRT-LLM `NIXL` variants for that image. This is not performance proof: it only proves the wrapper and manifest are runnable candidates once GPUs are free. The NIXL C++ agent still honors `TRTLLM_NIXL_KVCACHE_BACKEND`; set it explicitly to `UCX` for the current B200 baseline unless a different NIXL plugin is intentionally tested.

## Wrapper Build Blocker

The current image contains `libtensorrt_llm_ucx_wrapper.so` only. TensorRT-LLM source loads NIXL through `libtensorrt_llm_nixl_wrapper.so` and Mooncake through `libtensorrt_llm_mooncake_wrapper.so` from `cpp/include/tensorrt_llm/executor/transferAgent.h`. CMake only creates the NIXL wrapper when `NIXL_ROOT` is set (`cpp/tensorrt_llm/CMakeLists.txt` and `cpp/tensorrt_llm/executor/cache_transmission/nixl_utils/CMakeLists.txt`). Mooncake additionally requires `MOONCAKE_ROOT` plus `transfer_engine` library/header (`cpp/tensorrt_llm/executor/cache_transmission/mooncake_utils/CMakeLists.txt`). `scripts/build_wheel.py` only packages these wrappers if the built `.so` files exist. Removing this blocker requires a TensorRT-LLM C++/wheel rebuild with those roots available, then rerunning the same transport harness.
