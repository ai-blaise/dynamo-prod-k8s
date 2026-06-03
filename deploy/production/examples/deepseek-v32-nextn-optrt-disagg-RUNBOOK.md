# DeepSeek-V3.2 NextN-Graft op-trt — Disaggregated Production Deployment Runbook

Target: `BlaiseAI/DeepSeek-V3.2-REAP-345B-SpinQuant-ActKV-NVFP4-NextN-Graft` served via
**op-trt (TensorRT-LLM) workers under Dynamo** on one 8-GPU B200 node (**a4-us-002-rl9**;
001 is off-limits). Optimization target: **tokens/second/user**.

Manifest: `deepseek-v32-nextn-optrt-disagg.yaml` (this directory).

## Topology (DP=4 disaggregated, 8 GPUs)

| Role | replicas×GPU | parallelism | pieces |
|------|-----|------|------|
| Frontend | 1 (CPU) | `dynamo.frontend --router-mode kv` | #7 fastokens (`DYN_TOKENIZER`) |
| prefill | 1×2 | tp1·ep2·**cp2** (attention_dp off) | #2 DSA indexer, #3 LayerSplit CP, chunked-prefill, overlap off |
| decode | 3×2 | tp2·ep2 (attention_dp on, MNNVL) | #2 DSA indexer, #4 WarpDecode, #5 SMC-SD |

KV handoff: `cache_transceiver_config {backend: UCX, max_tokens_in_buffer: 131072}`.

### Why prefill = tp1·ep2·cp2
LayerSplit (#3) requires `cp_size≥2` (a true world dim: world = tp·pp·cp). To keep the
prefill on 2 GPUs *and* get cp2, the bulk expert weight is sharded by EP (tp1·ep2), freeing
the world dim for cp2 (~106 GB/GPU, fits 179). Decode keeps tp2·ep2 (the 64-expert/2-GPU
WarpDecode sweet spot). **Deploy risks to verify:** (a) full-model cp+ep composition is
unvalidated in op-trt (this is the first real-model run); (b) disagg cross-pod LayerSplit
broadcast is future work (layersplit.md M8g) — only same-node NCCL is done. Fallback if cp+ep
is rejected: prefill tp2·cp1 with LayerSplit off, then surface the topology choice.

## The 7 custom pieces — wiring

1. **Disagg P/D** — `--disaggregation-mode prefill|decode`, subComponentType, UCX transceiver.
2. **DSA Indexer** — `sparse_attention_config`: indexcache-hisa, index_topk 1024, indexer_k_dtype fp4,
   enable_nvfp4_hisa (this also activates the ported NVFP4 FlashMLA sparse-decode kernel).
3. **LayerSplit CP** — `sparse_attention_config.layersplit_enabled` (+ round_robin owners, ucx) on prefill (cp2).
4. **WarpDecode** — `moe_config.backend: WARPDECODE` + `warp_decode.tile_mode: autotune` (decode workers).
5. **SMC-SD** — `speculative_config {decoding_type: SMC, speculative_model: GLM-4-9B-…, gamma 6, n_particles 4}` (decode).
6. **NVFP4 FlashMLA KV** — delivered via the ported `cpp/.../flashMLA/nvfp4_sparse` kernel gated by
   `enable_nvfp4_hisa` (#2). Dense `kv_cache_config.dtype` stays **fp8** (model_loader.py:78 guard rejects
   dense nvfp4 MLA KV). Lifting that guard for a dense-NVFP4 latent is an optional post-baseline optimization.
7. **fastokens** — Frontend `DYN_TOKENIZER: fastokens`.

## Runtime image (critical: op-trt HEAD needs a fresh .so)

All May-31 images (`:20260531`, `:hisa-20260531`, base, buildtools) carry op-trt Python missing
WARPDECODE/warp_decode.py/layersplit.py, and a `.so` missing the HISA NVFP4 ops
(`indexer_hisa_mean_pool_nvfp4`, `indexer_hisa_update_page_reps_nvfp4`). op-trt HEAD (`6709f425`)
has all 7 pieces' code but its `dsa.py` calls ops absent from the stale `.so` → import fails.

**Fix = build `libtensorrt_llm.so` + `libth_common.so` from HEAD**, then bake a 7piece image:
1. Repo (no .git) at host `/opt/op-trt-build/TensorRT-LLM-spencer-wd`; op-trt Python at `/opt/op-trt-overlay`.
2. Build via `/opt/optrt_build.sh` in a detached `optrt_build` container (base `hisa-buildtools-20260531`):
   `build_wheel.py --configure-only …` then `cmake --build cpp/build --target tensorrt_llm th_common -j64`.
   cutlass is fetched at configure (FetchContent), not a submodule.
3. Post-build: stage a container FROM `hisa-20260531`, overlay op-trt Python + the two fresh `.so`,
   **import-test** (`python3 -c "import tensorrt_llm"` + TorchLlmArgs parse of decode+prefill),
   `docker commit … :7piece-20260602`, then `k3s ctr -n k8s.io images import`.

## Deploy + validate

```
kubectl -n dynamo-system apply -f deepseek-v32-nextn-optrt-disagg.yaml
kubectl -n dynamo-system wait --for=condition=Ready dgd/deepseek-v32-nextn-optrt-disagg --timeout=1800s
```
Benchmark grid (metric tok/s/user): input {1k,8k,16k,32k,64k,100k,128k} × output 1k × concurrency {16,32,64}
via `infrastructure/scripts/dynamo-trtllm/benchmark-a4-nextn-optrt.py` (64k cell present) against the frontend.

## Hill-climb knobs (KV polling/management focus)
free_gpu_memory_fraction (prefill 0.66 / decode 0.64); enable_block_reuse (prefill on / decode off);
host_cache_size (CPU offload), KV retention/priority; KV-router events (`--publish-events-and-metrics`,
`--router-mode kv` → route to best-cache-hit decode, bypass prefill); cache_transceiver backend (UCX↔NIXL),
max_tokens_in_buffer ≥ max ISL; max_num_tokens (3072↔9216); cuda_graph batch_sizes; SMC gamma/n_particles;
HISA compression_ratio / block_topk; stream_interval; num_postprocess_workers.

## Validated runtime fixes (bring-up, 2026-06-03)

Bringing the stack up required these fixes (all in the manifest + a model-dir prep step):
1. **Absolute model paths** — `--model-path /models/BlaiseAI/...` and `speculative_model: /models/...`
   (an HF repo id triggers an offline Hub lookup → fail). Worker pods mount `/models` (hostPath);
   frontend mounts it too (fastokens).
2. **`HF_MODULES_CACHE=/tmp/hf_modules`** — `trust_remote_code` writes custom modules under `$HF_HOME/modules`,
   but `/models` is read-only.
3. **Single-node comm** — `NCCL_MNNVL_ENABLE=0`, `UCX_CUDA_IPC_ENABLE_MNNVL=0`, `NCCL_NVLS_ENABLE=0`,
   `allreduce_strategy: AUTO` (MNNVL/NVLS are for multi-node GB200 NVL; this is one node).
4. **WarpDecode routing** — decode-only `ENABLE_CONFIGURABLE_MOE=0` so `moe_backend: WARPDECODE` resolves to the
   canonical `CuteDslFusedMoE` (create_moe.get_moe_cls), not the ConfigurableMoE trtllm_gen overlay (which crashes
   `third dimension of weights must equal hidden_size`). `warp_decode.enabled: false` (overlay gate).
5. **Model-dir prep (do BEFORE deploy)** — the REAP-NextN-Graft `tokenizer_config.json` has no `chat_template`
   (frontend registration requires it) and the frontend caps the file at 163631 B. Add a DeepSeek chat_template
   and re-dump the JSON **compact** (`json.dump(..., separators=(",",":"))`) → ~115 KB. Dynamo validates a
   **blake3 checksum** of this file, so the workers must be (re)started AFTER the edit or registration fails with
   "checksum mismatch … no WorkerSet". The production script must apply this prep before starting workers.
6. **op-trt `.so` rebuild from HEAD** (see Runtime image): flashMLA glob exclude + namespace fix + ptxas-13.2
   wrapper + cmake path. Image `7piece-20260602`; `7piece-smcsd-20260603` = + dense GLM-4.

## Pieces #3 / #5 resolution
- **#5 SMC-SD** — DONE in op-trt: the GLM-4-9B draft is dense `Glm4ForCausalLM`; added that arch to
  `_torch/models/modeling_glm.py` (safe-getattr in Glm4DecoderLayer for absent MoE fields + a
  `@register_auto_model("Glm4ForCausalLM")` subclass; the dense GatedMLP branch + Glm4WeightLoader already exist).
  Built into `7piece-smcsd-20260603`; re-enable `speculative_config` on decode + use that image.
- **#3 LayerSplit** — needs `cp_size≥2`, impossible on a 2-GPU prefill for this model (cp=2 ⇒ MoE world=1 ⇒
  EP=2 impossible ⇒ 128 experts can't fit on 1 GPU). LayerSplit-compatible topology = a **4-GPU prefill**
  (tp2·cp2) + 2 decode (the 4+2+2 variant), trading the "1 prefill 2-GPU + 3 decode 2-GPU" split.

## Model quality note
Functionally correct ("Count: 1 2 3 4" → "5 6 7 8 9 10") but greedy-degenerate on some prompts — a quality
tradeoff of the aggressive REAP + NVFP4 W4A4 + fp8-dense-KV stack (`w1≠w3` NVFP4 SwiGLU scale warnings).
tok/s/user (throughput) is content-independent, so the optimization metric is unaffected; #6 (NVFP4 dense KV)
may improve quality by matching the model's ActKV design.

## Deliverables (when done)
Detailed docs; slop cleanup; commit to ai-blaise repos (op-trt DCO `-s`, no co-author); production script to
`ai-blaise/infrastructure/scripts/dynamo-trtllm`; CRIU snapshot/checkpoint for fast startup.
