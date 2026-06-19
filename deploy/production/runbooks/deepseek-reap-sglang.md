<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DeepSeek REAP SGLang

This runbook validates `BlaiseAI/DeepSeek-V3.2-REAP-345B-SpinQuant-ActKV-NVFP4-NextN-Graft` on the Dynamo production Kubernetes profile with the SGLang backend from `ai-blaise/optimization-playground`.

The active topology runs on one A4 node with eight allocatable B200 GPUs: one
4-GPU prefill worker and two 2-GPU decode replicas. The production profile uses
decode-side HiSparse plus the target model's HF-declared HIGGS dense MLA KV and
NVFP4 IndexCache+HISA paths from `ai-blaise/optimization-playground`:

- target checkpoint: SpinQuant ActKV NVFP4 through the checkpoint's `compressed-tensors` quantization metadata
- target KV source dtype: BF16, with HIGGS dense 2-bit MLA KV storage declared by `quantization_config.kv_cache_scheme`
- decode-side HiSparse via `--enable-hisparse`
- NVFP4 IndexCache+HISA via `quantization_config.indexer_quantization`
- TokenSpeed MLA attention backends via `--nsa-prefill-backend tokenspeed_mla` and `--nsa-decode-backend tokenspeed_mla`
- no SGLang HiCache in this profile because HiSparse requires the decode no-radix path
- LayerSplit on the prefill worker via `--enable-dsa-prefill-context-parallel`, `--attention-context-parallel-size 4`, and `--dsa-prefill-cp-kv-storage-mode layersplit`
- WarpDecode enabled through `SGLANG_ENABLE_WARP_DECODE=1`
- FlashSampling and NCCLX are not enabled in this production profile
- Dynamo event-backed KV-aware routing via frontend `--router-mode kv --router-kv-events` and worker `--kv-events-config`
- Dynamo-native chat preprocessing via frontend `--dyn-chat-processor dynamo` and worker parser flags `--dyn-tool-call-parser deepseek_v3_2 --dyn-reasoning-parser deepseek_r1`
- Dynamo Frontend tokenization via `--tokenizer fastokens`, with HuggingFace decoding and fallback behavior handled inside Dynamo
- prefill: `--disaggregation-mode prefill`, `--dp 1`, `--tp 4`, `--mem-fraction-static 0.66`, `--max-running-requests 32`
- decode: `--disaggregation-mode decode`, `--dp 1`, `--tp 2`, two replicas, `--mem-fraction-static 0.64`, `--max-running-requests 32`, radix cache disabled as required by HiSparse
- SMC-SD draft on decode only: `BlaiseAI/GLM-4-9B-0414-FP8-DeepSeekV32-OMP`, FP8 draft KV, CUTLASS draft FP8 GEMM

Compatibility note: SGLang documents HiSparse as a decode-side DSA/PD feature
that keeps a hot GPU KV buffer and complete CPU pinned-memory KV, while HiCache
is documented as a RadixAttention/HiRadixTree prefix-KV reuse system. This
profile chooses HiSparse when HiSparse and HiCache conflict, but keeps the
HF-declared HIGGS dense KV and NVFP4 IndexCache+HISA implementation.

Activation artifact note: the current target checkpoint carries SpinQuant
activation/KV metadata in the published Hugging Face `config.json`. Keep the
deployment pointed at that artifact rather than synthesizing activation scales at
runtime.

## Production Profile

Use the production profile in this repository as the Kubernetes layer. The infrastructure entry point wraps these steps and should be preferred:

```bash
scripts/dynamo-reap/deploy-a4-production.sh
```

The script applies the full `deploy/production` GitOps stack, including baseline add-ons and optional production integrations, then renders and applies the REAP `DynamoGraphDeployment`.
The infrastructure wrapper treats `opentelemetry-operator` as deployable when it
is Healthy but OutOfSync, because the chart's webhook certificates and CRDs can
be controller-mutated after sync. Other production applications still need to
satisfy the Synced/Healthy gate.

Manual bootstrap is:

```bash
kubectl apply -f deploy/production/gitops/project.yaml
kubectl apply -f deploy/production/gitops/root-app.yaml
kubectl apply -f deploy/production/gitops/optional/keda.yaml
kubectl apply -f deploy/production/gitops/optional/opentelemetry.yaml
kubectl apply -f deploy/production/gitops/optional/actions-runner-controller.yaml
kubectl apply -f deploy/production/gitops/optional/parca.yaml
kubectl apply -f deploy/production/gitops/optional/volcano.yaml
kubectl apply -f deploy/production/gitops/optional/lws.yaml
deploy/pre-deployment/pre-deployment-check.sh --profile production
deploy/pre-deployment/pre-deployment-check.sh --require dynamo-crds,dynamo-webhooks,kai-queue
```

The GitOps manifests read from `https://github.com/ai-blaise/dynamo-prod-k8s.git` on `main`.

## Model Cache

Run the download on every node that can schedule the worker pod. The default manifest mounts the model from:

```text
/models/BlaiseAI/DeepSeek-V3.2-REAP-345B-SpinQuant-ActKV-NVFP4-NextN-Graft
```

and the SMC draft from:

```text
/models/smcsd/GLM-4-9B-0414-FP8-DeepSeekV32-OMP
```

The target model is private in the `BlaiseAI` organization, so the host environment must have an authenticated Hugging Face token.

```bash
export HF_TOKEN=...
export HF_XET_HIGH_PERFORMANCE=1

sudo mkdir -p /models/BlaiseAI /models/smcsd
sudo chown -R "$USER:$USER" /models

python3 -m venv ~/hf-download
source ~/hf-download/bin/activate
python -m pip install --upgrade pip
python -m pip install "huggingface_hub[hf_xet]>=0.36"

hf download \
  BlaiseAI/DeepSeek-V3.2-REAP-345B-SpinQuant-ActKV-NVFP4-NextN-Graft \
  --local-dir /models/BlaiseAI/DeepSeek-V3.2-REAP-345B-SpinQuant-ActKV-NVFP4-NextN-Graft

hf download \
  BlaiseAI/GLM-4-9B-0414-FP8-DeepSeekV32-OMP \
  --local-dir /models/smcsd/GLM-4-9B-0414-FP8-DeepSeekV32-OMP
```

The pod sets `HF_HUB_OFFLINE=1`; model access failures should be fixed during host-side download, not at runtime.

## ModelExpress Cache and P2P Modes

The production stack deploys ModelExpress and wires Dynamo through the native
operator setting `dynamo-operator.modelExpressURL`. With the default manifest,
workers still use absolute `/models/...` paths, so ModelExpress is present but
not on the hot path.

The A4 infrastructure wrapper owns the opt-in modes:

```bash
REAP_ENABLE_MODELEXPRESS=1 scripts/dynamo-reap/deploy-a4-production.sh
```

That mode renders the main worker `--model-path` as the Hugging Face repo id,
mounts the shared `/models` cache, and sets:

```text
HF_HOME=/models
HF_HUB_CACHE=/models/hub
MODEL_EXPRESS_CACHE_PATH=/models/hub
```

Dynamo then calls its native `fetch_model()` path before SGLang starts. The SMC
draft remains the local hostPath unless `REAP_MODELEXPRESS_SMC_DRAFT=1` is set;
with that flag, the renderer changes `--speculative-draft-model-path` to the
draft Hugging Face repo id and Dynamo prefetches it through the same
ModelExpress path.

SGLang remote-instance P2P loading is separate and must stay opt-in:

```bash
REAP_ENABLE_MODELEXPRESS=1 \
REAP_MODELEXPRESS_P2P=1 \
REAP_MODELEXPRESS_TRANSPORT=nixl \
scripts/dynamo-reap/deploy-a4-production.sh
```

The P2P mode adds `--load-format remote_instance`,
`--remote-instance-weight-loader-backend modelexpress`, and
`--modelexpress-config` to prefill and decode workers. It also requires the
runtime image to contain the ModelExpress Python package. Do not make this the
default until it has been validated with compressed-tensors NVFP4, HiSparse,
LayerSplit, and decode-only SMC-SD.

## Engine Image

Build the runtime image from `ai-blaise/optimization-playground` after the remaining custom kernels are added. The manifest defaults to:

```text
ghcr.io/ai-blaise/optimization-playground-sglang-runtime:reap-nvfp4
```

Override this image from the infrastructure script with `DYNAMO_REAP_IMAGE` when testing a candidate image.

The production path should use a registry image and let Kubernetes pull it. The
A4 k3s deployment wrapper also supports a validation fallback: if the registry
tag is unavailable, it can build a local overlay image from the
`ai-blaise/optimization-playground` checkout and import that image into k3s
containerd. That fallback is tracked by Docker image ID so unchanged repeat
launches do not pay the `docker save | k3s ctr images import` cost. The fallback
overlay must keep the engine's required `sglang-kernel` version and should not
downgrade `apache-tvm-ffi` below FlashInfer's supported range.

## Deploy

```bash
kubectl apply -n dynamo-system -f deploy/production/examples/deepseek-v32-reap-sglang.yaml
kubectl get dgd,dgdr,dgdsa,dm,pods -n dynamo-system
kubectl logs -n dynamo-system -l app.kubernetes.io/name=deepseek-v32-reap-sglang --all-containers --tail=200
```

The frontend starts with event-backed KV routing and Dynamo-native chat preprocessing:

```bash
python3 -m dynamo.frontend \
  --router-mode kv \
  --router-kv-events \
  --router-reset-states \
  --dyn-chat-processor dynamo \
  --tokenizer fastokens \
  --http-port 8000
```

The workers launch SGLang through Dynamo with the core stack enabled:

```bash
python3 -m dynamo.sglang \
  --model-path /models/BlaiseAI/DeepSeek-V3.2-REAP-345B-SpinQuant-ActKV-NVFP4-NextN-Graft \
  --served-model-name BlaiseAI/DeepSeek-V3.2-REAP-345B-SpinQuant-ActKV-NVFP4-NextN-Graft \
  --quantization compressed-tensors \
  --kv-cache-dtype bfloat16 \
  --tp 4|2 \
  --dp 1 \
  --mem-fraction-static 0.66|0.64 \
  --max-running-requests 32 \
  --context-length 140000 \
  --max-total-tokens 1048576 \
  --cuda-graph-max-bs 48 \
  --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:5557"}' \
  --nsa-prefill-backend tokenspeed_mla \
  --nsa-decode-backend tokenspeed_mla \
  --disaggregation-transfer-backend nixl \
  --disaggregation-bootstrap-port 12345 \
  --disaggregation-mode prefill|decode \
  --dyn-tool-call-parser deepseek_v3_2 \
  --dyn-reasoning-parser deepseek_r1
```

Omit the HiSparse and SMC-SD flags on the prefill role. The manifest sets them
only on decode:

```bash
--disable-radix-cache \
--enable-hisparse \
--hisparse-config '{"top_k":1024,"device_buffer_size":6144,"host_to_device_ratio":10}' \
--speculative-algorithm SMC
```

SMC-SD draft KV is decode-local in this topology. The prefill/decode transfer
registers target-model KV; it does not register the decode-side draft KV pool
because the prefill worker does not instantiate the draft model.

The `bfloat16` KV dtype is intentional: the HIGGS dense KV path quantizes BF16
MLA KV rows according to the model config, while HiSparse keeps the decode hot
set and host pool in the active compressed row format. Do not add
`--enable-hierarchical-cache` to this production profile without revalidating the
HiSparse no-radix contract.

## Smoke Test

Forward the frontend service and send a minimal OpenAI-compatible request:

```bash
kubectl port-forward -n dynamo-system svc/deepseek-v32-reap-sglang-frontend 8000:8000
```

```bash
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "BlaiseAI/DeepSeek-V3.2-REAP-345B-SpinQuant-ActKV-NVFP4-NextN-Graft",
    "messages": [{"role": "user", "content": "Say: dynamo-ready"}],
    "temperature": 0,
    "max_tokens": 64
  }'
```

## Cleanup

```bash
kubectl delete -n dynamo-system -f deploy/production/examples/deepseek-v32-reap-sglang.yaml
```
