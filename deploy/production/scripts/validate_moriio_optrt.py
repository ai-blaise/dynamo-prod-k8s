#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 BlaiseAI / ai-blaise. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate the op-trt MORI-IO-style disaggregated deployment contract."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml


DEFAULT_EXPECTED_IMAGE = "local/dynamo-trtllm-optrt-custom:moriio-ucx-kvarn-k2v2-optrt-cutedsl-overlay-20260606"


def _die(errors: list[str], msg: str) -> None:
    errors.append(msg)


def _find_configmap(docs: list[dict[str, Any]]) -> dict[str, Any]:
    for doc in docs:
        if doc and doc.get("kind") == "ConfigMap" and doc.get("data"):
            data = doc["data"]
            if "prefill.yaml" in data and "decode.yaml" in data:
                return doc
    raise SystemExit("no ConfigMap with prefill.yaml/decode.yaml found")


def _find_dgd(docs: list[dict[str, Any]]) -> dict[str, Any]:
    for doc in docs:
        if doc and doc.get("kind") == "DynamoGraphDeployment":
            return doc
    raise SystemExit("no DynamoGraphDeployment found")


def _env_map(dgd: dict[str, Any]) -> dict[str, str]:
    return {item.get("name"): str(item.get("value", "")) for item in dgd.get("spec", {}).get("envs", [])}


def _load_worker_configs(cm: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    data = cm["data"]
    return yaml.safe_load(data["prefill.yaml"]), yaml.safe_load(data["decode.yaml"])


def _walk_images(value: Any) -> list[str]:
    if isinstance(value, dict):
        found = []
        for key, child in value.items():
            if key == "image" and isinstance(child, str):
                found.append(child)
            else:
                found.extend(_walk_images(child))
        return found
    if isinstance(value, list):
        found = []
        for child in value:
            found.extend(_walk_images(child))
        return found
    return []


def validate(path: Path, transport: str = "UCX", expected_image: str = DEFAULT_EXPECTED_IMAGE) -> list[str]:
    text = path.read_text()
    docs = [doc for doc in yaml.safe_load_all(text) if doc]
    cm = _find_configmap(docs)
    dgd = _find_dgd(docs)
    prefill, decode = _load_worker_configs(cm)
    env = _env_map(dgd)
    transport = transport.upper()
    errors: list[str] = []
    if transport not in {"UCX", "NIXL", "MOONCAKE"}:
        _die(errors, f"unsupported transport {transport!r}")

    images = _walk_images(dgd)
    if not images:
        _die(errors, "manifest must pin container images")
    for image in images:
        if image != expected_image:
            _die(errors, f"unexpected container image {image!r}; expected {expected_image!r}")

    if prefill.get("cp_config", {}).get("cp_type") != "LAYERSPLIT":
        _die(errors, "prefill cp_config.cp_type must be LAYERSPLIT; HELIX fallback is not allowed")
    if "HELIX" in text:
        _die(errors, "manifest must not contain HELIX")
    if prefill.get("tensor_parallel_size") != 2 or prefill.get("context_parallel_size") != 2:
        _die(errors, "prefill tensor/context parallelism must be TP2xCP2 for the r20 LayerSplit handoff")
    if prefill.get("moe_expert_parallel_size") != 4:
        _die(errors, "prefill moe_expert_parallel_size must remain EP4 for the r20 target")
    if decode.get("tensor_parallel_size") != 4 or decode.get("context_parallel_size", 1) != 1:
        _die(errors, "decode tensor/context parallelism must be TP4xCP1 for the r20 WarpDecode handoff")
    if decode.get("moe_expert_parallel_size") != 4:
        _die(errors, "decode moe_expert_parallel_size must remain EP4 for the r20 target")
    if prefill.get("max_batch_size") != 32 or prefill.get("max_num_tokens") != 8192:
        _die(errors, "prefill max_batch_size/max_num_tokens must match the r20 TP2xCP2 profile")
    if decode.get("max_batch_size") != 64 or decode.get("max_num_tokens") != 2048:
        _die(errors, "decode max_batch_size/max_num_tokens must match the r20 TP4xCP1 profile")

    sparse_prefill = prefill.get("sparse_attention_config", {})
    sparse_decode = decode.get("sparse_attention_config", {})
    if not sparse_prefill.get("layersplit_enabled"):
        _die(errors, "prefill sparse_attention_config.layersplit_enabled must be true")
    if sparse_decode.get("layersplit_enabled"):
        _die(errors, "decode sparse_attention_config.layersplit_enabled must be false for CP1 decode")
    if sparse_prefill.get("layersplit_owner_assignment") != "contiguous":
        _die(errors, "LayerSplit owner assignment must be contiguous for the R20 topology")
    if sparse_prefill.get("layersplit_all_cp_ranks_transfer") is not True:
        _die(errors, "LayerSplit must transfer from all CP ranks")
    if sparse_prefill.get("layersplit_owner_local_alloc") is not False:
        _die(errors, "LayerSplit owner-local allocation must stay false for global block-id transceiver compatibility")
    expected_layersplit_backend = transport.lower()
    if str(sparse_prefill.get("layersplit_transfer_backend", "")).lower() != expected_layersplit_backend:
        _die(
            errors,
            f"LayerSplit transfer backend must be explicit {expected_layersplit_backend}; "
            f"got {sparse_prefill.get('layersplit_transfer_backend')!r}",
        )

    for role, cfg in (("prefill", prefill), ("decode", decode)):
        transceiver = cfg.get("cache_transceiver_config", {})
        backend = transceiver.get("backend")
        runtime = transceiver.get("transceiver_runtime")
        if backend != transport:
            _die(errors, f"{role} cache_transceiver_config.backend must be explicit {transport}, got {backend!r}")
        if runtime not in (None, "CPP", "PYTHON"):
            _die(errors, f"{role} transceiver_runtime must be CPP, PYTHON, or omitted")
        if runtime == "PYTHON" and backend != "NIXL":
            _die(errors, f"{role} Python transceiver supports only NIXL/DEFAULT")
        if transceiver.get("max_tokens_in_buffer", 0) < 131072:
            _die(errors, f"{role} max_tokens_in_buffer must cover the production ISL")
        if transceiver.get("backend_fallback"):
            _die(errors, f"{role} backend_fallback is not allowed; unsupported transports must fail closed")

    prefill_warp = prefill.get("moe_config", {}).get("warp_decode", {})
    if prefill_warp.get("enabled"):
        _die(errors, "prefill WarpDecode must remain disabled; WarpDecode is decode-only for r20")
    if prefill_warp.get("allow_parallelism_fallback") is not None:
        _die(errors, "prefill must not carry WarpDecode fallback policy fields")

    warp = decode.get("moe_config", {}).get("warp_decode", {})
    if not warp.get("enabled") or warp.get("policy") != "force":
        _die(errors, "decode WarpDecode must be enabled with policy=force")
    if warp.get("allow_parallelism_fallback") is not False:
        _die(errors, "decode WarpDecode fallback must be disabled unless explicitly changed")
    if decode.get("moe_config", {}).get("backend") != "WARPDECODE":
        _die(errors, "decode moe_config.backend must be WARPDECODE for the r20 target")
    spec = decode.get("speculative_config", {})
    if spec.get("decoding_type") != "SMC" or not spec.get("speculative_model"):
        _die(errors, "decode SMC speculative_config must stay enabled")

    for role, cfg in (("prefill", prefill), ("decode", decode)):
        dtype = str(
            cfg.get("mla_latent_kv_dtype")
            or cfg.get("kv_cache_config", {}).get("mla_latent_kv_dtype")
            or cfg.get("sparse_attention_config", {}).get("mla_latent_kv_dtype")
            or cfg.get("kv_cache_config", {}).get("dtype")
            or cfg.get("kv_cache_dtype")
            or ""
        ).lower()
        amortize = bool(
            cfg.get("mla_latent_kv_amortize")
            or cfg.get("kv_cache_config", {}).get("mla_latent_kv_amortize")
            or cfg.get("sparse_attention_config", {}).get("mla_latent_kv_amortize")
        )
        if dtype != "kvarn_k2v2":
            _die(errors, f"{role} must use dense MLA latent KVarN kvarn_k2v2 for this MORI composition test")
        if not amortize:
            _die(errors, f"{role} KVarN requires explicit mla_latent_kv_amortize validation")
        for key, value in cfg.items():
            if "indexer" in str(key).lower() and "kvarn" in str(value).lower():
                _die(errors, f"{role} Indexer must not use KVarN as a storage dtype")

    if env.get("DYN_TRTLLM_MORIIO_PINNING_REQUIRED") != "1":
        _die(errors, "DYN_TRTLLM_MORIIO_PINNING_REQUIRED=1 is required")
    mode = env.get("DYN_TRTLLM_MORIIO_TRANSFER_MODE", "write").lower()
    if mode not in ("write", "read"):
        _die(errors, "DYN_TRTLLM_MORIIO_TRANSFER_MODE must be write or explicit read")
    if mode == "read" and env.get("DYN_TRTLLM_MORIIO_ENABLE_READ_MODE") != "1":
        _die(errors, "MORI-IO read mode requires DYN_TRTLLM_MORIIO_ENABLE_READ_MODE=1")
    if mode == "write" and env.get("DYN_TRTLLM_MORIIO_ENABLE_READ_MODE") == "1":
        _die(errors, "do not set DYN_TRTLLM_MORIIO_ENABLE_READ_MODE in write mode")
    if env.get("DYN_TRTLLM_MORIIO_BACKEND") != transport:
        _die(errors, f"DYN_TRTLLM_MORIIO_BACKEND must be {transport} for this validation")
    if not env.get("DYN_TRTLLM_MORIIO_PRODUCER_ID"):
        _die(errors, "DYN_TRTLLM_MORIIO_PRODUCER_ID is required for pin observability")
    if not env.get("DYN_TRTLLM_MORIIO_CONSUMER_ID"):
        _die(errors, "DYN_TRTLLM_MORIIO_CONSUMER_ID is required for pin observability")
    if env.get("DYN_TRTLLM_MORIIO_PRODUCER_TOPOLOGY") != "tp2-cp2-pp1-layersplit-full-layer":
        _die(errors, "DYN_TRTLLM_MORIIO_PRODUCER_TOPOLOGY must pin TP2xCP2 LayerSplit full-layer prefill")
    if env.get("DYN_TRTLLM_MORIIO_CONSUMER_TOPOLOGY") != "tp4-cp1-pp1-warpdecode-smc":
        _die(errors, "DYN_TRTLLM_MORIIO_CONSUMER_TOPOLOGY must pin TP4xCP1 WarpDecode+SMC decode")
    if transport == "UCX":
        if env.get("TRTLLM_USE_UCX_KVCACHE") != "1":
            _die(errors, "TRTLLM_USE_UCX_KVCACHE=1 is required with backend UCX")
        if env.get("TRTLLM_USE_NIXL_KVCACHE") == "1" or env.get("TRTLLM_USE_MOONCAKE_KVCACHE") == "1":
            _die(errors, "NIXL/Mooncake envs must not be enabled in UCX validation")
    elif transport == "NIXL":
        if env.get("TRTLLM_USE_NIXL_KVCACHE") != "1":
            _die(errors, "TRTLLM_USE_NIXL_KVCACHE=1 is required with backend NIXL")
        if env.get("TRTLLM_NIXL_KVCACHE_BACKEND", "UCX") != "UCX":
            _die(errors, "single-node A4 B200 NIXL variant should use UCX plugin unless validated otherwise")
    elif transport == "MOONCAKE":
        if env.get("TRTLLM_USE_MOONCAKE_KVCACHE") != "1":
            _die(errors, "TRTLLM_USE_MOONCAKE_KVCACHE=1 is required with backend MOONCAKE")
    if env.get("UCX_CUDA_IPC_ENABLE_MNNVL") != "0":
        _die(errors, "UCX_CUDA_IPC_ENABLE_MNNVL=0 is required for the current B200 target")
    if env.get("UCX_RNDV_SCHEME") not in (None, "get_zcopy", "put_zcopy"):
        _die(errors, "UCX_RNDV_SCHEME should be unset, get_zcopy, or put_zcopy for B200 validation")
    if env.get("TRTLLM_WARP_DECODE_FIXED_TACTIC") != "1":
        _die(errors, "TRTLLM_WARP_DECODE_FIXED_TACTIC=1 is required for the forced WarpDecode target")
    if env.get("TRTLLM_ENABLE_PDL") != "1":
        _die(errors, "TRTLLM_ENABLE_PDL=1 is required for the forced WarpDecode target")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--transport", choices=("UCX", "NIXL", "MOONCAKE"), default="UCX")
    parser.add_argument("--image", default=DEFAULT_EXPECTED_IMAGE, help="Expected image tag for every container role.")
    args = parser.parse_args()
    errors = validate(args.manifest, transport=args.transport, expected_image=args.image)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"OK: {args.manifest} satisfies MORI-IO op-trt contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
