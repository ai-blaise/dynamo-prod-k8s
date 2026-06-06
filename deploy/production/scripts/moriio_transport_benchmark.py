#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Render/apply MORI-IO transport comparison variants on a4-us-001.

Fail-closed by design: UCX is not treated as NIXL/Mooncake/MoRI. NIXL and
Mooncake variants require TensorRT-LLM wrapper libraries in the selected image;
native MORI requires the `mori` Python package and is expected to be absent on
this CUDA/B200 stack unless separately installed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

WRAPPER_BY_TRANSPORT = {
    "UCX": "libtensorrt_llm_ucx_wrapper.so",
    "NIXL": "libtensorrt_llm_nixl_wrapper.so",
    "MOONCAKE": "libtensorrt_llm_mooncake_wrapper.so",
    "MORI": "mori.io",
    "NATIVE_MORI": "mori.io",
    "UCX_BASELINE": "libtensorrt_llm_ucx_wrapper.so",
    "NIXL_BASELINE": "libtensorrt_llm_nixl_wrapper.so",
    "MOONCAKE_BASELINE": "libtensorrt_llm_mooncake_wrapper.so",
}


def run(cmd: list[str], check: bool = True, capture: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check, text=True, capture_output=capture)


def image_probe(image: str) -> dict[str, Any]:
    script = """
import importlib.util, json, os
probes={}
for name in [\"mori\", \"nixl\", \"tensorrt_llm\"]:
    spec=importlib.util.find_spec(name)
    probes[name]=bool(spec)
libs=[]
for root, _, files in os.walk(\"/\"):
    if not (root.startswith(\"/opt\") or root.startswith(\"/usr/local\") or root.startswith(\"/workspace\")):
        continue
    for f in files:
        if f.endswith(\".so\") or \".so.\" in f:
            if any(k in f for k in [\"tensorrt_llm_ucx_wrapper\", \"tensorrt_llm_nixl_wrapper\", \"tensorrt_llm_mooncake_wrapper\", \"nixl\", \"mooncake\", \"mori\"]):
                libs.append(os.path.join(root, f))
print(json.dumps({\"packages\": probes, \"libs\": sorted(libs)}))
"""
    cp = run(["docker", "run", "--rm", "--entrypoint", "python3", image, "-c", script])
    return json.loads(cp.stdout)


def set_env(envs: list[dict[str, str]], name: str, value: str | None) -> None:
    envs[:] = [e for e in envs if e.get("name") != name]
    if value is not None:
        envs.append({"name": name, "value": str(value)})


def walk_set_image(value: Any, image: str) -> None:
    if isinstance(value, dict):
        if "image" in value:
            value["image"] = image
        for child in value.values():
            walk_set_image(child, image)
    elif isinstance(value, list):
        for child in value:
            walk_set_image(child, image)


def render_variant(base: Path, out: Path, transport: str, image: str, mori: bool) -> None:
    docs = [d for d in yaml.safe_load_all(base.read_text()) if d]
    suffix = f"moriio-{transport.lower()}" if mori else f"baseline-{transport.lower()}"
    for doc in docs:
        meta = doc.get("metadata", {})
        # The Dynamo operator copies the service volume reference by name, so keep
        # the ConfigMap name stable unless the volume references are rewritten too.
        if meta.get("name") and doc.get("kind") != "ConfigMap":
            meta["name"] = f"{meta['name']}-{suffix}"[:63]
        if doc.get("kind") == "ConfigMap":
            for key in ["prefill.yaml", "decode.yaml"]:
                cfg = yaml.safe_load(doc["data"][key])
                cfg["cache_transceiver_config"]["backend"] = transport
                sparse = cfg.get("sparse_attention_config")
                if isinstance(sparse, dict) and sparse.get("layersplit_enabled"):
                    sparse["layersplit_transfer_backend"] = transport.lower()
                doc["data"][key] = yaml.safe_dump(cfg, sort_keys=False)
        if doc.get("kind") == "DynamoGraphDeployment":
            walk_set_image(doc, image)
            envs = doc.setdefault("spec", {}).setdefault("envs", [])
            if mori:
                set_env(envs, "DYN_TRTLLM_MORIIO_PINNING_REQUIRED", "1")
                set_env(envs, "DYN_TRTLLM_MORIIO_TRANSFER_MODE", "write")
                set_env(envs, "DYN_TRTLLM_MORIIO_BACKEND", transport)
            else:
                for key in list(e.get("name") for e in envs):
                    if key and key.startswith("DYN_TRTLLM_MORIIO"):
                        set_env(envs, key, None)
            set_env(envs, "TRTLLM_USE_UCX_KVCACHE", "1" if transport == "UCX" else None)
            set_env(envs, "TRTLLM_USE_NIXL_KVCACHE", "1" if transport == "NIXL" else None)
            set_env(envs, "TRTLLM_USE_MOONCAKE_KVCACHE", "1" if transport == "MOONCAKE" else None)
            set_env(envs, "TRTLLM_NIXL_KVCACHE_BACKEND", "UCX" if transport == "NIXL" else None)
            set_env(envs, "UCX_CUDA_IPC_ENABLE_MNNVL", "0")
    out.write_text("---\n" + "---\n".join(yaml.safe_dump(d, sort_keys=False) for d in docs))


def actual_backend_and_mode(transport: str) -> tuple[str, bool]:
    if transport.endswith("_BASELINE"):
        return transport[: -len("_BASELINE")], False
    if transport in {"MORI", "NATIVE_MORI"}:
        return "UCX", False
    return transport, True


def wrapper_available(probe: dict[str, Any], transport: str) -> bool:
    if transport in {"MORI", "NATIVE_MORI"}:
        return probe["packages"].get("mori", False)
    needle = WRAPPER_BY_TRANSPORT[transport]
    return any(needle in p for p in probe["libs"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=Path("deploy/production/examples/deepseek-v32-nextn-optrt.yaml"))
    ap.add_argument("--image", default="local/dynamo-trtllm-optrt-custom:moriio-ucx-kvarn-k2v2-optrt-cutedsl-overlay-20260606")
    ap.add_argument("--namespace", default="moriio-bench")
    ap.add_argument("--transports", default="UCX,NIXL,MOONCAKE,MORI")
    ap.add_argument("--mode", choices=("render", "apply", "cleanup"), default="render")
    ap.add_argument("--out-dir", type=Path, default=Path("/tmp/moriio-bench"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    probe = image_probe(args.image)
    report = {"image": args.image, "probe": probe, "variants": []}
    transports = [t.strip().upper() for t in args.transports.split(",") if t.strip()]
    for transport in transports:
        actual_transport, mori_semantics = actual_backend_and_mode(transport)
        variant = {"transport": transport, "actual_backend": actual_transport, "mori_semantics": mori_semantics}
        if transport not in WRAPPER_BY_TRANSPORT:
            variant.update(status="blocked", reason="unknown transport")
            report["variants"].append(variant)
            continue
        if not wrapper_available(probe, transport):
            variant.update(status="blocked", reason=f"missing {WRAPPER_BY_TRANSPORT[transport]} in image; cannot benchmark real {transport}")
            report["variants"].append(variant)
            continue
        manifest = args.out_dir / f"moriio-{transport.lower()}.yaml"
        render_variant(args.manifest, manifest, actual_transport, args.image, mori=mori_semantics)
        variant.update(status="rendered", manifest=str(manifest))
        if args.mode == "apply":
            run(["sudo", "-E", "/usr/local/bin/k3s", "kubectl", "create", "namespace", args.namespace], check=False)
            run(["sudo", "-E", "/usr/local/bin/k3s", "kubectl", "-n", args.namespace, "apply", "-f", str(manifest)], capture=False)
            variant["status"] = "applied"
        report["variants"].append(variant)
    if args.mode == "cleanup":
        run(["sudo", "-E", "/usr/local/bin/k3s", "kubectl", "delete", "namespace", args.namespace, "--ignore-not-found=true"], capture=False)
        report["cleanup"] = args.namespace
    report_path = args.out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
