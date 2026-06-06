#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Probe MORI-IO / NIXL / Mooncake runtime availability without GPUs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

PACKAGE_SNIPPET = r"""
import importlib, importlib.util, json, os
packages = {}
attrs = {}
for name in ["mori", "mori.io", "mori.cpp", "nixl", "nixl._api", "tensorrt_llm"]:
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ModuleNotFoundError):
        spec = None
    packages[name] = bool(spec)
    if spec:
        try:
            mod = importlib.import_module(name)
            attrs[name] = sorted(a for a in dir(mod) if a in [
                "BackendType", "EngineDesc", "IOEngine", "IOEngineConfig",
                "MemoryDesc", "MemoryLocationType", "PollCqMode",
                "RdmaBackendConfig", "TransferStatus", "XgmiBackendConfig",
            ])
        except Exception as exc:
            attrs[name] = {"import_error": repr(exc)}
libs = []
for root, _, files in os.walk("/"):
    if not (root.startswith("/opt") or root.startswith("/usr/local") or root.startswith("/workspace")):
        continue
    for f in files:
        if f.endswith(".so") or ".so." in f:
            if any(k in f for k in [
                "tensorrt_llm_ucx_wrapper", "tensorrt_llm_nixl_wrapper",
                "tensorrt_llm_mooncake_wrapper", "tensorrt_llm_mori_wrapper",
                "transfer_engine", "nixl", "mooncake", "mori",
            ]):
                libs.append(os.path.join(root, f))
lifecycle = {}
# NIXL: construct a CPU-only UCX agent and verify metadata can be produced.
try:
    from nixl import _api as nixl_api
    cfg = nixl_api.nixl_agent_config(
        enable_prog_thread=False,
        enable_listen_thread=False,
        listen_port=0,
        capture_telemetry=False,
        num_threads=0,
        backends=["UCX"],
    )
    agent = nixl_api.nixl_agent("probe-agent", cfg, False)
    metadata = agent.get_agent_metadata()
    lifecycle["NIXL"] = {
        "agent_created": True,
        "agent_metadata_bytes": len(metadata) if hasattr(metadata, "__len__") else None,
    }
except Exception as exc:
    lifecycle["NIXL"] = {"error": repr(exc)}

# Native MORI: mirror vLLM/SGLang's minimal IOEngine lifecycle when available.
try:
    from mori.io import BackendType, EngineDesc, IOEngine, IOEngineConfig, PollCqMode, RdmaBackendConfig
    cfg = IOEngineConfig("127.0.0.1", 0)
    engine = IOEngine("probe-engine", cfg)
    desc = engine.get_engine_desc()
    packed = desc.pack()
    unpacked = EngineDesc.unpack(packed)
    mori_lifecycle = {
        "engine_created": True,
        "engine_desc_pack_roundtrip": bool(getattr(unpacked, "key", None)),
        "engine_port": getattr(desc, "port", None),
    }
    try:
        rdma_cfg = RdmaBackendConfig(1, 1, 1, PollCqMode.POLLING, False)
        engine.create_backend(BackendType.RDMA, rdma_cfg)
        mori_lifecycle["rdma_backend_created"] = True
    except Exception as backend_exc:
        mori_lifecycle["rdma_backend_error"] = repr(backend_exc)
    lifecycle["NATIVE_MORI"] = mori_lifecycle
except Exception as exc:
    lifecycle["NATIVE_MORI"] = {"error": repr(exc)}

# Mooncake: verify Transfer Engine C API symbols when the shared library exists.
try:
    import ctypes
    transfer_lib = next((p for p in libs if p.endswith("libtransfer_engine.so")), None)
    if transfer_lib:
        lib = ctypes.CDLL(transfer_lib)
        required = [
            "createTransferEngine", "destroyTransferEngine", "registerLocalMemory",
            "unregisterLocalMemory", "allocateBatchID", "submitTransfer",
            "getTransferStatus", "freeBatchID",
        ]
        missing = [name for name in required if not hasattr(lib, name)]
        lifecycle["MOONCAKE"] = {"c_api_symbols_present": not missing, "missing_symbols": missing}
    else:
        lifecycle["MOONCAKE"] = {"error": "libtransfer_engine.so not found"}
except Exception as exc:
    lifecycle["MOONCAKE"] = {"error": repr(exc)}

print(json.dumps({"packages": packages, "attrs": attrs, "libs": sorted(libs), "lifecycle": lifecycle}))
"""

REQUIRED_MORI_IO = {
    "BackendType",
    "EngineDesc",
    "IOEngine",
    "IOEngineConfig",
    "MemoryDesc",
    "PollCqMode",
    "RdmaBackendConfig",
}


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=True)


def parse_probe_stdout(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise ValueError("probe did not emit JSON payload")


def probe_image(image: str) -> dict[str, Any]:
    inspect = run(["docker", "image", "inspect", image])
    if inspect.returncode != 0:
        return {"image": image, "status": "missing", "error": inspect.stderr.strip().splitlines()[-1:]}
    cp = run(["docker", "run", "--rm", "--entrypoint", "python3", image, "-c", PACKAGE_SNIPPET])
    if cp.returncode != 0:
        return {"image": image, "status": "probe_failed", "error": (cp.stderr or cp.stdout).strip().splitlines()[-3:]}
    try:
        payload = parse_probe_stdout(cp.stdout)
    except Exception as exc:
        return {"image": image, "status": "probe_failed", "error": [repr(exc), cp.stdout[-2000:]]}
    return classify({"image": image, "status": "ok", **payload})


def probe_host() -> dict[str, Any]:
    cp = run(["python3", "-c", PACKAGE_SNIPPET])
    if cp.returncode != 0:
        return {"host": True, "status": "probe_failed", "error": (cp.stderr or cp.stdout).strip().splitlines()[-3:]}
    try:
        payload = parse_probe_stdout(cp.stdout)
    except Exception as exc:
        return {"host": True, "status": "probe_failed", "error": [repr(exc), cp.stdout[-2000:]]}
    return classify({"host": True, "status": "ok", **payload})


def probe_mooncake_root(root: str | None) -> dict[str, Any] | None:
    if not root:
        return None
    prefix = Path(root)
    header = prefix / "include" / "transfer_engine_c.h"
    shared = prefix / "lib" / "libtransfer_engine.so"
    static = prefix / "lib" / "libtransfer_engine.a"
    return {
        "path": str(prefix),
        "include_transfer_engine_c_h": header.exists(),
        "libtransfer_engine_so": shared.exists(),
        "libtransfer_engine_a": static.exists(),
        "usable_for_trtllm_wrapper": header.exists() and shared.exists(),
    }


def has_lib(payload: dict[str, Any], needle: str) -> bool:
    return any(needle in p for p in payload.get("libs", []))


def classify(payload: dict[str, Any]) -> dict[str, Any]:
    packages = payload.get("packages", {})
    attrs = payload.get("attrs", {})
    mori_attrs = set(attrs.get("mori.io") or []) if isinstance(attrs.get("mori.io"), list) else set()
    cpp_attrs = set(attrs.get("mori.cpp") or []) if isinstance(attrs.get("mori.cpp"), list) else set()
    lifecycle = payload.get("lifecycle", {})
    nixl_lifecycle = lifecycle.get("NIXL", {}) if isinstance(lifecycle.get("NIXL"), dict) else {}
    mori_lifecycle = lifecycle.get("NATIVE_MORI", {}) if isinstance(lifecycle.get("NATIVE_MORI"), dict) else {}
    mooncake_lifecycle = lifecycle.get("MOONCAKE", {}) if isinstance(lifecycle.get("MOONCAKE"), dict) else {}
    payload["capabilities"] = {
        "ucx_wrapper": has_lib(payload, "libtensorrt_llm_ucx_wrapper.so"),
        "nixl_wrapper": has_lib(payload, "libtensorrt_llm_nixl_wrapper.so"),
        "mooncake_wrapper": has_lib(payload, "libtensorrt_llm_mooncake_wrapper.so"),
        "mori_wrapper": has_lib(payload, "libtensorrt_llm_mori_wrapper.so"),
        "nixl_python": bool(packages.get("nixl") or packages.get("nixl._api")),
        "nixl_agent_lifecycle": bool(nixl_lifecycle.get("agent_created") and nixl_lifecycle.get("agent_metadata_bytes")),
        "native_mori_python": bool(packages.get("mori.io")),
        "native_mori_cpp_transfer_status": "TransferStatus" in cpp_attrs,
        "native_mori_min_api": REQUIRED_MORI_IO.issubset(mori_attrs),
        "native_mori_engine_lifecycle": bool(mori_lifecycle.get("engine_created") and mori_lifecycle.get("engine_desc_pack_roundtrip")),
        "native_mori_rdma_backend_lifecycle": bool(mori_lifecycle.get("rdma_backend_created")),
        "mooncake_transfer_engine": has_lib(payload, "libtransfer_engine.so"),
        "mooncake_c_api_symbols": bool(mooncake_lifecycle.get("c_api_symbols_present")),
    }
    caps = payload["capabilities"]
    payload["runnable"] = {
        "UCX": caps["ucx_wrapper"],
        "NIXL": caps["nixl_wrapper"] and caps["nixl_python"] and caps["nixl_agent_lifecycle"],
        "MOONCAKE": caps["mooncake_wrapper"] and caps["mooncake_transfer_engine"] and caps["mooncake_c_api_symbols"],
        "NATIVE_MORI": caps["mori_wrapper"] and caps["native_mori_python"] and caps["native_mori_cpp_transfer_status"] and caps["native_mori_min_api"] and caps["native_mori_engine_lifecycle"] and caps["native_mori_rdma_backend_lifecycle"],
    }
    backend_blockers: dict[str, list[str]] = {"UCX": [], "NIXL": [], "MOONCAKE": [], "NATIVE_MORI": []}
    if not caps["ucx_wrapper"]:
        backend_blockers["UCX"].append("missing libtensorrt_llm_ucx_wrapper.so")
    if not caps["nixl_wrapper"]:
        backend_blockers["NIXL"].append("missing libtensorrt_llm_nixl_wrapper.so")
    if not caps["nixl_python"]:
        backend_blockers["NIXL"].append("missing nixl/nixl._api Python package")
    if not caps["nixl_agent_lifecycle"]:
        backend_blockers["NIXL"].append("NIXL agent metadata lifecycle failed")
    if not caps["mooncake_wrapper"]:
        backend_blockers["MOONCAKE"].append("missing libtensorrt_llm_mooncake_wrapper.so")
    if not caps["mooncake_transfer_engine"]:
        backend_blockers["MOONCAKE"].append("missing libtransfer_engine.so")
    if not caps["mooncake_c_api_symbols"]:
        backend_blockers["MOONCAKE"].append("Mooncake Transfer Engine C API symbol probe failed")
    if not caps["mori_wrapper"]:
        backend_blockers["NATIVE_MORI"].append("missing libtensorrt_llm_mori_wrapper.so")
    if not caps["native_mori_python"]:
        backend_blockers["NATIVE_MORI"].append("missing mori.io package")
    if not caps["native_mori_cpp_transfer_status"]:
        backend_blockers["NATIVE_MORI"].append("missing mori.cpp.TransferStatus")
    if not caps["native_mori_min_api"]:
        backend_blockers["NATIVE_MORI"].append("missing mori.io minimum IOEngine/MemoryDesc/RDMA API")
    if not caps["native_mori_engine_lifecycle"]:
        backend_blockers["NATIVE_MORI"].append("native MORI IOEngine metadata lifecycle failed")
    if not caps["native_mori_rdma_backend_lifecycle"]:
        backend_blockers["NATIVE_MORI"].append("native MORI RDMA backend lifecycle failed")
    payload["backend_blockers"] = {k: v for k, v in backend_blockers.items() if v}
    payload["blockers"] = sorted({item for values in backend_blockers.values() for item in values})
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", action="append", default=[], help="Docker image to probe; may be repeated")
    ap.add_argument("--host", action="store_true", help="Probe host Python environment")
    ap.add_argument("--mooncake-root", default=os.environ.get("MOONCAKE_ROOT"))
    ap.add_argument("--out", type=Path)
    ap.add_argument("--require", action="append", choices=("UCX", "NIXL", "MOONCAKE", "NATIVE_MORI"), default=[], help="Fail if any probed image lacks this runnable backend")
    args = ap.parse_args()

    report: dict[str, Any] = {"images": [], "mooncake_root": probe_mooncake_root(args.mooncake_root)}
    if args.host:
        report["host"] = probe_host()
    for image in args.image:
        report["images"].append(probe_image(image))
    failures = []
    for payload in report["images"]:
        runnable = payload.get("runnable", {})
        for backend in args.require:
            if not runnable.get(backend):
                reason = "; ".join(payload.get("backend_blockers", {}).get(backend, [])) or payload.get("status", "not runnable")
                failures.append(f"{payload.get('image')}: {backend} not runnable ({reason})")
    if failures:
        report["requirement_failures"] = failures
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(text)
    if failures:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
