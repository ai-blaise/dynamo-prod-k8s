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
print(json.dumps({"packages": packages, "attrs": attrs, "libs": sorted(libs)}))
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


def probe_image(image: str) -> dict[str, Any]:
    inspect = run(["docker", "image", "inspect", image])
    if inspect.returncode != 0:
        return {"image": image, "status": "missing", "error": inspect.stderr.strip().splitlines()[-1:]}
    cp = run(["docker", "run", "--rm", "--entrypoint", "python3", image, "-c", PACKAGE_SNIPPET])
    if cp.returncode != 0:
        return {"image": image, "status": "probe_failed", "error": (cp.stderr or cp.stdout).strip().splitlines()[-3:]}
    payload = json.loads(cp.stdout)
    return classify({"image": image, "status": "ok", **payload})


def probe_host() -> dict[str, Any]:
    cp = run(["python3", "-c", PACKAGE_SNIPPET])
    if cp.returncode != 0:
        return {"host": True, "status": "probe_failed", "error": (cp.stderr or cp.stdout).strip().splitlines()[-3:]}
    payload = json.loads(cp.stdout)
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
    payload["capabilities"] = {
        "ucx_wrapper": has_lib(payload, "libtensorrt_llm_ucx_wrapper.so"),
        "nixl_wrapper": has_lib(payload, "libtensorrt_llm_nixl_wrapper.so"),
        "mooncake_wrapper": has_lib(payload, "libtensorrt_llm_mooncake_wrapper.so"),
        "mori_wrapper": has_lib(payload, "libtensorrt_llm_mori_wrapper.so"),
        "nixl_python": bool(packages.get("nixl") or packages.get("nixl._api")),
        "native_mori_python": bool(packages.get("mori.io")),
        "native_mori_cpp_transfer_status": "TransferStatus" in cpp_attrs,
        "native_mori_min_api": REQUIRED_MORI_IO.issubset(mori_attrs),
        "mooncake_transfer_engine": has_lib(payload, "libtransfer_engine.so"),
    }
    caps = payload["capabilities"]
    payload["runnable"] = {
        "UCX": caps["ucx_wrapper"],
        "NIXL": caps["nixl_wrapper"] and caps["nixl_python"],
        "MOONCAKE": caps["mooncake_wrapper"] and caps["mooncake_transfer_engine"],
        "NATIVE_MORI": caps["mori_wrapper"] and caps["native_mori_python"] and caps["native_mori_cpp_transfer_status"] and caps["native_mori_min_api"],
    }
    blockers = []
    if not payload["runnable"]["MOONCAKE"]:
        if not caps["mooncake_wrapper"]:
            blockers.append("missing libtensorrt_llm_mooncake_wrapper.so")
        if not caps["mooncake_transfer_engine"]:
            blockers.append("missing libtransfer_engine.so")
    if not payload["runnable"]["NATIVE_MORI"]:
        if not caps["mori_wrapper"]:
            blockers.append("missing libtensorrt_llm_mori_wrapper.so")
        if not caps["native_mori_python"]:
            blockers.append("missing mori.io package")
        if not caps["native_mori_cpp_transfer_status"]:
            blockers.append("missing mori.cpp.TransferStatus")
        if not caps["native_mori_min_api"]:
            blockers.append("missing mori.io minimum IOEngine/MemoryDesc/RDMA API")
    payload["blockers"] = sorted(set(blockers))
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", action="append", default=[], help="Docker image to probe; may be repeated")
    ap.add_argument("--host", action="store_true", help="Probe host Python environment")
    ap.add_argument("--mooncake-root", default=os.environ.get("MOONCAKE_ROOT"))
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    report: dict[str, Any] = {"images": [], "mooncake_root": probe_mooncake_root(args.mooncake_root)}
    if args.host:
        report["host"] = probe_host()
    for image in args.image:
        report["images"].append(probe_image(image))
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
