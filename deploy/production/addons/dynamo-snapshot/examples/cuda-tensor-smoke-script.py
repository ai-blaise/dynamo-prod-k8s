# SPDX-FileCopyrightText: Copyright (c) 2026 BlaiseAI / ai-blaise. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Pairs with cuda-tensor-smoke.yaml. Allocates a deterministic-content GPU
# tensor, runs a matmul to warm the CUDA context, touches the upstream
# Dynamo Snapshot "ready for checkpoint" signal file, then enters a
# steady-state loop printing the tensor sum. Snapshot at iter=N, restore,
# and both the iter counter and the tensor sum survive the round trip.
import os
import time

import torch

control_dir = os.environ.get("DYN_SNAPSHOT_CONTROL_DIR", "/snapshot-control")
os.makedirs(control_dir, exist_ok=True)

print(f"[cuda-smoke] torch={torch.__version__} cuda={torch.version.cuda}", flush=True)
assert torch.cuda.is_available(), "no CUDA"
device = torch.device("cuda:0")
print(
    f"[cuda-smoke] device={torch.cuda.get_device_name(0)} "
    f"count={torch.cuda.device_count()}",
    flush=True,
)

# Deterministic content so we can prove HBM state survives the checkpoint.
# sum(range(4096)) = 4095 * 4096 / 2 = 8_386_560.
torch.manual_seed(42)
tensor = torch.arange(0, 4096, dtype=torch.float32, device=device).reshape(64, 64)
expected_sum = tensor.sum().item()
print(
    f"[cuda-smoke] allocated 64x64 tensor, sum={expected_sum:.1f} "
    f"(expect 8386560.0)",
    flush=True,
)

# Warm the CUDA context with a small kernel.
y = tensor @ tensor.T
print(f"[cuda-smoke] matmul ok, y[0,0]={y[0,0].item():.1f}", flush=True)

# Blog protocol: workload signals it has reached a snapshottable state
# after engine init but before distributed-runtime startup. We have no
# distributed runtime here; the signal just unblocks the checkpoint Job's
# readiness probe.
ready_path = os.path.join(control_dir, "ready-for-checkpoint")
with open(ready_path, "w") as fp:
    fp.write("ready")
print(f"[cuda-smoke] touched {ready_path}; entering steady-state loop", flush=True)

iteration = 0
while True:
    print(
        f"[cuda-smoke] iter={iteration} tensor.sum={tensor.sum().item():.1f}",
        flush=True,
    )
    iteration += 1
    time.sleep(10)
