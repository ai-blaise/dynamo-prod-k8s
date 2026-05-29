<!--
SPDX-FileCopyrightText: Copyright (c) 2026 BlaiseAI / ai-blaise. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Dynamo Snapshot Runbook (upstream `snapshot-agent` + `snapshotctl`)

Operational playbook for the `dynamo-snapshot` addon — the upstream NVIDIA
Dynamo Snapshot feature described in
[NVIDIA's blog post][blog], shipped by the chart at
[`deploy/helm/charts/snapshot/`](../../helm/charts/snapshot/) and wired into
gitops via [`deploy/production/gitops/apps/56-dynamo-snapshot.yaml`](../gitops/apps/56-dynamo-snapshot.yaml).

This runbook covers the runtime, not the controller-driven DGDS path. For
DGDS, see [`criu-snapshots.md`](criu-snapshots.md). For the side-by-side
contract see
[`addons/dynamo-snapshot/README.md`](../addons/dynamo-snapshot/README.md).

## Pre-flight

1. Both addons are healthy on the target node.

   ```bash
   k3s kubectl -n dynamo-system get daemonset/dynamo-snapshot-agent pvc/snapshot-pvc
   k3s kubectl -n criu-snapshots get pods
   ```

   Agent pod must be `1/1 Running`; PVC must be `Bound`.

2. `snapshotctl` is on the host. Build it once from this repo:

   ```bash
   cd deploy/snapshot
   CGO_ENABLED=0 go build -trimpath -buildvcs=false -ldflags="-s -w" \
       -o /usr/local/bin/snapshotctl ./cmd/snapshotctl
   /usr/local/bin/snapshotctl --help
   ```

3. The worker container image you intend to snapshot ships
   `cuda-checkpoint`, `nsrestore`, **and** `criu` on `$PATH` plus the
   CRIU runtime libs. The agent runs `cuda-checkpoint`/CRIU host-side at
   checkpoint time but at restore time it `nsenter`s the placeholder
   pod's PID namespace and execs `/usr/local/sbin/criu` plus
   `/usr/local/bin/nsrestore` inside the worker, so all three plus their
   shared-library deps must be present. The blog-shipped agent image
   (`nvcr.io/nvidia/ai-dynamo/snapshot-agent`) has them under
   `/usr/local/sbin/{criu,criu-ns,cuda-checkpoint}` and
   `/usr/local/bin/{nsrestore,cuda-checkpoint-helper}`. The full
   verified Dockerfile snippet (Ubuntu-24.04 base):

   ```dockerfile
   USER root
   RUN apt-get update \
       && apt-get install -y --no-install-recommends \
           libbsd0 libcap2 libnet1 libnl-3-200 libnl-route-3-200 \
           libprotobuf-c1 libgnutls30t64 libnftables1 libmnl0 libnftnl11 \
           iproute2 iptables procps \
       && rm -rf /var/lib/apt/lists/*
   COPY --from=nvcr.io/nvidia/ai-dynamo/snapshot-agent:1.1.1 \
        /usr/local/sbin/cuda-checkpoint /usr/local/sbin/cuda-checkpoint
   COPY --from=nvcr.io/nvidia/ai-dynamo/snapshot-agent:1.1.1 \
        /usr/local/sbin/criu /usr/local/sbin/criu
   COPY --from=nvcr.io/nvidia/ai-dynamo/snapshot-agent:1.1.1 \
        /usr/local/sbin/criu-ns /usr/local/sbin/criu-ns
   COPY --from=nvcr.io/nvidia/ai-dynamo/snapshot-agent:1.1.1 \
        /usr/local/bin/nsrestore /usr/local/bin/nsrestore
   COPY --from=nvcr.io/nvidia/ai-dynamo/snapshot-agent:1.1.1 \
        /usr/local/bin/cuda-checkpoint-helper /usr/local/bin/cuda-checkpoint-helper
   RUN ln -sf /usr/local/sbin/cuda-checkpoint /usr/local/bin/cuda-checkpoint \
       && chmod 0755 /usr/local/sbin/{cuda-checkpoint,criu,criu-ns} \
            /usr/local/bin/{nsrestore,cuda-checkpoint-helper}
   ```

   **Always pass `--disable-cuda-checkpoint-job-file` to
   `snapshotctl checkpoint`** against today's NVIDIA-published
   `cuda-checkpoint` binary (v580.105.08, latest commit on
   `github.com/NVIDIA/cuda-checkpoint` dated 2025-09-15). The upstream
   snapshotctl protocol wraps the workload command with
   `cuda-checkpoint --launch-job`, but that subcommand is not actually
   implemented in the published binary — only
   `--action lock|checkpoint|restore|unlock`. The agent's own
   `cuda-checkpoint-helper` calls the `cuCheckpointProcess*` driver APIs
   directly against the live PID, so for single-GPU workloads HBM
   capture works fully without the launch-job wrap. Multi-GPU work
   that strictly requires `--launch-job` is blocked on a newer
   `cuda-checkpoint` release from NVIDIA.

4. The workload writes `${DYN_SNAPSHOT_CONTROL_DIR}/ready-for-checkpoint`
   (an empty file) when it has finished engine init but before bringing
   distributed runtime (NCCL, NIXL, KV router) up. For SGLang under our
   stack this happens via `python/sglang/srt/snapshot_hooks.py` when
   `SGLANG_SNAPSHOT_HOOKS=1`. The chart's checkpoint Job wraps the
   workload with a readiness probe that `cat`s this file.

## Checkpoint

```bash
# Drop a Pod manifest. Must include the workload container under
# spec.containers, the image must have cuda-checkpoint + nsrestore on PATH.
cat > /tmp/worker.yaml <<'EOF'
apiVersion: v1
kind: Pod
metadata:
  name: reap-prefill-snapshot
  namespace: dynamo-system
spec:
  nodeName: a4-us-002-rl9
  restartPolicy: Never
  containers:
    - name: main
      image: ghcr.io/ai-blaise/optimization-playground-sglang-runtime:reap-nvfp4-dynamo-1.1.1
      command: [...]   # the exact prefill or decode launch command
      env:
        - { name: SGLANG_SNAPSHOT_HOOKS, value: "1" }
EOF

# Run the checkpoint. The --disable-cuda-checkpoint-job-file flag is
# mandatory until NVIDIA's published cuda-checkpoint binary implements
# --launch-job (see pre-flight). The agent's cuda-checkpoint-helper
# still does the HBM dump via cuCheckpointProcessCheckpoint regardless.
sudo KUBECONFIG=/etc/rancher/k3s/k3s.yaml \
  /usr/local/bin/snapshotctl checkpoint \
    --manifest /tmp/worker.yaml \
    --container main \
    --checkpoint-id reap-prefill-$(date +%s) \
    --disable-cuda-checkpoint-job-file \
    --timeout 30m
```

Successful output ends with:

```
status=completed
checkpoint_location=/checkpoints/<id>/versions/1
```

The artifact lands on the `snapshot-pvc` PVC at the printed path. Files
include CRIU images (`mm-*.img`, `core-*.img`, `inventory.img`, etc.), a
`dump.log`, and a `manifest.yaml` describing the captured container.

## Restore

```bash
sudo KUBECONFIG=/etc/rancher/k3s/k3s.yaml \
  /usr/local/bin/snapshotctl restore \
    --manifest /tmp/worker.yaml \
    --checkpoint-id reap-prefill-<id> \
    --containers main
```

The agent uses `nsenter` to enter the restore Pod's PID namespace and exec
`nsrestore`, which dispatches CRIU restore + (when present)
`cuda-checkpoint --action restore`. Watch the agent's log for the per-step
timing, and the Pod's annotations for the contract status:

```bash
k3s kubectl -n dynamo-system get pod reap-prefill-snapshot \
    -o jsonpath='{.metadata.annotations.nvidia\.com/snapshot-restore-status}'
```

`completed` is success; `failed` plus an inspectable Pod log narrows the
cause.

## Common failure modes

- **`Failed to connect to containerd … dial /run/containerd/containerd.sock: timeout`**:
  the agent dialed the conventional path but the host runs a non-default
  containerd. The chart since v1.2.1 always mounts the host's
  `runtime.socketPath` directory at `/run/containerd` inside the agent, so
  this only fires when `runtime.socketPath` is wrong for the host. Verify
  with `ls /run/k3s/containerd/containerd.sock` on the node.

- **`tar: /var/lib/.../snapshots/<n>/fs: Cannot open: No such file or directory`**
  during the checkpoint's `overlay_capture` phase: the chart's
  `runtime.storageDir` doesn't match where the runtime keeps overlay
  upperdirs. k3s and RKE2 use `/var/lib/rancher/k3s/agent/containerd`;
  CRI-O uses `/var/lib/containers`; vanilla containerd uses
  `/var/lib/containerd`.

- **`exec: "cuda-checkpoint": executable file not found in $PATH`** when
  the checkpoint Pod starts: the worker image is missing
  `cuda-checkpoint`. Bake it in via the `COPY --from=` snippet in
  pre-flight, or pass `--disable-cuda-checkpoint-job-file` to skip the
  cuda-checkpoint wrap (CRIU-only smoke).

- **`pod ... has no batch.kubernetes.io/job-name label`** in the agent
  log: a workload was labelled with
  `nvidia.com/snapshot-is-checkpoint-source` outside the
  `snapshotctl`-created Job. The agent only ever drives checkpoints
  spawned by a Job (per the upstream contract); apply the source label
  through `snapshotctl checkpoint` rather than `kubectl annotate`.

- **`nsenter: failed to execute /usr/local/bin/nsrestore: No such file or
  directory`** on restore: the workload image is missing `nsrestore`. Bake
  it in via the same `COPY --from=` snippet.

- **`criu binary not found at /usr/local/sbin/criu`** in the agent's
  `nsrestore` log: the workload image is missing CRIU itself. `nsrestore`
  is the orchestrator; it shells out to the real CRIU binary inside the
  worker's PID namespace. Add the `COPY --from=...criu...` lines (and the
  apt-installed runtime libs) per pre-flight.

- **`UnexpectedAdmissionError ... Available: 0` for nvidia.com/gpu**
  immediately after a restore: a previous checkpoint Job's Pod is still
  holding the device allocation in the kubelet device-plugin index.
  Force-delete the prior Pod and Job (`kubectl delete pod ... --force
  --grace-period=0`), wait ~5s, and re-issue the restore.

## End-to-end verification (a4-us-002-rl9, 2026-05-29)

Verified the full blog flow against a single-GPU CUDA tensor smoke pod
(B200 GPU 0, runtime image
`ghcr.io/ai-blaise/optimization-playground-sglang-runtime:reap-nvfp4-dynamo-1.1.1-v9`,
chart `snapshot-1.2.1` from this branch, agent
`nvcr.io/nvidia/ai-dynamo/snapshot-agent:1.1.1`):

| Phase     | Duration | Notes |
| --------- | -------- | ----- |
| Checkpoint `prepare`         | 10 ms   | Lookup container PID + OCI spec via containerd CRI socket |
| Checkpoint `cuda` (lock + dump) | 642 ms | `cuCheckpointProcessLock` + `cuCheckpointProcessCheckpoint` via the agent's `cuda-checkpoint-helper` |
| Checkpoint `criu_dump`       | 2070 ms | CRIU 4.x froze + dumped 26 process threads |
| Checkpoint `overlay_capture` | 5 ms    | `tar` of `/var/lib/rancher/k3s/agent/containerd/.../fs` (depends on `runtime.storageDir` chart fix) |
| **Checkpoint total wall**    | **2.73 s** | From agent detection to `status=completed` |
| Restore `host_inspect`       | 12 ms   | Discover placeholder pod via kubelet PodResources |
| Restore `nsrestore_setup`    | 3 ms    | `nsenter` into placeholder pod's PID namespace |
| Restore `criu_restore`       | 632 ms  | CRIU walked the saved tree back into running processes |
| Restore `cuda`               | 1979 ms | `cuCheckpointProcessRestore` brought HBM back into the GPU |
| **Restore total wall**       | **2.66 s** | From agent detection to placeholder pod 1/1 Running |

Artifact lived at `/checkpoints/cuda-v9-1/versions/1/` on the
`snapshot-pvc` PVC, 1.3 GB total. `pages-8.img` (1.3 GB) holds the CUDA
tensor's HBM dumped into CPU pages — this is the blog's "cuda-checkpoint
dumps all device state into CPU memory" path made concrete. `rootfs-diff.tar`
proves overlay capture worked (depends on `runtime.storageDir`).

State preservation across the round trip:

- Python iteration counter: was `iter=0` at checkpoint time, resumed at
  `iter=1, 2, 3, ..., 9` after restore (CRIU restored the in-process
  `iteration` variable).
- CUDA tensor sum: `8386560.0` (== `sum(range(4096))`) every iteration
  post-restore — HBM contents survived intact.

Both fixtures (Pod manifest + Python script) live under
[`addons/dynamo-snapshot/examples/`](../addons/dynamo-snapshot/examples/);
the values + chart they ran against are pinned by the same PR that
landed this runbook.

[blog]: https://developer.nvidia.com/blog/nvidia-dynamo-snapshot-fast-startup-for-inference-workloads-on-kubernetes/
