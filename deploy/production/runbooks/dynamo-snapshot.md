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

3. The worker container image you intend to snapshot ships both
   `cuda-checkpoint` and `nsrestore` on `$PATH`. The blog-shipped agent
   image (`nvcr.io/nvidia/ai-dynamo/snapshot-agent`) has them under
   `/usr/local/bin/`; copy them into your worker image, e.g.

   ```dockerfile
   COPY --from=nvcr.io/nvidia/ai-dynamo/snapshot-agent:1.1.1 \
        /usr/local/bin/cuda-checkpoint /usr/local/bin/cuda-checkpoint
   COPY --from=nvcr.io/nvidia/ai-dynamo/snapshot-agent:1.1.1 \
        /usr/local/bin/nsrestore /usr/local/bin/nsrestore
   ```

   If the worker image lacks `cuda-checkpoint`, pass
   `--disable-cuda-checkpoint-job-file` to `snapshotctl checkpoint` to fall
   back to CRIU-only (CPU-only) capture for smoke testing.

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

# Run the checkpoint.
sudo KUBECONFIG=/etc/rancher/k3s/k3s.yaml \
  /usr/local/bin/snapshotctl checkpoint \
    --manifest /tmp/worker.yaml \
    --container main \
    --checkpoint-id reap-prefill-$(date +%s) \
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

[blog]: https://developer.nvidia.com/blog/nvidia-dynamo-snapshot-fast-startup-for-inference-workloads-on-kubernetes/
