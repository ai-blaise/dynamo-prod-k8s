<!--
SPDX-FileCopyrightText: Copyright (c) 2026 BlaiseAI / ai-blaise. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# dynamo-snapshot addon

The upstream NVIDIA Dynamo Snapshot agent + PVC (the feature in
[NVIDIA's Dynamo Snapshot blog post][blog]) deployed as a per-namespace
DaemonSet alongside the workload it snapshots.

| File           | Purpose                                                                                          |
| -------------- | ------------------------------------------------------------------------------------------------ |
| `values.yaml`  | Production overrides for the a4-us-002-rl9 k3s cluster (k3s socket + storageDir, local-path PVC) |
| `README.md`    | This file                                                                                        |

The chart itself lives at `deploy/helm/charts/snapshot/` in this repo and
ships:

- `snapshot-agent` DaemonSet (`nvcr.io/nvidia/ai-dynamo/snapshot-agent`)
- A PVC for checkpoint storage
- A ConfigMap with the agent's CRIU + restore options
- A seccomp profile ConfigMap + initContainer that drops it under
  `/var/lib/kubelet/seccomp/profiles/block-iouring.json` (CRIU cannot
  checkpoint `io_uring`)
- A namespace-scoped Role/RoleBinding (or ClusterRole/ClusterRoleBinding
  for cluster-wide agents)

## Relationship to the criu-snapshots addon

This addon and `addons/criu-snapshots` deploy **two complementary snapshot
paths**. Both are wanted in production:

| Axis                    | `criu-snapshots` (this org's CRD-driven operator)                                     | `dynamo-snapshot` (upstream Dynamo)                                                 |
| ----------------------- | ------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| Subject                 | A live DGD worker, snapshotted in place                                                | A fresh worker the tool launches as a Job, snapshotted, torn down                    |
| Trigger                 | `kubectl apply -f dgds.yaml` (operator-driven, declarative)                            | `snapshotctl checkpoint --manifest worker.yaml` (CLI) or DynamoCheckpoint CR         |
| Where the checkpoint lives | OCI artifact in GHCR via ORAS                                                       | A PVC the agent and the workload share                                              |
| Per-rank semantics      | First-class (`status.perRank`)                                                          | One Job = one container = one checkpoint; rank fanout is the caller's job            |
| Fingerprint guard       | Baked into the OCI artifact, enforced on restore by the snapshot-pull initContainer    | Out of scope at this layer; restore onto a drifted host is the caller's responsibility |
| cuda-checkpoint placement | Host-side (`/opt/criu-snapshots/bin/cuda-checkpoint`, mounted from DaemonSet)         | Inside the worker container (the agent wraps the command with `cuda-checkpoint --launch-job`) |

The full coexistence contract lives in
[`ai-blaise/criu-snapshots/docs/upstream-dynamo-snapshot.md`][coexistence].

## Per-namespace install

`values.yaml` here is suitable for a single workload namespace. Install
into every namespace whose DynamoGraphDeployments you want to be able to
snapshot through the upstream flow. Example for `dynamo-system`:

```sh
helm install dynamo-snapshot deploy/helm/charts/snapshot \
  --namespace dynamo-system \
  --values deploy/production/addons/dynamo-snapshot/values.yaml \
  --kubeconfig /etc/rancher/k3s/k3s.yaml
```

The Argo CD `Application` at `deploy/production/gitops/apps/56-dynamo-snapshot.yaml`
manages this for the canonical `dynamo-system` install.

## Runbook

See [`deploy/production/runbooks/dynamo-snapshot.md`](../../runbooks/dynamo-snapshot.md)
for the verified checkpoint + restore commands against a live REAP DGD on
the a4 fleet.

[blog]: https://developer.nvidia.com/blog/nvidia-dynamo-snapshot-fast-startup-for-inference-workloads-on-kubernetes/
[coexistence]: https://github.com/ai-blaise/criu-snapshots/blob/main/docs/upstream-dynamo-snapshot.md
