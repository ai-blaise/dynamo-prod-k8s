<!--
SPDX-FileCopyrightText: Copyright (c) 2026 BlaiseAI / ai-blaise. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# ModelExpress

This add-on deploys `ai-dynamo/modelexpress` for the REAP production lane. It
is wired through Dynamo's native operator integration: the Dynamo Platform chart
sets `dynamo-operator.modelExpressURL`, and the operator injects
`MODEL_EXPRESS_URL` into generated worker pods.

The default REAP manifest still uses absolute local model paths, so merely
installing this add-on does not change SGLang loading behavior. The
infrastructure renderer must opt in to ModelExpress by rendering the worker
`--model-path` as the Hugging Face repo id instead of `/models/...`.

## Responsibilities

- Run the ModelExpress gRPC service on port `8001`.
- Use the Kubernetes metadata backend with the upstream `ModelMetadata` and
  `ModelCacheEntry` CRDs.
- Share the host `/models` cache with REAP workers for the first integration
  mode.
- Expose the server at
  `http://modelexpress.modelexpress.svc.cluster.local:8001`.

## Exclusions

- It does not make SGLang P2P loading the production default.
- It does not remove the local-path deployment fallback.
- It does not own the `DynamoGraphDeployment`; that stays under
  `deploy/production/examples`.

## Required Secrets

The upstream chart image defaults to
`nvcr.io/nvidia/ai-dynamo/modelexpress-server:0.3.0`, so the `modelexpress`
namespace needs an `nvcr-secret` image pull secret unless the image is
overridden to a mirrored registry image.

For private Hugging Face downloads, create `hf-token-secret` in the
`modelexpress` namespace. The A4 infrastructure wrapper creates it when
`HF_TOKEN` is set.

## Verification

```bash
kubectl get crd modelmetadatas.modelexpress.nvidia.com modelcacheentries.modelexpress.nvidia.com
kubectl get pods,svc -n modelexpress
kubectl logs -n modelexpress deploy/modelexpress --tail=200
kubectl get modelcacheentries.modelexpress.nvidia.com -n modelexpress
```

When `REAP_ENABLE_MODELEXPRESS=1` is used in the infrastructure wrapper, REAP
workers should log a ModelExpress connection attempt before SGLang starts.
