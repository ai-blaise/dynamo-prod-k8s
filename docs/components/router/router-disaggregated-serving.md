---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Disaggregated Serving
subtitle: Prefill and decode routing with the Dynamo router
---

Dynamo supports disaggregated serving where prefill (prompt processing) and decode (token generation) are handled by separate worker pools. When you register workers with `ModelType.Prefill`, the frontend automatically detects them and activates an internal prefill router.

For the high-level deployment matrix, see [Router Guide](router-guide.md). For the router flags used in this setup, see [Configuration and Tuning](router-configuration.md).

## Automatic Prefill Router Activation

The prefill router is automatically created when:
1. A decode model is registered, for example via `register_model()` with `ModelType.Chat | ModelType.Completions`.
2. A prefill worker is detected with the same model name and `ModelType.Prefill`.

Key characteristics of the prefill router:
- **Always disables active block tracking** (`track_active_blocks=false`) since prefill workers do not perform decode.
- **Seamlessly integrates** into the request pipeline between preprocessing and decode routing.
- **Falls back gracefully** to decode-only mode if prefill fails or no prefill workers are available.

Key characteristics of the decode routing stage in disaggregated mode:
- **Disables overlap scoring** (`overlap_score_credit=0`) because decode routing should not chase prefix reuse.
- **Disables KV reuse assumption** (`assume_kv_reuse=false`) unless the backend can truly deduplicate transferred blocks.
- **Disables prefill-token tracking** (`track_prefill_tokens=false`) so decode-side load reflects decode work rather than already-completed prompt work.

## Setup Example

When both workers are registered, requests are automatically routed.

```python
# Decode worker registration (in your decode worker)
decode_endpoint = runtime.endpoint("dynamo.decode.generate")

await register_model(
    model_input=ModelInput.Tokens,
    model_type=ModelType.Chat | ModelType.Completions,
    endpoint=decode_endpoint,
    model_name="meta-llama/Llama-2-7b-hf",
    # ... other parameters
)

await decode_endpoint.serve_endpoint(decode_handler.generate)

# Prefill worker registration (in your prefill worker)
prefill_endpoint = runtime.endpoint("dynamo.prefill.generate")

await register_model(
    model_input=ModelInput.Tokens,
    model_type=ModelType.Prefill,
    endpoint=prefill_endpoint,
    model_name="meta-llama/Llama-2-7b-hf",
    # ... other parameters
)

await prefill_endpoint.serve_endpoint(prefill_handler.generate)
```

>[!Note]
> The automatic disaggregated routing setup described here is currently supported by the integrated `dynamo.frontend` path. It is not provided as a single turnkey mode by the standalone Python router (`python -m dynamo.router`). If you build this topology with standalone routers, you must launch and connect the prefill and decode routing stages yourself and handle request handoff, including the `disaggregated_params` returned by prefill. For an advanced reference, see the [Global Router](https://github.com/ai-dynamo/dynamo/tree/main/components/src/dynamo/global_router), which composes local prefill and decode router pools explicitly.

## Request Flow

The following diagram shows an overview of the major components in disaggregated serving:

```mermaid
graph TD
    HTTP[HTTP]
    ROUTER[Router]
    PREFILL[Prefill Worker]
    DECODE[Decode Worker]

    classDef worker_style fill:#f3e5f5,stroke:#333,stroke-width:2px,color:#333;
    classDef router_style fill:#2e8b57,stroke:#333,stroke-width:2px,color:#fff;

    class PREFILL,DECODE worker_style
    class ROUTER router_style

    HTTP <--> |"request/response"| ROUTER
    ROUTER --> |"1. send to prefill"| PREFILL
    PREFILL --> |"2. return NIXL metadata"| ROUTER
    ROUTER --> |"3. send with metadata"| DECODE
    DECODE --> |"4. stream response"| ROUTER

    PREFILL -.-> |"publish kv events"| ROUTER

    linkStyle 0,1,2,3,4 stroke:#8b4513,stroke-width:2px
    linkStyle 5 stroke:#2196f3,stroke-width:2px
```

## Request Pinning Proof Markers

In disaggregated deployments that require deterministic request-level prefill
to decode handoff, the frontend/router logs authoritative pin lifecycle markers.
The markers are used by production smoke tests to distinguish a real pinned KV
handoff from ordinary worker selection.

For the bootstrap path, the router emits:

```text
dynamo disagg request pin established request_id=<id> prefill_worker_id=<worker> prefill_dp_rank=Some(<rank>) bootstrap_host=<host> bootstrap_port=<port>
dynamo disagg request pin outbound to decode request_id=<id> bootstrap_host=<host> bootstrap_port=<port>
```

For the synchronous completed-prefill path, the router must still prove the same
request affinity before decode. It emits the same marker names with
`handoff_mode=completed_prefill` and a decode-consumable `ctx_info_endpoint`:

```text
dynamo disagg request pin established request_id=<id> disagg_request_id=<ctx-id> prefill_worker_id=<worker> prefill_dp_rank=<rank> ctx_info_endpoint=<endpoint> handoff_mode=completed_prefill
dynamo disagg request pin outbound to decode request_id=<id> disagg_request_id=<ctx-id> ctx_info_endpoint=<endpoint> handoff_mode=completed_prefill
```

Completed-prefill handoff is fail-closed for the metadata required by those
markers. If prefill does not return worker id, prefill DP rank, or a non-empty
`ctx_info_endpoint`, the router refuses to route decode rather than allowing an
unpinned KV receive. The decode request still receives the original
`PrefillResult.disaggregated_params`; the markers are proof and audit metadata,
not a replacement for the engine-owned transfer payload.

Production gates should require all of the following for a pinned disaggregated
request:

- prefill and decode `dynamo request pin route selected` markers for the same
  request id;
- one `dynamo disagg request pin established` marker;
- one `dynamo disagg request pin outbound to decode` marker;
