// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_kv_router::config::RouterConfigOverride;

use crate::protocols::common::preprocessor::{BootstrapInfo, PrefillResult};

/// Errors that can occur during prefill routing
#[derive(Debug, thiserror::Error)]
pub enum PrefillError {
    /// Prefill router has not been activated yet
    #[error("Prefill router not yet activated")]
    NotActivated,

    /// TODO: Separate prefill worker error from prefill router error
    /// Error during prefill execution
    #[error("Prefill execution failed: {0}")]
    PrefillError(
        String,
        #[source] Option<Box<dyn std::error::Error + Send + Sync + 'static>>,
    ),

    /// Disaggregated params not found in prefill response
    #[error("No disaggregated params in prefill response: {0}")]
    NoDisaggregatedParams(String),
}

/// Result of the prefill phase in `generate()`.
pub(super) enum PrefillOutcome {
    /// Bootstrap optimization: prefill spawned in background, bootstrap info ready
    Bootstrap(BootstrapInfo),
    /// Synchronous prefill completed with result. `worker_link` carries the
    /// prefill worker's `engine.generate` span pointer for the decode side
    /// to render as an OTel `Link` via `PreprocessedRequest.migration_link`.
    Completed {
        result: PrefillResult,
        worker_info: Option<(u64, Option<u32>)>,
        worker_link: Option<crate::protocols::common::preprocessor::TraceLink>,
    },
}

pub(super) struct CompletedPrefillPinMetadata {
    pub prefill_worker_id: u64,
    pub prefill_dp_rank: u32,
    pub ctx_info_endpoint: String,
    pub disagg_request_id: String,
}

pub(super) fn completed_prefill_pin_metadata(
    result: &PrefillResult,
    worker_info: Option<(u64, Option<u32>)>,
    request_id: &str,
) -> anyhow::Result<CompletedPrefillPinMetadata> {
    let Some((prefill_worker_id, prefill_dp_rank)) = worker_info else {
        return Err(anyhow::anyhow!(
            "Completed prefill request {} did not return worker_id/prefill_dp_rank metadata; refusing unpinned decode handoff",
            request_id
        ));
    };
    let Some(prefill_dp_rank) = prefill_dp_rank else {
        return Err(anyhow::anyhow!(
            "Completed prefill request {} did not return prefill_dp_rank metadata; refusing unpinned decode handoff",
            request_id
        ));
    };
    let Some(ctx_info_endpoint) = result
        .disaggregated_params
        .get("ctx_info_endpoint")
        .and_then(|v| v.as_str())
        .filter(|value| !value.is_empty())
    else {
        return Err(anyhow::anyhow!(
            "Completed prefill request {} did not return ctx_info_endpoint metadata; refusing unpinned decode handoff",
            request_id
        ));
    };
    let disagg_request_id = result
        .disaggregated_params
        .get("disagg_request_id")
        .and_then(|v| v.as_str())
        .unwrap_or(request_id);

    Ok(CompletedPrefillPinMetadata {
        prefill_worker_id,
        prefill_dp_rank,
        ctx_info_endpoint: ctx_info_endpoint.to_string(),
        disagg_request_id: disagg_request_id.to_string(),
    })
}

pub(super) enum PrefillResolveDecision {
    Resolved {
        worker_id: u64,
        dp_rank: Option<u32>,
        bootstrap_info: BootstrapInfo,
    },
    Unavailable,
    NotActivated,
    NoBootstrapEndpoint,
}

pub(super) fn build_decode_router_override(
    existing_override: Option<RouterConfigOverride>,
) -> RouterConfigOverride {
    RouterConfigOverride {
        overlap_score_credit: Some(0.0),
        assume_kv_reuse: Some(false),
        track_prefill_tokens: Some(false),
        ..existing_override.unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn prefill_result(params: serde_json::Value) -> PrefillResult {
        PrefillResult {
            disaggregated_params: params,
            prompt_tokens_details: None,
        }
    }

    #[test]
    fn completed_prefill_pin_metadata_accepts_pinned_handoff() {
        let result = prefill_result(json!({
            "ctx_info_endpoint": "nixl://ctx/0",
            "disagg_request_id": "disagg-123"
        }));

        let metadata =
            completed_prefill_pin_metadata(&result, Some((42, Some(3))), "request-123")
                .expect("metadata should be valid");

        assert_eq!(metadata.prefill_worker_id, 42);
        assert_eq!(metadata.prefill_dp_rank, 3);
        assert_eq!(metadata.ctx_info_endpoint, "nixl://ctx/0");
        assert_eq!(metadata.disagg_request_id, "disagg-123");
    }

    #[test]
    fn completed_prefill_pin_metadata_defaults_disagg_request_id_to_request_id() {
        let result = prefill_result(json!({"ctx_info_endpoint": "nixl://ctx/0"}));

        let metadata =
            completed_prefill_pin_metadata(&result, Some((42, Some(3))), "request-123")
                .expect("metadata should be valid");

        assert_eq!(metadata.disagg_request_id, "request-123");
    }

    #[test]
    fn completed_prefill_pin_metadata_requires_worker_info() {
        let result = prefill_result(json!({"ctx_info_endpoint": "nixl://ctx/0"}));

        let err = completed_prefill_pin_metadata(&result, None, "request-123")
            .expect_err("missing worker info must fail closed");

        assert!(err.to_string().contains("worker_id/prefill_dp_rank"));
    }

    #[test]
    fn completed_prefill_pin_metadata_requires_dp_rank() {
        let result = prefill_result(json!({"ctx_info_endpoint": "nixl://ctx/0"}));

        let err = completed_prefill_pin_metadata(&result, Some((42, None)), "request-123")
            .expect_err("missing dp rank must fail closed");

        assert!(err.to_string().contains("prefill_dp_rank"));
    }

    #[test]
    fn completed_prefill_pin_metadata_requires_ctx_info_endpoint() {
        let result = prefill_result(json!({"ctx_info_endpoint": ""}));

        let err = completed_prefill_pin_metadata(&result, Some((42, Some(3))), "request-123")
            .expect_err("missing ctx_info_endpoint must fail closed");

        assert!(err.to_string().contains("ctx_info_endpoint"));
    }
}
