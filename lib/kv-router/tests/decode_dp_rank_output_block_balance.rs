// SPDX-FileCopyrightText: Copyright (c) 2026 BlaiseAI / ai-blaise. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// VALIDATION (kv-aware-routing dimension): proves that the Dynamo decode-worker
// dp_rank load signal (`decode_blocks`, the dominant term in the decode logit
// `prefill_load_scale*adjusted_prefill_blocks + decode_blocks`) is STALE during
// sustained decode unless `--router-track-output-blocks` is set, and that
// enabling it lets the load-only decode selector pick the genuinely
// least-loaded dp_rank within a single 4-DP decode worker.
//
// This is the exact production scenario: 1 decode worker, tp4/ep4 attention-DP
// (4 dp_ranks), `overlap_score_credit=0` decode override (set by the prefill
// router for every decode request), `router_temperature=0` (default), so the
// selector is argmin over per-(worker,dp_rank) `potential_decode_blocks`.
//
// Run (CPU only, no GPU, no model load):
//   cargo test -p dynamo-kv-router --test decode_dp_rank_output_block_balance

use std::collections::HashMap;
use std::time::Instant;

use rustc_hash::FxHashMap;

use dynamo_kv_router::config::KvRouterConfig;
use dynamo_kv_router::protocols::{
    PrefillLoadHint, RoutingConstraints, SharedCacheHits, WorkerWithDpRank,
};
use dynamo_kv_router::scheduling::TierOverlapBlocks;
use dynamo_kv_router::selector::{DefaultWorkerSelector, WorkerSelector};
use dynamo_kv_router::test_utils::{NoopSequencePublisher, SimpleWorkerConfig};
use dynamo_kv_router::{ActiveSequencesMultiWorker, SchedulingRequest, SequenceRequest};

const DECODE_WORKER_ID: u64 = 7;
const DP_SIZE: u32 = 4;
const BLOCK_SIZE: usize = 64;

fn make_decode_worker() -> ActiveSequencesMultiWorker<NoopSequencePublisher> {
    // One decode worker (id=7) advertising dp_range [0, 4): exactly the live
    // tp4/ep4 attention-DP decode topology.
    ActiveSequencesMultiWorker::new(
        NoopSequencePublisher,
        BLOCK_SIZE,
        HashMap::from([(DECODE_WORKER_ID, (0_u32, DP_SIZE))]),
        false,
        0,
        "decode",
    )
}

fn rank(dp: u32) -> WorkerWithDpRank {
    WorkerWithDpRank::new(DECODE_WORKER_ID, dp)
}

/// Admit a request directly onto a chosen dp_rank (mirrors the Dynamo decode
/// path: the router pins `routing.dp_rank` → op-trt strict `attention_dp_rank`).
/// Decode admits set `track_prefill_tokens=false` (the decode override).
fn admit(
    seqs: &ActiveSequencesMultiWorker<NoopSequencePublisher>,
    request_id: &str,
    dp: u32,
    isl_tokens: usize,
    now: Instant,
) {
    // token_sequence None => no prefix-overlap contribution; decode load is
    // driven purely by the active/output block accounting under test.
    seqs.add_request(
        SequenceRequest {
            request_id: request_id.to_string(),
            token_sequence: None,
            track_prefill_tokens: false,
            expected_output_tokens: None,
            prefill_load_hint: Some(PrefillLoadHint {
                initial_effective_prefill_tokens: isl_tokens,
                expected_prefill_duration: None,
            }),
            worker: rank(dp),
            lora_name: None,
        },
        now,
    )
    .expect("admit");
    // Decode requests arrive post-prefill: prefill is already complete.
    seqs.mark_prefill_completed(&request_id.to_string(), now)
        .expect("mark_prefill_completed");
}

/// Build the decode-phase scheduling request the selector actually sees:
/// overlap_score_credit=0 + temperature=0, decode_blocks = measured per-rank
/// potential load. We pin nothing (unpinned) so the selector iterates all 4
/// dp_ranks and returns its argmin — i.e. the rank the router would assign.
fn select_decode_rank(
    seqs: &ActiveSequencesMultiWorker<NoopSequencePublisher>,
    new_isl_tokens: usize,
) -> u32 {
    let (decode_blocks, _prefill_tokens) =
        seqs.potential_blocks_and_tokens(None, &Default::default());

    // decode override: overlap_score_credit=0, temperature=0 (deterministic argmin).
    let config = KvRouterConfig {
        overlap_score_credit: 0.0,
        prefill_load_scale: 1.0,
        router_temperature: 0.0,
        ..Default::default()
    };
    let selector = DefaultWorkerSelector::new(Some(config), "decode");

    // One worker, 4 dp_ranks.
    let workers = HashMap::from([(DECODE_WORKER_ID, SimpleWorkerConfig::default())]);

    // Feed the measured per-rank decode load (the only differentiating signal).
    let mut decode_blocks_fx: FxHashMap<WorkerWithDpRank, usize> = FxHashMap::default();
    for (w, blocks) in &decode_blocks {
        decode_blocks_fx.insert(*w, *blocks);
    }

    let request = SchedulingRequest {
        maybe_request_id: Some("new-decode".to_string()),
        token_seq: None,
        isl_tokens: new_isl_tokens,
        lora_name: None,
        expected_output_tokens: None,
        pinned_worker: None,
        allowed_worker_ids: None,
        routing_constraints: RoutingConstraints::default(),
        router_config_override: None,
        track_prefill_tokens: false,
        priority_jump: 0.0,
        tier_overlap_blocks: TierOverlapBlocks::default(),
        effective_overlap_blocks: HashMap::default(),
        effective_cached_tokens: HashMap::default(),
        shared_cache_hits: None::<SharedCacheHits>,
        decode_blocks: decode_blocks_fx,
        prefill_tokens: FxHashMap::default(),
        update_states: false,
        resp_tx: None,
    };

    let result = selector
        .select_worker(&workers, &request, request.eligibility(), BLOCK_SIZE as u32)
        .expect("select_worker");
    assert_eq!(result.worker.worker_id, DECODE_WORKER_ID);
    result.worker.dp_rank
}

/// Grow a request's output by `n` blocks (simulates `router_track_output_blocks`
/// firing once per generated KV block during decode).
fn grow_output(
    seqs: &ActiveSequencesMultiWorker<NoopSequencePublisher>,
    request_id: &str,
    n_blocks: usize,
) {
    for _ in 0..n_blocks {
        // decay_fraction None => full-weight block (worker sets no
        // expected_output_tokens, so this is the production decay behavior).
        seqs.add_output_block(&request_id.to_string(), None)
            .expect("add_output_block");
    }
}

#[test]
fn without_output_tracking_decode_load_is_stale_across_dp_ranks() {
    let now = Instant::now();
    let seqs = make_decode_worker();

    // Admit one request per dp_rank with IDENTICAL short ISL (1 block each).
    // This is the cold-start fan-out: every rank looks equal at admit.
    for dp in 0..DP_SIZE {
        admit(&seqs, &format!("r{dp}"), dp, BLOCK_SIZE, now);
    }

    // Now ranks 0 and 1 generate LONG outputs (e.g. reasoning traces),
    // ranks 2 and 3 finish almost immediately. WITHOUT output-block tracking
    // (router_track_output_blocks=false) we do NOT call add_output_block, so
    // the router's per-rank decode_blocks stays frozen at the admit value.
    // (No grow_output calls — this models the disabled-tracking world.)

    let blocks = seqs.active_blocks();
    let b0 = blocks.get(&rank(0)).copied().unwrap_or(0);
    let b3 = blocks.get(&rank(3)).copied().unwrap_or(0);

    // Stale: rank 0 (long generation) looks identical to rank 3 (idle-ish).
    assert_eq!(
        b0, b3,
        "without output tracking, a long-generating rank and a short one report the same load \
         (stale signal): b0={b0}, b3={b3}"
    );

    // Consequence: a burst of new decode requests all see equal load → the
    // selector cannot distinguish ranks and spreads by tie-break, even though
    // ranks 0/1 are actually carrying far more decode KV than 2/3. The router
    // pins each pick (relax=false), so op-trt cannot correct the resulting skew.
}

#[test]
fn with_output_tracking_decode_load_reflects_generation_and_steers_rank() {
    let now = Instant::now();
    let seqs = make_decode_worker();

    // Same cold-start fan-out: one request per dp_rank, 1 block each at admit.
    for dp in 0..DP_SIZE {
        admit(&seqs, &format!("r{dp}"), dp, BLOCK_SIZE, now);
    }

    // WITH output-block tracking: ranks 0 and 1 generate long outputs,
    // ranks 2 and 3 generate almost nothing. add_output_block fires per block.
    grow_output(&seqs, "r0", 30); // ~30 blocks of decode KV (long trace)
    grow_output(&seqs, "r1", 24);
    grow_output(&seqs, "r2", 1);
    // r3: no growth (just the admit block)

    let blocks = seqs.active_blocks();
    let b0 = blocks.get(&rank(0)).copied().unwrap_or(0);
    let b1 = blocks.get(&rank(1)).copied().unwrap_or(0);
    let b2 = blocks.get(&rank(2)).copied().unwrap_or(0);
    let b3 = blocks.get(&rank(3)).copied().unwrap_or(0);

    // The per-rank decode load now reflects true generation pressure.
    assert!(
        b0 > b2 && b1 > b2 && b2 > b3,
        "with output tracking, decode load tracks generation: b0={b0} b1={b1} b2={b2} b3={b3}"
    );

    // The load-only decode selector now routes a NEW decode request to the
    // genuinely least-loaded rank (rank 3), keeping the 4 ranks balanced so
    // their per-step batch sizes converge and CUDA graphs fire uniformly.
    let chosen = select_decode_rank(&seqs, BLOCK_SIZE);
    assert_eq!(
        chosen, 3,
        "selector must steer the new request to the least-loaded rank (3); chose {chosen} \
         (loads: 0={b0} 1={b1} 2={b2} 3={b3})"
    );
}

#[test]
fn stale_vs_tracked_diverge_on_the_routing_decision() {
    // Side-by-side: identical generation history, the ONLY difference is whether
    // output-block tracking ran. Show the routing decision diverges.
    let now = Instant::now();

    // ---- World A: tracking OFF (production default) ----
    let off = make_decode_worker();
    for dp in 0..DP_SIZE {
        admit(&off, &format!("r{dp}"), dp, BLOCK_SIZE, now);
    }
    // rank 3 was the one that happened to take a huge generation, but with
    // tracking OFF the router never learns it.
    // (no grow_output)
    let off_blocks = off.active_blocks();
    let off_min_load = (0..DP_SIZE)
        .map(|dp| off_blocks.get(&rank(dp)).copied().unwrap_or(0))
        .min()
        .unwrap();
    let off_max_load = (0..DP_SIZE)
        .map(|dp| off_blocks.get(&rank(dp)).copied().unwrap_or(0))
        .max()
        .unwrap();

    // ---- World B: tracking ON ----
    let on = make_decode_worker();
    for dp in 0..DP_SIZE {
        admit(&on, &format!("r{dp}"), dp, BLOCK_SIZE, now);
    }
    grow_output(&on, "r3", 40); // rank 3 is heavily loaded in reality
    let on_chosen = select_decode_rank(&on, BLOCK_SIZE);

    // Tracking OFF: all ranks tie (max==min), so the router may well pick the
    // already-overloaded rank 3 (tie-break is uniform-random over all 4).
    assert_eq!(
        off_min_load, off_max_load,
        "tracking OFF: router sees a flat, stale load profile (min={off_min_load} max={off_max_load}) \
         and can route onto the truly-overloaded rank"
    );

    // Tracking ON: the router avoids the overloaded rank 3.
    assert_ne!(
        on_chosen, 3,
        "tracking ON: router must NOT route onto the heavily-loaded rank 3 (chose {on_chosen})"
    );
}
