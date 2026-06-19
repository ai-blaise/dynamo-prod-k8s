# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import yaml


MANIFEST = (
    Path(__file__).resolve().parents[2]
    / "deploy"
    / "production"
    / "examples"
    / "deepseek-v32-reap-sglang.yaml"
)
OPTRT_R20_MANIFEST = (
    Path(__file__).resolve().parents[2]
    / "deploy"
    / "production"
    / "examples"
    / "deepseek-v32-nextn-optrt-r20.yaml"
)
FRONTEND_ARGS = (
    Path(__file__).resolve().parents[2]
    / "components"
    / "src"
    / "dynamo"
    / "frontend"
    / "frontend_args.py"
)
FRONTEND_MAIN = (
    Path(__file__).resolve().parents[2]
    / "components"
    / "src"
    / "dynamo"
    / "frontend"
    / "main.py"
)
MODEL_CARD = Path(__file__).resolve().parents[2] / "lib" / "llm" / "src" / "model_card.rs"
TARGET_MODEL = (
    "BlaiseAI/DeepSeek-V3.2-REAP-345B-SpinQuant-ActKV-NVFP4-NextN-Graft"
)


def _manifest() -> dict:
    return yaml.safe_load(MANIFEST.read_text())


def _optrt_r20_docs() -> list[dict]:
    return list(yaml.safe_load_all(OPTRT_R20_MANIFEST.read_text()))


def _optrt_r20_configmap() -> dict:
    return next(doc for doc in _optrt_r20_docs() if doc["kind"] == "ConfigMap")


def _optrt_r20_dgd() -> dict:
    return next(doc for doc in _optrt_r20_docs() if doc["kind"] == "DynamoGraphDeployment")


def _optrt_r20_engine_config(name: str) -> dict:
    return yaml.safe_load(_optrt_r20_configmap()["data"][name])


def _optrt_r20_envs(service_name: str) -> dict[str, str]:
    services = _optrt_r20_dgd()["spec"]["services"]
    return {item["name"]: item["value"] for item in services[service_name]["envs"]}


def _optrt_r20_args(service_name: str) -> list[str]:
    services = _optrt_r20_dgd()["spec"]["services"]
    return services[service_name]["extraPodSpec"]["mainContainer"]["args"]


def _service_for(service_name: str) -> dict:
    return _manifest()["spec"]["services"][service_name]


def _args_for(service_name: str) -> list[str]:
    service = _service_for(service_name)
    return service["extraPodSpec"]["mainContainer"]["args"]


def _arg_value(args: list[str], flag: str) -> str:
    return args[args.index(flag) + 1]


def test_fastokens_is_dynamo_frontend_default_tokenizer():
    frontend_args = FRONTEND_ARGS.read_text()
    tokenizer_arg_start = frontend_args.index('flag_name="--tokenizer"')
    tokenizer_arg_end = frontend_args.index('flag_name="--trust-remote-code"')
    tokenizer_arg = frontend_args[tokenizer_arg_start:tokenizer_arg_end]
    frontend_main = FRONTEND_MAIN.read_text()
    model_card = MODEL_CARD.read_text()
    tokenizer_fn = model_card[model_card.index("pub fn tokenizer(&self)") :]
    tokenizer_env_end = tokenizer_fn.index("match &self.tokenizer")
    tokenizer_env = tokenizer_fn[:tokenizer_env_end]

    assert 'env_var="DYN_TOKENIZER"' in tokenizer_arg
    assert 'default="fastokens"' in tokenizer_arg
    assert 'os.environ["DYN_TOKENIZER"] = config.tokenizer_backend' in frontend_main
    assert 'Ok(v) if v == "default" || v.is_empty() => false' in tokenizer_env
    assert "Err(_) => true" in tokenizer_env


def test_deepseek_reap_frontend_uses_event_backed_kv_routing():
    args = _args_for("Frontend")

    assert _arg_value(args, "--router-mode") == "kv"
    assert _arg_value(args, "--dyn-chat-processor") == "dynamo"
    assert _arg_value(args, "--tokenizer") == "fastokens"
    assert "--router-kv-events" in args
    assert "--no-kv-events" not in args
    assert "--no-router-kv-events" not in args


def test_deepseek_reap_workers_keep_selected_setup2_custom_stack_contract():
    assert _service_for("prefill")["replicas"] == 1
    assert _service_for("prefill")["resources"]["limits"]["gpu"] == "4"
    assert _service_for("decode")["replicas"] == 2
    assert _service_for("decode")["resources"]["limits"]["gpu"] == "2"

    for service_name, mode, tp, mem_fraction in (
        ("prefill", "prefill", "4", "0.66"),
        ("decode", "decode", "2", "0.64"),
    ):
        args = _args_for(service_name)

        assert _arg_value(args, "--disaggregation-mode") == mode
        assert _arg_value(args, "--disaggregation-transfer-backend") == "nixl"
        assert _arg_value(args, "--dyn-tool-call-parser") == "deepseek_v3_2"
        assert _arg_value(args, "--dyn-reasoning-parser") == "deepseek_r1"
        assert _arg_value(args, "--served-model-name") == TARGET_MODEL
        assert _arg_value(args, "--model-path") == f"/models/{TARGET_MODEL}"
        assert _arg_value(args, "--tp") == tp
        assert _arg_value(args, "--dp") == "1"
        assert "--enable-dp-attention" not in args
        assert _arg_value(args, "--quantization") == "compressed-tensors"
        assert "--kv-events-config" in args
        assert _arg_value(args, "--kv-cache-dtype") == "bfloat16"
        assert _arg_value(args, "--nsa-prefill-backend") == "tokenspeed_mla"
        assert _arg_value(args, "--nsa-decode-backend") == "tokenspeed_mla"
        assert _arg_value(args, "--mem-fraction-static") == mem_fraction
        assert _arg_value(args, "--max-running-requests") == "32"
        assert _arg_value(args, "--context-length") == "140000"
        assert _arg_value(args, "--max-total-tokens") == "1048576"
        assert _arg_value(args, "--cuda-graph-max-bs") == "48"

        assert "--enable-hierarchical-cache" not in args
        assert "--json-model-override-args" not in args
        assert "--nsa-indexer-mode" not in args
        assert "--enable-turboquant-dense-kv-cache" not in args
        assert "--turboquant-dense-kv-preset" not in args
        assert "--disable-cuda-graph" not in args
        assert "--load-format" not in args
        assert "--remote-instance-weight-loader-backend" not in args
        assert "--modelexpress-config" not in args

    prefill_args = _args_for("prefill")
    decode_args = _args_for("decode")

    assert "--enable-dsa-prefill-context-parallel" in prefill_args
    assert _arg_value(prefill_args, "--attention-context-parallel-size") == "4"
    assert _arg_value(prefill_args, "--dsa-prefill-cp-kv-storage-mode") == "layersplit"

    assert "--enable-hisparse" not in prefill_args
    assert "--disable-radix-cache" not in prefill_args

    assert "--enable-hisparse" in decode_args
    assert "--disable-radix-cache" in decode_args
    assert "--hisparse-config" in decode_args
    assert '"top_k":1024' in _arg_value(decode_args, "--hisparse-config")

    envs = {item["name"]: item["value"] for item in _manifest()["spec"]["envs"]}
    assert envs["SGLANG_ENABLE_WARP_DECODE"] == "1"
    assert envs["SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER"] == "1"
    assert envs["UCX_TLS"] == "cuda_copy,cuda_ipc,tcp"


def test_deepseek_reap_smc_is_decode_only():
    prefill_args = _args_for("prefill")
    decode_args = _args_for("decode")

    assert "--speculative-algorithm" not in prefill_args
    assert _arg_value(decode_args, "--speculative-algorithm") == "SMC"
    assert (
        _arg_value(decode_args, "--speculative-draft-model-path")
        == "/models/smcsd/GLM-4-9B-0414-FP8-DeepSeekV32-OMP"
    )
    assert _arg_value(decode_args, "--speculative-draft-model-quantization") == "fp8"
    assert _arg_value(decode_args, "--speculative-draft-attention-backend") == "triton"
    assert _arg_value(decode_args, "--smc-draft-kv-cache-dtype") == "fp8_e4m3"
    assert _arg_value(decode_args, "--smc-n-particles") == "4"
    assert _arg_value(decode_args, "--smc-gamma") == "6"


def test_optrt_r20_manifest_keeps_optimized_custom_stack_default():
    prefill = _optrt_r20_engine_config("prefill.yaml")
    decode = _optrt_r20_engine_config("decode.yaml")
    prefill_env = _optrt_r20_envs("prefill")
    decode_env = _optrt_r20_envs("decode")
    prefill_args = _optrt_r20_args("prefill")
    decode_args = _optrt_r20_args("decode")

    for cfg in (prefill, decode):
        transceiver = cfg["cache_transceiver_config"]
        assert transceiver["backend"] == "NIXL"
        assert transceiver["transceiver_runtime"] == "PYTHON"
        assert transceiver["max_tokens_in_buffer"] == 131072
        assert cfg["sparse_attention_config"]["mla_latent_kv_dtype"] == "kvarn_k2v2"
        assert cfg["sparse_attention_config"]["mla_latent_kv_amortize"] is True
        assert cfg["sparse_attention_config"]["indexer_k_dtype"] == "fp4"
        assert cfg["moe_config"]["backend"] == "WARPDECODE"

    for envs in (prefill_env, decode_env):
        # This is the NIXL plugin backend. The direct UCX transceiver remains
        # banned by cache_transceiver_config.backend=NIXL above.
        assert envs["TRTLLM_NIXL_KVCACHE_BACKEND"] == "UCX"
        assert envs["TRTLLM_NIXL_ENABLE_COALESCE"] == "1"
        assert envs["TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP"] == "0"
        assert envs["TRTLLM_ENABLE_KVCACHE_RECEIVE_PARALLEL"] == "1"
        assert "TRTLLM_USE_UCX_KVCACHE" not in envs
        assert "TRTLLM_USE_MOONCAKE_KVCACHE" not in envs
        assert "TRTLLM_USE_MPI_KVCACHE" not in envs

    for args in (prefill_args, decode_args):
        # TRT-LLM's Dynamo runtime connector supports only "none" or "kvbm".
        # NIXL is selected by cache_transceiver_config, so --connector must not
        # be rewritten to an unsupported "--connector nixl" value.
        assert _arg_value(args, "--connector") == "none"

    prefill_sparse = prefill["sparse_attention_config"]
    assert prefill["context_parallel_size"] == 2
    assert prefill["cp_config"]["cp_type"] == "LAYERSPLIT"
    assert prefill_sparse["layersplit_enabled"] is True
    assert prefill_sparse["layersplit_transfer_backend"] == "nixl"
    assert prefill_sparse["layersplit_all_cp_ranks_transfer"] is True
    assert prefill_sparse["layersplit_owner_local_alloc"] is True
    assert prefill_env["TRTLLM_FORCE_COMM_METHOD"] == "NVLINK_TWO_SIDED"

    assert decode["context_parallel_size"] == 1
    assert decode["enable_attention_dp"] is True
    assert decode["sparse_attention_config"]["layersplit_enabled"] is False
    assert decode["moe_config"]["use_low_precision_moe_combine"] is False
    assert decode["moe_config"]["warp_decode"]["enabled"] is True
    assert decode["moe_config"]["warp_decode"]["policy"] == "force"
    assert decode["moe_config"]["warp_decode"]["allow_parallelism_fallback"] is False
    assert decode_env["TRTLLM_FORCE_COMM_METHOD"] == "DEEPEPLOWLATENCY"
    assert decode_env["TRTLLM_DEEP_EP_TOKEN_LIMIT"] == "64"
    assert decode_env["TRTLLM_DEEP_EP_DISABLE_P2P_FOR_LOW_LATENCY_MODE"] == "0"
    assert decode_env["TRTLLM_MOE_POST_QUANT_ALLTOALLV"] == "1"

    smc = decode["speculative_config"]
    assert smc["decoding_type"] == "SMC"
    assert smc["speculative_model"] == "/models/BlaiseAI/GLM-4-9B-0414-FP8-DeepSeekV32-OMP"
    assert smc["gamma"] == 6
    assert smc["n_particles"] == 4
    assert smc["draft_attention_backend"] == "triton"
    assert smc["draft_kv_cache_dtype"] == "bfloat16"
