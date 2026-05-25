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
    "BlaiseAI/DeepSeek-V3.2-REAP-345B-SpinQuant-ActKV-NVFP4"
)


def _manifest() -> dict:
    return yaml.safe_load(MANIFEST.read_text())


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
