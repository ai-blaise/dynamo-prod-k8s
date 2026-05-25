# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
PRODUCTION = ROOT / "deploy" / "production"
MODELEXPRESS_VALUES = PRODUCTION / "addons" / "modelexpress" / "values.yaml"
MODELEXPRESS_APP = PRODUCTION / "gitops" / "apps" / "56-modelexpress.yaml"
GITOPS_KUSTOMIZATION = PRODUCTION / "gitops" / "kustomization.yaml"
GITOPS_PROJECT = PRODUCTION / "gitops" / "project.yaml"
DYNAMO_PLATFORM_VALUES = PRODUCTION / "addons" / "dynamo-platform" / "values.yaml"
SGLANG_ARGS = ROOT / "components" / "src" / "dynamo" / "sglang" / "args.py"

MODELEXPRESS_URL = "http://modelexpress.modelexpress.svc.cluster.local:8001"
MODELEXPRESS_REVISION = "9fd703ede7eb0a04b538c6edcb973507a523fd5c"


def test_modelexpress_is_a_baseline_gitops_addon():
    kustomization = yaml.safe_load(GITOPS_KUSTOMIZATION.read_text())
    project = yaml.safe_load(GITOPS_PROJECT.read_text())
    app = yaml.safe_load(MODELEXPRESS_APP.read_text())

    assert "apps/56-modelexpress.yaml" in kustomization["resources"]
    assert "https://github.com/ai-dynamo/modelexpress.git" in project["spec"]["sourceRepos"]
    assert app["metadata"]["name"] == "modelexpress"
    assert app["metadata"]["annotations"]["argocd.argoproj.io/sync-wave"] == "56"
    assert app["spec"]["destination"]["namespace"] == "modelexpress"

    crd_source, helm_source, values_source = app["spec"]["sources"]
    assert crd_source["repoURL"] == "https://github.com/ai-dynamo/modelexpress.git"
    assert crd_source["targetRevision"] == MODELEXPRESS_REVISION
    assert crd_source["path"] == "examples"
    assert crd_source["directory"]["include"] == "crds.yaml"
    assert helm_source["targetRevision"] == MODELEXPRESS_REVISION
    assert helm_source["path"] == "helm"
    assert helm_source["helm"]["releaseName"] == "modelexpress"
    assert (
        "$values/deploy/production/addons/modelexpress/values.yaml"
        in helm_source["helm"]["valueFiles"]
    )
    assert values_source["ref"] == "values"


def test_modelexpress_values_use_kubernetes_backend_and_shared_models_cache():
    values = yaml.safe_load(MODELEXPRESS_VALUES.read_text())

    assert values["image"]["repository"] == "nvcr.io/nvidia/ai-dynamo/modelexpress-server"
    assert values["image"]["tag"] == "0.3.0"
    assert values["imagePullSecrets"] == [{"name": "nvcr-secret"}]
    assert values["service"]["port"] == 8001
    assert values["serviceAccount"]["automount"] is True
    assert values["serviceAccount"]["rbac"]["enabled"] is True
    assert values["persistence"]["enabled"] is False
    assert values["env"]["MX_METADATA_BACKEND"] == "kubernetes"
    assert values["env"]["MODEL_EXPRESS_CACHE_DIRECTORY"] == "/models/hub"

    assert values["extraVolumes"] == [
        {
            "name": "model-cache-host",
            "hostPath": {"path": "/models", "type": "DirectoryOrCreate"},
        }
    ]
    assert values["extraVolumeMounts"] == [
        {"name": "model-cache-host", "mountPath": "/models"}
    ]
    assert values["extraEnv"][0]["name"] == "HF_TOKEN"
    assert values["extraEnv"][0]["valueFrom"]["secretKeyRef"]["optional"] is True


def test_dynamo_platform_wires_native_model_express_url():
    values = yaml.safe_load(DYNAMO_PLATFORM_VALUES.read_text())

    assert values["dynamo-operator"]["modelExpressURL"] == MODELEXPRESS_URL


def test_sglang_prefetches_speculative_draft_model_without_special_dgd_wiring():
    source = SGLANG_ARGS.read_text()

    assert 'getattr(\n        parsed_args, "speculative_draft_model_path", None\n    )' in source
    assert "fetch_model(speculative_draft_model_path)" in source
