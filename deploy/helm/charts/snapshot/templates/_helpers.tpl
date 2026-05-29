# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
{{/*
Expand the name of the chart.
*/}}
{{- define "snapshot.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "snapshot.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "snapshot.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "snapshot.labels" -}}
helm.sh/chart: {{ include "snapshot.chart" . }}
{{ include "snapshot.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/component: snapshot-agent
{{- end }}

{{/*
Selector labels
*/}}
{{- define "snapshot.selectorLabels" -}}
app.kubernetes.io/name: {{ include "snapshot.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "snapshot.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "snapshot.fullname" . ) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Fail fast on unsupported runtime.type values. Called once from daemonset.yaml.
*/}}
{{- define "snapshot.validateRuntime" -}}
{{- if not (has .Values.runtime.type (list "containerd" "crio")) }}
{{- fail (printf "runtime.type must be 'containerd' or 'crio', got %q" .Values.runtime.type) }}
{{- end }}
{{- end }}

{{/*
Resolve the runtime socket path on the host. Uses .Values.runtime.socketPath
when set, otherwise falls back to the per-runtime default. This is the path
the chart bind-mounts from the node into the agent container; it is NOT what
the agent dials inside the container (see snapshot.runtimeSocketContainer).
*/}}
{{- define "snapshot.runtimeSocket" -}}
{{- if .Values.runtime.socketPath }}
{{- .Values.runtime.socketPath }}
{{- else if eq .Values.runtime.type "crio" }}
{{- "/var/run/crio/crio.sock" }}
{{- else }}
{{- "/run/containerd/containerd.sock" }}
{{- end }}
{{- end }}

{{/*
Resolve the conventional in-container socket path for the chosen runtime.
The agent binary's runtime client falls back to this hard-coded default
inside the container even when --runtime-socket is set (the flag parser in
the published snapshot-agent images does not actually re-route the dial),
so we always mount the host socket dir at the conventional path and pass
the same path as --runtime-socket. That way non-default host socket layouts
(k3s puts it at /run/k3s/containerd/containerd.sock; RKE2 at
/run/k3s-containerd/containerd.sock; CRI-O on RHCOS at /run/crio/crio.sock)
work without forking the agent image.
*/}}
{{- define "snapshot.runtimeSocketContainer" -}}
{{- if eq .Values.runtime.type "crio" -}}/var/run/crio/crio.sock{{- else -}}/run/containerd/containerd.sock{{- end -}}
{{- end }}

{{/*
Host directory holding per-container storage (overlay upperdirs the agent
reads for rootfs-diff capture, and CRI-O config.json fallback). Uses
.Values.runtime.storageDir when set, otherwise falls back to the per-runtime
default. k3s stores containerd state under /var/lib/rancher/k3s/agent/containerd
rather than /var/lib/containerd, so single-node k3s and RKE2 clusters need to
override this.
*/}}
{{- define "snapshot.runtimeStorageDir" -}}
{{- if .Values.runtime.storageDir -}}
{{- .Values.runtime.storageDir -}}
{{- else if eq .Values.runtime.type "crio" -}}/var/lib/containers{{- else -}}/var/lib/containerd{{- end -}}
{{- end }}
