/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package v1alpha1

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/utils/ptr"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
)

// roundTripFromV1beta1 converts a v1beta1 DGD to v1alpha1 and back, returning
// the final v1beta1 object. For any valid v1beta1 input V the returned V'
// must equal V (syntactic round-trip invariant).
func roundTripFromV1beta1(t *testing.T, src *v1beta1.DynamoGraphDeployment) *v1beta1.DynamoGraphDeployment {
	t.Helper()
	a := &DynamoGraphDeployment{}
	if err := a.ConvertFrom(src); err != nil {
		t.Fatalf("ConvertFrom: %v", err)
	}
	out := &v1beta1.DynamoGraphDeployment{}
	if err := a.ConvertTo(out); err != nil {
		t.Fatalf("ConvertTo: %v", err)
	}
	return out
}

// roundTripFromV1alpha1 converts a v1alpha1 DGD to v1beta1 and back. The
// returned object should equal the input for v1alpha1 shapes that survive the
// full round-trip. Services-map ordering is not preserved (set-based equality
// is used by the caller when needed).
func roundTripFromV1alpha1(t *testing.T, src *DynamoGraphDeployment) *DynamoGraphDeployment {
	t.Helper()
	b := &v1beta1.DynamoGraphDeployment{}
	if err := src.ConvertTo(b); err != nil {
		t.Fatalf("ConvertTo: %v", err)
	}
	out := &DynamoGraphDeployment{}
	if err := out.ConvertFrom(b); err != nil {
		t.Fatalf("ConvertFrom: %v", err)
	}
	return out
}

func TestDGD_RoundTrip_Empty(t *testing.T) {
	src := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "empty", Namespace: "ns"},
	}
	got := roundTripFromV1beta1(t, src)
	if diff := cmp.Diff(src, got); diff != "" {
		t.Errorf("round-trip mismatch (-want +got):\n%s", diff)
	}
}

func TestDGD_RoundTrip_Minimal(t *testing.T) {
	replicas := int32(2)
	src := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "min", Namespace: "ns"},
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			BackendFramework: "vllm",
			Services: []v1beta1.DynamoComponentDeploymentService{
				{
					Name: "worker",
					DynamoComponentDeploymentSharedSpec: v1beta1.DynamoComponentDeploymentSharedSpec{
						ComponentType: v1beta1.ComponentTypeWorker,
						Replicas:      &replicas,
					},
				},
			},
		},
	}
	got := roundTripFromV1beta1(t, src)
	if diff := cmp.Diff(src, got); diff != "" {
		t.Errorf("round-trip mismatch (-want +got):\n%s", diff)
	}
}

func TestDGD_RoundTrip_SpecLevelFields(t *testing.T) {
	src := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "spec", Namespace: "ns"},
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			Annotations:      map[string]string{"a": "1"},
			Labels:           map[string]string{"l": "v"},
			BackendFramework: "sglang",
			Env: []corev1.EnvVar{
				{Name: "FOO", Value: "bar"},
			},
			Restart: &v1beta1.Restart{
				ID: "r1",
				Strategy: &v1beta1.RestartStrategy{
					Type:  v1beta1.RestartStrategyTypeParallel,
					Order: []string{"a", "b"},
				},
			},
			TopologyConstraint: &v1beta1.SpecTopologyConstraint{
				TopologyProfile: "default",
				PackDomain:      v1beta1.TopologyDomain("rack"),
			},
		},
	}
	got := roundTripFromV1beta1(t, src)
	if diff := cmp.Diff(src, got); diff != "" {
		t.Errorf("round-trip mismatch (-want +got):\n%s", diff)
	}
}

func TestDGD_RoundTrip_MultipleServicesOrderStable(t *testing.T) {
	// Services in alphabetical order match what ConvertTo emits from the map.
	src := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "multi", Namespace: "ns"},
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			Services: []v1beta1.DynamoComponentDeploymentService{
				{Name: "aa-frontend", DynamoComponentDeploymentSharedSpec: v1beta1.DynamoComponentDeploymentSharedSpec{ComponentType: v1beta1.ComponentTypeFrontend}},
				{Name: "bb-worker", DynamoComponentDeploymentSharedSpec: v1beta1.DynamoComponentDeploymentSharedSpec{ComponentType: v1beta1.ComponentTypeWorker}},
				{Name: "cc-planner", DynamoComponentDeploymentSharedSpec: v1beta1.DynamoComponentDeploymentSharedSpec{ComponentType: v1beta1.ComponentTypePlanner}},
			},
		},
	}
	got := roundTripFromV1beta1(t, src)
	if diff := cmp.Diff(src, got); diff != "" {
		t.Errorf("round-trip mismatch (-want +got):\n%s", diff)
	}
}

func TestDGD_RoundTrip_Experimental(t *testing.T) {
	ref := "my-checkpoint"
	src := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "exp", Namespace: "ns"},
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			Services: []v1beta1.DynamoComponentDeploymentService{
				{
					Name: "worker",
					DynamoComponentDeploymentSharedSpec: v1beta1.DynamoComponentDeploymentSharedSpec{
						ComponentType: v1beta1.ComponentTypeWorker,
						Experimental: &v1beta1.ExperimentalSpec{
							GPUMemoryService: &v1beta1.GPUMemoryServiceSpec{
								Mode:            v1beta1.GMSModeIntraPod,
								DeviceClassName: "gpu.nvidia.com",
							},
							Failover: &v1beta1.FailoverSpec{
								Mode:       v1beta1.GMSModeIntraPod,
								NumShadows: 1,
							},
							Checkpoint: &v1beta1.ServiceCheckpointConfig{
								Mode:          v1beta1.CheckpointModeAuto,
								CheckpointRef: &ref,
							},
						},
					},
				},
			},
		},
	}
	got := roundTripFromV1beta1(t, src)
	if diff := cmp.Diff(src, got); diff != "" {
		t.Errorf("round-trip mismatch (-want +got):\n%s", diff)
	}
}

func TestDGD_RoundTrip_PodTemplate(t *testing.T) {
	shm := resource.MustParse("4Gi")
	src := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "pt", Namespace: "ns"},
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			Services: []v1beta1.DynamoComponentDeploymentService{
				{
					Name: "worker",
					DynamoComponentDeploymentSharedSpec: v1beta1.DynamoComponentDeploymentSharedSpec{
						ComponentType:    v1beta1.ComponentTypeWorker,
						SharedMemorySize: &shm,
						PodTemplate: &corev1.PodTemplateSpec{
							ObjectMeta: metav1.ObjectMeta{
								Annotations: map[string]string{"prom.io/scrape": "true"},
								Labels:      map[string]string{"tier": "gpu"},
							},
							Spec: corev1.PodSpec{
								Containers: []corev1.Container{
									{
										Name:  "main",
										Image: "dynamo:latest",
										Env: []corev1.EnvVar{
											{Name: "DYN_COMPONENT", Value: "worker"},
										},
										Resources: corev1.ResourceRequirements{
											Requests: corev1.ResourceList{
												corev1.ResourceCPU:                    resource.MustParse("2"),
												corev1.ResourceMemory:                 resource.MustParse("4Gi"),
												corev1.ResourceName("nvidia.com/gpu"): resource.MustParse("1"),
											},
										},
									},
								},
							},
						},
					},
				},
			},
		},
	}
	got := roundTripFromV1beta1(t, src)
	// corev1.ResourceList equality can be quantity-representation-sensitive;
	// use cmpopts to compare canonical forms.
	opts := cmp.Options{
		cmpopts.EquateEmpty(),
	}
	if diff := cmp.Diff(src, got, opts); diff != "" {
		t.Errorf("round-trip mismatch (-want +got):\n%s", diff)
	}
}

func TestDGD_RoundTrip_CompilationCache(t *testing.T) {
	src := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "cc", Namespace: "ns"},
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			Services: []v1beta1.DynamoComponentDeploymentService{
				{
					Name: "worker",
					DynamoComponentDeploymentSharedSpec: v1beta1.DynamoComponentDeploymentSharedSpec{
						ComponentType: v1beta1.ComponentTypeWorker,
						CompilationCache: &v1beta1.CompilationCacheConfig{
							PVCName:   "cache-pvc",
							MountPath: "/opt/cache",
						},
						PodTemplate: &corev1.PodTemplateSpec{
							Spec: corev1.PodSpec{
								Containers: []corev1.Container{
									{
										Name: "main",
										VolumeMounts: []corev1.VolumeMount{
											{Name: "cache-pvc", MountPath: "/opt/cache"},
										},
									},
								},
							},
						},
					},
				},
			},
		},
	}
	got := roundTripFromV1beta1(t, src)
	if diff := cmp.Diff(src, got, cmpopts.EquateEmpty()); diff != "" {
		t.Errorf("round-trip mismatch (-want +got):\n%s", diff)
	}
}

func TestDGD_RoundTrip_ScalingAdapter(t *testing.T) {
	src := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "sa", Namespace: "ns"},
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			Services: []v1beta1.DynamoComponentDeploymentService{
				{
					Name: "worker",
					DynamoComponentDeploymentSharedSpec: v1beta1.DynamoComponentDeploymentSharedSpec{
						ComponentType:  v1beta1.ComponentTypeWorker,
						ScalingAdapter: &v1beta1.ScalingAdapter{},
					},
				},
			},
		},
	}
	got := roundTripFromV1beta1(t, src)
	if diff := cmp.Diff(src, got); diff != "" {
		t.Errorf("round-trip mismatch (-want +got):\n%s", diff)
	}
}

// TestDGD_FromV1alpha1_PVCsPreserved verifies that legacy v1alpha1 PVCs survive
// a v1alpha1 -> v1beta1 -> v1alpha1 round-trip via the origin annotation.
func TestDGD_FromV1alpha1_PVCsPreserved(t *testing.T) {
	createTrue := true
	name := "model-pvc"
	src := &DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "pvc", Namespace: "ns"},
		Spec: DynamoGraphDeploymentSpec{
			PVCs: []PVC{
				{
					Create:           &createTrue,
					Name:             &name,
					StorageClass:     "standard",
					Size:             resource.MustParse("10Gi"),
					VolumeAccessMode: corev1.ReadWriteOnce,
				},
			},
		},
	}
	got := roundTripFromV1alpha1(t, src)
	if diff := cmp.Diff(src, got, cmpopts.EquateEmpty()); diff != "" {
		t.Errorf("round-trip mismatch (-want +got):\n%s", diff)
	}
}

// TestDGD_FromV1alpha1_DisabledExperimental verifies that v1alpha1
// GMS/Failover/Checkpoint with Enabled=false and payloads survive the
// round-trip via origin annotations.
func TestDGD_FromV1alpha1_DisabledExperimental(t *testing.T) {
	src := &DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "disabled", Namespace: "ns"},
		Spec: DynamoGraphDeploymentSpec{
			Services: map[string]*DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType: "worker",
					GPUMemoryService: &GPUMemoryServiceSpec{
						Enabled:         false,
						Mode:            GMSModeIntraPod,
						DeviceClassName: "gpu.nvidia.com",
					},
					Failover: &FailoverSpec{
						Enabled:    false,
						Mode:       GMSModeIntraPod,
						NumShadows: 1,
					},
				},
			},
		},
	}
	got := roundTripFromV1alpha1(t, src)
	if diff := cmp.Diff(src, got, cmpopts.EquateEmpty()); diff != "" {
		t.Errorf("round-trip mismatch (-want +got):\n%s", diff)
	}
}

// TestDGD_FromV1alpha1_SubComponentType verifies that a v1alpha1-only
// subComponentType string survives via origin annotation.
func TestDGD_FromV1alpha1_SubComponentType(t *testing.T) {
	src := &DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "sub", Namespace: "ns"},
		Spec: DynamoGraphDeploymentSpec{
			Services: map[string]*DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType:    "worker",
					SubComponentType: "prefill",
				},
			},
		},
	}
	got := roundTripFromV1alpha1(t, src)
	if diff := cmp.Diff(src, got, cmpopts.EquateEmpty()); diff != "" {
		t.Errorf("round-trip mismatch (-want +got):\n%s", diff)
	}
}

// -----------------------------------------------------------------------------
// Expanded coverage: status, rich shared-spec fields, pod-template details,
// v1alpha1-only shapes, annotation hygiene, JSON byte-identity.
// -----------------------------------------------------------------------------

// TestDGD_RoundTrip_Status exercises every populated Status sub-struct so that
// the ConvertTo / ConvertFrom status paths are covered (conditions, services
// map, restart, checkpoints, rollingUpdate).
func TestDGD_RoundTrip_Status(t *testing.T) {
	now := metav1.NewTime(metav1.Now().Rfc3339Copy().Time)
	later := metav1.NewTime(now.Time.Add(60 * time.Second))
	src := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "status", Namespace: "ns"},
		Status: v1beta1.DynamoGraphDeploymentStatus{
			ObservedGeneration: 7,
			State:              v1beta1.DGDStateSuccessful,
			Conditions: []metav1.Condition{
				{
					Type:               "Ready",
					Status:             metav1.ConditionTrue,
					Reason:             "AllServicesReady",
					Message:            "all services are ready",
					LastTransitionTime: now,
				},
			},
			Services: map[string]v1beta1.ServiceReplicaStatus{
				"worker": {
					ComponentKind:     v1beta1.ComponentKindDeployment,
					ComponentName:     "dgd-worker-0",
					ComponentNames:    []string{"dgd-worker-0", "dgd-worker-1"},
					Replicas:          2,
					UpdatedReplicas:   2,
					ReadyReplicas:     ptr.To(int32(2)),
					AvailableReplicas: ptr.To(int32(2)),
				},
			},
			Restart: &v1beta1.RestartStatus{
				ObservedID: "r-123",
				Phase:      v1beta1.RestartPhaseRestarting,
				InProgress: []string{"worker"},
			},
			Checkpoints: map[string]v1beta1.ServiceCheckpointStatus{
				"worker": {
					CheckpointName: "ckpt-abc",
					IdentityHash:   "sha256:deadbeef",
					Ready:          true,
				},
			},
			RollingUpdate: &v1beta1.RollingUpdateStatus{
				Phase:           v1beta1.RollingUpdatePhaseInProgress,
				StartTime:       &now,
				EndTime:         &later,
				UpdatedServices: []string{"worker"},
			},
		},
	}
	got := roundTripFromV1beta1(t, src)
	if diff := cmp.Diff(src, got); diff != "" {
		t.Errorf("round-trip mismatch (-want +got):\n%s", diff)
	}
}

// TestDGD_RoundTrip_FullSharedSpec covers every first-class v1beta1 shared-spec
// field that has not been exercised elsewhere (DynamoNamespace is v1alpha1-only
// so it lives in a separate test): GlobalDynamoNamespace, Multinode, ModelRef,
// per-service TopologyConstraint, EPPConfig.
func TestDGD_RoundTrip_FullSharedSpec(t *testing.T) {
	src := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "full", Namespace: "ns"},
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			Services: []v1beta1.DynamoComponentDeploymentService{
				{
					Name: "epp",
					DynamoComponentDeploymentSharedSpec: v1beta1.DynamoComponentDeploymentSharedSpec{
						ComponentType: v1beta1.ComponentTypeEPP,
						EPPConfig: &v1beta1.EPPConfig{
							ConfigMapRef: &corev1.ConfigMapKeySelector{
								LocalObjectReference: corev1.LocalObjectReference{Name: "epp-cfg"},
								Key:                  "config.yaml",
							},
						},
					},
				},
				{
					Name: "worker",
					DynamoComponentDeploymentSharedSpec: v1beta1.DynamoComponentDeploymentSharedSpec{
						ComponentType:         v1beta1.ComponentTypeWorker,
						GlobalDynamoNamespace: true,
						Multinode:             &v1beta1.MultinodeSpec{NodeCount: 4},
						ModelRef: &v1beta1.ModelReference{
							Name:     "llama-3-70b-instruct",
							Revision: "v1",
						},
						TopologyConstraint: &v1beta1.TopologyConstraint{
							PackDomain: v1beta1.TopologyDomain("rack"),
						},
					},
				},
			},
		},
	}
	got := roundTripFromV1beta1(t, src)
	if diff := cmp.Diff(src, got); diff != "" {
		t.Errorf("round-trip mismatch (-want +got):\n%s", diff)
	}
}

// TestDGD_RoundTrip_PodTemplateProbesAndEnvFrom covers the main-container
// fields that decomposePodTemplate preserves through ExtraPodSpec.MainContainer:
// EnvFrom, LivenessProbe, ReadinessProbe, StartupProbe.
func TestDGD_RoundTrip_PodTemplateProbesAndEnvFrom(t *testing.T) {
	src := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "probes", Namespace: "ns"},
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			Services: []v1beta1.DynamoComponentDeploymentService{
				{
					Name: "worker",
					DynamoComponentDeploymentSharedSpec: v1beta1.DynamoComponentDeploymentSharedSpec{
						ComponentType: v1beta1.ComponentTypeWorker,
						PodTemplate: &corev1.PodTemplateSpec{
							Spec: corev1.PodSpec{
								Containers: []corev1.Container{
									{
										Name:  "main",
										Image: "dynamo:latest",
										EnvFrom: []corev1.EnvFromSource{
											{
												SecretRef: &corev1.SecretEnvSource{
													LocalObjectReference: corev1.LocalObjectReference{Name: "aws-secret"},
												},
											},
										},
										LivenessProbe: &corev1.Probe{
											ProbeHandler: corev1.ProbeHandler{
												HTTPGet: &corev1.HTTPGetAction{Path: "/healthz", Port: intstrFromInt32(8080)},
											},
											InitialDelaySeconds: 5,
										},
										ReadinessProbe: &corev1.Probe{
											ProbeHandler: corev1.ProbeHandler{
												HTTPGet: &corev1.HTTPGetAction{Path: "/ready", Port: intstrFromInt32(8080)},
											},
										},
										StartupProbe: &corev1.Probe{
											ProbeHandler: corev1.ProbeHandler{
												HTTPGet: &corev1.HTTPGetAction{Path: "/startup", Port: intstrFromInt32(8080)},
											},
											FailureThreshold: 30,
										},
									},
								},
							},
						},
					},
				},
			},
		},
	}
	got := roundTripFromV1beta1(t, src)
	if diff := cmp.Diff(src, got, cmpopts.EquateEmpty()); diff != "" {
		t.Errorf("round-trip mismatch (-want +got):\n%s", diff)
	}
}

// TestDGD_RoundTrip_PodSpecExtras covers the non-main-container PodSpec fields
// that flow through ExtraPodSpec.PodSpec: NodeSelector, Tolerations,
// ServiceAccountName, ImagePullSecrets, Volumes.
func TestDGD_RoundTrip_PodSpecExtras(t *testing.T) {
	src := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "extras", Namespace: "ns"},
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			Services: []v1beta1.DynamoComponentDeploymentService{
				{
					Name: "worker",
					DynamoComponentDeploymentSharedSpec: v1beta1.DynamoComponentDeploymentSharedSpec{
						ComponentType: v1beta1.ComponentTypeWorker,
						PodTemplate: &corev1.PodTemplateSpec{
							Spec: corev1.PodSpec{
								NodeSelector:       map[string]string{"node-pool": "gpu"},
								ServiceAccountName: "dynamo-sa",
								ImagePullSecrets:   []corev1.LocalObjectReference{{Name: "ghcr-creds"}},
								Tolerations: []corev1.Toleration{
									{Key: "nvidia.com/gpu", Operator: corev1.TolerationOpExists, Effect: corev1.TaintEffectNoSchedule},
								},
								Volumes: []corev1.Volume{
									{
										Name: "cache",
										VolumeSource: corev1.VolumeSource{
											EmptyDir: &corev1.EmptyDirVolumeSource{},
										},
									},
								},
								Containers: []corev1.Container{
									{Name: "main", Image: "dynamo:latest"},
								},
							},
						},
					},
				},
			},
		},
	}
	got := roundTripFromV1beta1(t, src)
	if diff := cmp.Diff(src, got, cmpopts.EquateEmpty()); diff != "" {
		t.Errorf("round-trip mismatch (-want +got):\n%s", diff)
	}
}

// TestDGD_RoundTrip_FrontendSidecar starts from v1beta1 (hub) with the
// FrontendSidecar string naming a sidecar container in podTemplate.containers.
func TestDGD_RoundTrip_FrontendSidecar(t *testing.T) {
	src := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "fs", Namespace: "ns"},
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			Services: []v1beta1.DynamoComponentDeploymentService{
				{
					Name: "epp",
					DynamoComponentDeploymentSharedSpec: v1beta1.DynamoComponentDeploymentSharedSpec{
						ComponentType:   v1beta1.ComponentTypeEPP,
						FrontendSidecar: ptr.To("sidecar-frontend"),
						PodTemplate: &corev1.PodTemplateSpec{
							Spec: corev1.PodSpec{
								Containers: []corev1.Container{
									{Name: "main", Image: "dynamo:latest"},
									{
										Name:  "sidecar-frontend",
										Image: "dynamo-frontend:latest",
										Args:  []string{"-m", "dynamo.frontend"},
									},
								},
							},
						},
					},
				},
			},
		},
	}
	got := roundTripFromV1beta1(t, src)
	if diff := cmp.Diff(src, got, cmpopts.EquateEmpty()); diff != "" {
		t.Errorf("round-trip mismatch (-want +got):\n%s", diff)
	}
}

// TestDGD_RoundTrip_SharedMemoryDisabledZero asserts that an explicit
// size="0" (Disabled=true equivalent) survives. Starts from v1beta1.
func TestDGD_RoundTrip_SharedMemoryDisabledZero(t *testing.T) {
	zero := resource.MustParse("0")
	src := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "shm", Namespace: "ns"},
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			Services: []v1beta1.DynamoComponentDeploymentService{
				{
					Name: "worker",
					DynamoComponentDeploymentSharedSpec: v1beta1.DynamoComponentDeploymentSharedSpec{
						ComponentType:    v1beta1.ComponentTypeWorker,
						SharedMemorySize: &zero,
					},
				},
			},
		},
	}
	got := roundTripFromV1beta1(t, src)
	if diff := cmp.Diff(src, got); diff != "" {
		t.Errorf("round-trip mismatch (-want +got):\n%s", diff)
	}
}

// TestDGD_FromV1alpha1_SharedMemoryEdgeCases covers the two v1alpha1-only
// SharedMemorySpec shapes that need origin annotations to round-trip:
// Disabled=true and the empty struct &SharedMemorySpec{}.
func TestDGD_FromV1alpha1_SharedMemoryEdgeCases(t *testing.T) {
	cases := []struct {
		name string
		shm  *SharedMemorySpec
	}{
		{name: "disabled", shm: &SharedMemorySpec{Disabled: true}},
		{name: "empty", shm: &SharedMemorySpec{}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			src := &DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{Name: "shm-" + tc.name, Namespace: "ns"},
				Spec: DynamoGraphDeploymentSpec{
					Services: map[string]*DynamoComponentDeploymentSharedSpec{
						"worker": {
							ComponentType: "worker",
							SharedMemory:  tc.shm,
						},
					},
				},
			}
			got := roundTripFromV1alpha1(t, src)
			if diff := cmp.Diff(src, got, cmpopts.EquateEmpty()); diff != "" {
				t.Errorf("round-trip mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// TestDGD_FromV1alpha1_ScalingAdapterDisabled checks that the otherwise-unreachable
// &ScalingAdapter{Enabled:false} shape round-trips via the scaling-adapter-disabled
// annotation.
func TestDGD_FromV1alpha1_ScalingAdapterDisabled(t *testing.T) {
	src := &DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "sad", Namespace: "ns"},
		Spec: DynamoGraphDeploymentSpec{
			Services: map[string]*DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType:  "worker",
					ScalingAdapter: &ScalingAdapter{Enabled: false},
				},
			},
		},
	}
	got := roundTripFromV1alpha1(t, src)
	if diff := cmp.Diff(src, got, cmpopts.EquateEmpty()); diff != "" {
		t.Errorf("round-trip mismatch (-want +got):\n%s", diff)
	}
}

// TestDGD_FromV1alpha1_CheckpointDisabled checks that Checkpoint{Enabled:false}
// with a non-trivial payload survives via annotation.
func TestDGD_FromV1alpha1_CheckpointDisabled(t *testing.T) {
	ref := "my-ckpt"
	src := &DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "ckpt-disabled", Namespace: "ns"},
		Spec: DynamoGraphDeploymentSpec{
			Services: map[string]*DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType: "worker",
					Checkpoint: &ServiceCheckpointConfig{
						Enabled:       false,
						Mode:          CheckpointModeAuto,
						CheckpointRef: &ref,
					},
				},
			},
		},
	}
	got := roundTripFromV1alpha1(t, src)
	if diff := cmp.Diff(src, got, cmpopts.EquateEmpty()); diff != "" {
		t.Errorf("round-trip mismatch (-want +got):\n%s", diff)
	}
}

// TestDGD_FromV1alpha1_DynamoNamespaceAndServiceName verifies the two simple
// v1alpha1-only string fields round-trip via annotations.
func TestDGD_FromV1alpha1_DynamoNamespaceAndServiceName(t *testing.T) {
	ns := "legacy-dyn-ns"
	src := &DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "legacy", Namespace: "ns"},
		Spec: DynamoGraphDeploymentSpec{
			Services: map[string]*DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType:   "worker",
					ServiceName:     "worker-svc",
					DynamoNamespace: &ns,
				},
			},
		},
	}
	got := roundTripFromV1alpha1(t, src)
	if diff := cmp.Diff(src, got, cmpopts.EquateEmpty()); diff != "" {
		t.Errorf("round-trip mismatch (-want +got):\n%s", diff)
	}
}

// TestDGD_FromV1alpha1_PerServiceAnnotationsAndLabels verifies that v1alpha1
// per-service Annotations/Labels (which target Pod+Service+Ingress in the
// v1alpha1 reconcile model and cannot be faithfully placed in
// podTemplate.metadata alone) are preserved via origin annotations.
func TestDGD_FromV1alpha1_PerServiceAnnotationsAndLabels(t *testing.T) {
	src := &DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "pa", Namespace: "ns"},
		Spec: DynamoGraphDeploymentSpec{
			Services: map[string]*DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType: "worker",
					Annotations:   map[string]string{"team": "alpha"},
					Labels:        map[string]string{"tier": "gpu"},
				},
			},
		},
	}
	got := roundTripFromV1alpha1(t, src)
	if diff := cmp.Diff(src, got, cmpopts.EquateEmpty()); diff != "" {
		t.Errorf("round-trip mismatch (-want +got):\n%s", diff)
	}
}

// TestDGD_FromV1alpha1_AutoscalingAndIngress covers both deprecated blocks.
func TestDGD_FromV1alpha1_AutoscalingAndIngress(t *testing.T) {
	src := &DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "ai", Namespace: "ns"},
		Spec: DynamoGraphDeploymentSpec{
			Services: map[string]*DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType: "worker",
					Autoscaling: &Autoscaling{
						Enabled:     true,
						MinReplicas: 1,
						MaxReplicas: 5,
					},
					Ingress: &IngressSpec{
						Enabled: true,
						Host:    "api.example.com",
					},
				},
			},
		},
	}
	got := roundTripFromV1alpha1(t, src)
	if diff := cmp.Diff(src, got, cmpopts.EquateEmpty()); diff != "" {
		t.Errorf("round-trip mismatch (-want +got):\n%s", diff)
	}
}

// TestDGD_FromV1alpha1_Resources_ForwardOnly asserts that a v1alpha1 Resources
// struct with a non-default GPUType and Custom keys translates into the
// expected corev1.ResourceList on the v1beta1 side. Full bitwise round-trip
// isn't promised for this shape (v1beta1 -> v1alpha1 folds Resources into
// ExtraPodSpec.MainContainer), but the forward translation is exercised here
// to cover resourcesToNative's GPUType/Custom branches.
func TestDGD_FromV1alpha1_Resources_ForwardOnly(t *testing.T) {
	src := &DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "res", Namespace: "ns"},
		Spec: DynamoGraphDeploymentSpec{
			Services: map[string]*DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType: "worker",
					Resources: &Resources{
						Requests: &ResourceItem{
							CPU:     "2",
							Memory:  "4Gi",
							GPU:     "2",
							GPUType: "gpu.intel.com/xe",
							Custom:  map[string]string{"example.com/fpga": "1"},
						},
						Limits: &ResourceItem{CPU: "4", Memory: "8Gi"},
					},
				},
			},
		},
	}
	b := &v1beta1.DynamoGraphDeployment{}
	if err := src.ConvertTo(b); err != nil {
		t.Fatalf("ConvertTo: %v", err)
	}
	if len(b.Spec.Services) != 1 {
		t.Fatalf("expected 1 service, got %d", len(b.Spec.Services))
	}
	pt := b.Spec.Services[0].PodTemplate
	if pt == nil || len(pt.Spec.Containers) == 0 {
		t.Fatalf("expected main container in podTemplate, got %+v", pt)
	}
	req := pt.Spec.Containers[0].Resources.Requests
	gpu := req[corev1.ResourceName("gpu.intel.com/xe")]
	if gpu.String() != "2" {
		t.Errorf("gpu.intel.com/xe = %q, want %q", gpu.String(), "2")
	}
	fpga := req[corev1.ResourceName("example.com/fpga")]
	if fpga.String() != "1" {
		t.Errorf("example.com/fpga = %q, want %q", fpga.String(), "1")
	}
}

// TestDGD_ConvertFrom_ScrubsLingeringAnnotations asserts that a stale
// "nvidia.com/dgd-svc-*" annotation that does not correspond to any current
// service is dropped by ConvertFrom. This protects users from leaking origin
// annotations across deletions.
func TestDGD_ConvertFrom_ScrubsLingeringAnnotations(t *testing.T) {
	src := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "scrub",
			Namespace: "ns",
			Annotations: map[string]string{
				"nvidia.com/dgd-svc-deleted-dynamo-namespace": "stale-value",
				"user/keep-me": "kept",
			},
		},
	}
	a := &DynamoGraphDeployment{}
	if err := a.ConvertFrom(src); err != nil {
		t.Fatalf("ConvertFrom: %v", err)
	}
	if _, stale := a.ObjectMeta.Annotations["nvidia.com/dgd-svc-deleted-dynamo-namespace"]; stale {
		t.Errorf("stale dgd-svc- annotation was not scrubbed: %v", a.ObjectMeta.Annotations)
	}
	if v, ok := a.ObjectMeta.Annotations["user/keep-me"]; !ok || v != "kept" {
		t.Errorf("user annotations must be preserved, got %v", a.ObjectMeta.Annotations)
	}
}

// TestDGD_JSONRoundTrip_Bytes is the strongest form of syntactic equality:
// marshal the v1beta1 input to JSON, round-trip through v1alpha1, marshal the
// result, and require byte-identical output. This catches any nil-vs-empty
// divergence that cmp.Diff+EquateEmpty would collapse.
func TestDGD_JSONRoundTrip_Bytes(t *testing.T) {
	shm := resource.MustParse("4Gi")
	replicas := int32(2)
	src := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "json", Namespace: "ns"},
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			BackendFramework: "vllm",
			Env:              []corev1.EnvVar{{Name: "FOO", Value: "bar"}},
			Restart:          &v1beta1.Restart{ID: "r1"},
			Services: []v1beta1.DynamoComponentDeploymentService{
				{
					Name: "worker",
					DynamoComponentDeploymentSharedSpec: v1beta1.DynamoComponentDeploymentSharedSpec{
						ComponentType:    v1beta1.ComponentTypeWorker,
						Replicas:         &replicas,
						SharedMemorySize: &shm,
						ScalingAdapter:   &v1beta1.ScalingAdapter{},
						Experimental: &v1beta1.ExperimentalSpec{
							GPUMemoryService: &v1beta1.GPUMemoryServiceSpec{
								Mode:            v1beta1.GMSModeIntraPod,
								DeviceClassName: "gpu.nvidia.com",
							},
						},
					},
				},
			},
		},
	}
	got := roundTripFromV1beta1(t, src)

	wantBytes, err := json.Marshal(src)
	if err != nil {
		t.Fatalf("marshal src: %v", err)
	}
	gotBytes, err := json.Marshal(got)
	if err != nil {
		t.Fatalf("marshal got: %v", err)
	}
	if string(wantBytes) != string(gotBytes) {
		t.Errorf("JSON byte-level round-trip mismatch:\nwant: %s\ngot:  %s", wantBytes, gotBytes)
	}
}

// intstrFromInt32 returns an intstr.IntOrString wrapping the given int32 port.
// Kept as a small helper so probe definitions stay compact in the expanded
// round-trip tests.
func intstrFromInt32(v int32) intstr.IntOrString {
	return intstr.FromInt32(v)
}

// TestDGD_FromV1alpha1_FrontendSidecarFullRoundTrip exercises the v1alpha1-first
// FrontendSidecar path: the full FrontendSidecarSpec is stashed under the
// suffixFrontendSidecar origin annotation on ConvertTo (covers
// buildPodTemplateTo's "full spec -> name reference" branch) and restored from
// that annotation on ConvertFrom (covers decomposePodTemplate's
// "annotation present -> unmarshal + drop container from other" branch).
func TestDGD_FromV1alpha1_FrontendSidecarFullRoundTrip(t *testing.T) {
	secret := "frontend-secret"
	src := &DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "fs-full", Namespace: "ns"},
		Spec: DynamoGraphDeploymentSpec{
			Services: map[string]*DynamoComponentDeploymentSharedSpec{
				"epp": {
					ComponentType: "epp",
					FrontendSidecar: &FrontendSidecarSpec{
						Image:         "dynamo-frontend:1.2.3",
						Args:          []string{"-m", "dynamo.frontend", "--router-mode", "direct"},
						EnvFromSecret: &secret,
						Envs: []corev1.EnvVar{
							{Name: "FRONTEND_FLAG", Value: "true"},
						},
					},
				},
			},
		},
	}

	b := &v1beta1.DynamoGraphDeployment{}
	if err := src.ConvertTo(b); err != nil {
		t.Fatalf("ConvertTo: %v", err)
	}
	if len(b.Spec.Services) != 1 {
		t.Fatalf("expected 1 service on v1beta1, got %d", len(b.Spec.Services))
	}
	svc := b.Spec.Services[0]
	if svc.FrontendSidecar == nil || *svc.FrontendSidecar != "sidecar-frontend" {
		t.Fatalf("expected FrontendSidecar pointer = %q, got %v", "sidecar-frontend", svc.FrontendSidecar)
	}
	if svc.PodTemplate == nil {
		t.Fatalf("expected podTemplate to carry the sidecar container")
	}
	var sidecar *corev1.Container
	for i := range svc.PodTemplate.Spec.Containers {
		if svc.PodTemplate.Spec.Containers[i].Name == "sidecar-frontend" {
			sidecar = &svc.PodTemplate.Spec.Containers[i]
			break
		}
	}
	if sidecar == nil {
		t.Fatalf("expected 'sidecar-frontend' container in podTemplate, got %+v", svc.PodTemplate.Spec.Containers)
	}
	if sidecar.Image != "dynamo-frontend:1.2.3" {
		t.Errorf("sidecar image: got %q, want %q", sidecar.Image, "dynamo-frontend:1.2.3")
	}
	want := "nvidia.com/dgd-svc-epp-" + suffixFrontendSidecar
	if _, ok := b.Annotations[want]; !ok {
		t.Errorf("expected origin annotation %q to be set, got %v", want, b.Annotations)
	}

	got := &DynamoGraphDeployment{}
	if err := got.ConvertFrom(b); err != nil {
		t.Fatalf("ConvertFrom: %v", err)
	}
	if diff := cmp.Diff(src, got, cmpopts.EquateEmpty()); diff != "" {
		t.Errorf("FrontendSidecar round-trip mismatch (-want +got):\n%s", diff)
	}
	if _, leaked := got.Annotations["nvidia.com/dgd-svc-epp-"+suffixFrontendSidecar]; leaked {
		t.Errorf("origin annotation was not consumed: %v", got.Annotations)
	}
}

// TestDGD_FromV1alpha1_GMSEnabledFalseEmptyPayload targets the
// "Enabled=false with zero-valued payload -> `{}` annotation" branch in
// convertExperimentalTo for GPUMemoryService. The v1alpha1 pointer
// &GPUMemoryServiceSpec{} (no Mode, no DeviceClassName) must round-trip
// through the annotation without being collapsed to nil.
func TestDGD_FromV1alpha1_GMSEnabledFalseEmptyPayload(t *testing.T) {
	src := &DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "gms-empty", Namespace: "ns"},
		Spec: DynamoGraphDeploymentSpec{
			Services: map[string]*DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType:    "worker",
					GPUMemoryService: &GPUMemoryServiceSpec{Enabled: false},
				},
			},
		},
	}
	b := &v1beta1.DynamoGraphDeployment{}
	if err := src.ConvertTo(b); err != nil {
		t.Fatalf("ConvertTo: %v", err)
	}
	key := "nvidia.com/dgd-svc-worker-" + suffixGMSDisabled
	if v, ok := b.Annotations[key]; !ok || v != `{}` {
		t.Errorf("expected annotation %q=%q, got %v", key, `{}`, b.Annotations)
	}

	got := roundTripFromV1alpha1(t, src)
	if diff := cmp.Diff(src, got, cmpopts.EquateEmpty()); diff != "" {
		t.Errorf("GMS empty-payload round-trip mismatch (-want +got):\n%s", diff)
	}
}

// TestDGD_FromV1alpha1_FailoverEnabledFalseEmptyPayload targets the
// sibling branch for Failover.
func TestDGD_FromV1alpha1_FailoverEnabledFalseEmptyPayload(t *testing.T) {
	src := &DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "fo-empty", Namespace: "ns"},
		Spec: DynamoGraphDeploymentSpec{
			Services: map[string]*DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType: "worker",
					Failover:      &FailoverSpec{Enabled: false},
				},
			},
		},
	}
	b := &v1beta1.DynamoGraphDeployment{}
	if err := src.ConvertTo(b); err != nil {
		t.Fatalf("ConvertTo: %v", err)
	}
	key := "nvidia.com/dgd-svc-worker-" + suffixFailoverDisabled
	if v, ok := b.Annotations[key]; !ok || v != `{}` {
		t.Errorf("expected annotation %q=%q, got %v", key, `{}`, b.Annotations)
	}

	got := roundTripFromV1alpha1(t, src)
	if diff := cmp.Diff(src, got, cmpopts.EquateEmpty()); diff != "" {
		t.Errorf("Failover empty-payload round-trip mismatch (-want +got):\n%s", diff)
	}
}

// TestDGD_FromV1alpha1_CheckpointEnabledFalseEmptyPayload covers the same
// "Enabled=false with zero-valued payload -> `{}` annotation" branch for the
// Checkpoint sibling in convertExperimentalTo.
func TestDGD_FromV1alpha1_CheckpointEnabledFalseEmptyPayload(t *testing.T) {
	src := &DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "ckpt-empty", Namespace: "ns"},
		Spec: DynamoGraphDeploymentSpec{
			Services: map[string]*DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType: "worker",
					Checkpoint:    &ServiceCheckpointConfig{Enabled: false},
				},
			},
		},
	}
	b := &v1beta1.DynamoGraphDeployment{}
	if err := src.ConvertTo(b); err != nil {
		t.Fatalf("ConvertTo: %v", err)
	}
	key := "nvidia.com/dgd-svc-worker-" + suffixCheckpointDisabl
	if v, ok := b.Annotations[key]; !ok || v != `{}` {
		t.Errorf("expected annotation %q=%q, got %v", key, `{}`, b.Annotations)
	}

	got := roundTripFromV1alpha1(t, src)
	if diff := cmp.Diff(src, got, cmpopts.EquateEmpty()); diff != "" {
		t.Errorf("Checkpoint empty-payload round-trip mismatch (-want +got):\n%s", diff)
	}
}
