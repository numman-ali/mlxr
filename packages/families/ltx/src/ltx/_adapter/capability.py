from __future__ import annotations

from typing import TYPE_CHECKING

from mlx_runtime_core import (
    ConversionPlan,
    ConversionSource,
    ExecutionProfile,
    FamilyInspection,
    FetchPolicy,
    LoadedModelHandle,
    PortableArtifact,
)
from mlx_runtime_schemas import (
    CapabilityDescriptor,
    ExtensionSchemaDescriptor,
    HardwareTier,
    PortableArtifactRecord,
    ResolvedSource,
)

from .state import LoadedLTXRuntimeState

if TYPE_CHECKING:
    from ..adapter import LTXFamilyAdapter


def inspect_source(self: LTXFamilyAdapter, source: ResolvedSource) -> FamilyInspection:
    role_candidates = self._role_candidates(source)
    return FamilyInspection(
        family=self.family_id,
        variant="fast",
        tasks=("video.generate", "video.condition.image", "video.condition.audio"),
        scheduler_class="media_video_dit",
        metadata={
            "source_provider": source.provider,
            "required_source_roles": list(self._required_roles),
            "role_candidates": role_candidates,
            "bundle_ready": "bundle" in role_candidates,
            "implemented_tasks": [
                "video.generate",
                "video.condition.image",
                "video.condition.audio",
            ],
            "upstream_tasks": [
                "video.generate",
                "video.condition.image",
                "video.condition.video",
                "video.condition.audio",
                "video.interpolate",
                "video.retake",
            ],
        },
    )


def fetch_policy_for_conversion(
    self: LTXFamilyAdapter, role: str, source: ResolvedSource
) -> FetchPolicy:
    del source
    allow_patterns: tuple[str, ...]
    if role not in (*self._required_roles, "bundle"):
        raise ValueError(f"LTX does not support conversion role '{role}'")
    if role == "checkpoint":
        allow_patterns = (self._checkpoint_filename, "*.json")
    elif role == "spatial_upsampler":
        allow_patterns = (self._spatial_upsampler_filename, "*.json")
    elif role == "text_encoder":
        allow_patterns = (
            "*.json",
            "*.safetensors",
            "*.txt",
            "*.model",
            "*.tiktoken",
        )
    else:
        allow_patterns = (
            self._checkpoint_filename,
            self._spatial_upsampler_filename,
            "*.json",
            "*.safetensors",
            "*.txt",
            "*.model",
            "*.tiktoken",
        )
    return FetchPolicy(
        allow_patterns=allow_patterns,
        options={
            "family": self.family_id,
            "role": role,
            "reason": "ltx-fast-conversion",
            "strict_local_text_encoding": True,
        },
    )


def convert(
    self: LTXFamilyAdapter,
    sources: dict[str, ConversionSource],
    plan: ConversionPlan,
) -> PortableArtifact:
    prepared = self._prepare_components(sources)
    component_records, payload_items = self._artifact_components(prepared)
    artifact_digest = self._artifact_digest(component_records, plan)
    primary_provenance = prepared["checkpoint"].provenance
    policy = self._combined_policy(prepared)
    capability = CapabilityDescriptor(
        model_id=plan.model_id,
        artifact_digest=artifact_digest,
        family=self.family_id,
        family_variant="fast",
        tasks=["video.generate", "video.condition.image", "video.condition.audio"],
        modalities_in=["text", "image", "audio"],
        modalities_out=["video", "audio"],
        constraints={
            "width": {"multiple_of": 32},
            "height": {"multiple_of": 32},
            "num_frames": {"formula": "8n+1"},
        },
        conditioning={"image": True, "video": False, "audio": True, "lora": False},
        profiles_by_task={
            "video.generate": ["bf16"],
            "video.condition.image": ["bf16"],
            "video.condition.audio": ["bf16"],
        },
        streaming={
            "progress_events": True,
            "partial_artifacts": False,
            "segment_events": False,
            "token_deltas": False,
        },
        artifacts_out=["mp4", "wav"],
        scheduler_class="media_video_dit",
        hardware_tiers=[
            HardwareTier(
                tier="recommended", memory_gb=64, notes="Full fast profile target"
            ),
            HardwareTier(
                tier="degraded",
                memory_gb=32,
                notes="Reduced profiles only; benchmark-gated",
            ),
        ],
        dependencies={
            "checkpoint": {
                "required": True,
                "role": "checkpoint",
                "kind": "file",
            },
            "spatial_upsampler": {
                "required": True,
                "role": "spatial_upsampler",
                "kind": "file",
            },
            "text_encoder": {
                "required": True,
                "role": "text_encoder",
                "kind": "directory",
                "mode": "strict-local",
            },
            "media_encode": {"required": True, "policy": "runtime-managed"},
        },
        policy=policy,
        extensions_schema=ExtensionSchemaDescriptor(namespace="ltx", version="1"),
        metadata={
            "artifact_layout": "componentized_payload",
            "precision": plan.precision,
            "primary_component_role": "checkpoint",
            "source_count": len(sources),
            "implemented_surface": {
                "tasks": [
                    "video.generate",
                    "video.condition.image",
                    "video.condition.audio",
                ],
                "pipeline_variants": ["distilled_two_stage"],
                "artifact_formats": ["mp4", "wav"],
                "audio_output_modes": ["muxed_mp4", "wav"],
            },
            "upstream_surface": {
                "tasks": [
                    "video.generate",
                    "video.condition.image",
                    "video.condition.video",
                    "video.condition.audio",
                    "video.interpolate",
                    "video.retake",
                ],
                "pipeline_variants": [
                    "distilled_two_stage",
                    "one_stage",
                    "two_stage",
                    "two_stage_hq",
                ],
                "control_variants": [
                    "ic_lora",
                    "union_ic_lora",
                    "distilled_lora",
                ],
            },
            "capability_matrix_doc": "docs/research/11-ltx-capability-matrix.md",
        },
    )
    record = PortableArtifactRecord(
        model_id=plan.model_id,
        artifact_digest=artifact_digest,
        family=self.family_id,
        family_variant="fast",
        format_version="0.2.0",
        weight_format="source_packaged_fastpath_assets",
        storage_key=f"ltx/{plan.model_id}/{artifact_digest}",
        capability=capability,
        provenance=primary_provenance,
        components=component_records,
        metadata={
            "artifact_layout": "componentized_payload",
            "primary_component_role": "checkpoint",
            "required_source_roles": list(self._required_roles),
            "source_count": len(sources),
        },
    )
    return PortableArtifact(
        record=record,
        storage_path=None,
        payload_items=tuple(payload_items),
    )


def load(
    self: LTXFamilyAdapter, artifact: PortableArtifact, profile: ExecutionProfile
) -> LoadedModelHandle:
    if artifact.storage_path is None or not artifact.record.components:
        return LoadedModelHandle(
            model_id=artifact.record.model_id,
            family=self.family_id,
            artifact_digest=artifact.record.artifact_digest,
            capability=artifact.record.capability,
            metadata={
                "status": "scaffold",
                "task": profile.task,
                "profile": profile.profile,
                "device": profile.device,
            },
        )

    component_paths = self._component_paths(
        artifact.storage_path, artifact.record.components
    )
    runtime_state = LoadedLTXRuntimeState(component_paths=component_paths)
    return LoadedModelHandle(
        model_id=artifact.record.model_id,
        family=self.family_id,
        artifact_digest=artifact.record.artifact_digest,
        capability=artifact.record.capability,
        metadata={
            "artifact_layout": "componentized_payload",
            "component_paths": {
                role: str(path) for role, path in component_paths.items()
            },
            "task": profile.task,
            "profile": profile.profile,
            "device": profile.device,
            self._runtime_state_key: runtime_state,
        },
    )


def capabilities(
    self: LTXFamilyAdapter, artifact: PortableArtifact
) -> CapabilityDescriptor:
    return artifact.record.capability
