from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from mlxr.core.runtime import (
    ConversionPlan,
    ConversionSource,
    ExecutionProfile,
    FamilyInspection,
    FetchPolicy,
    LoadedModelHandle,
    PortableArtifact,
)
from mlxr.core.schemas import (
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
    file_paths = {record.path for record in source.files}
    if self._checkpoint_filename in file_paths:
        variant = "fast"
    elif self._dev_checkpoint_filename in file_paths:
        variant = "dev"
    else:
        variant = "fast"
    dev_two_stage_ready = (
        variant == "dev"
        and self._spatial_upsampler_filename in file_paths
        and self._distilled_lora_filename in file_paths
    )
    required_roles = (
        ("checkpoint", "spatial_upsampler", "text_encoder")
        if variant == "fast"
        else ("checkpoint", "text_encoder")
    )
    implemented_tasks = (
        [
            "video.generate",
            "video.condition.image",
            "video.condition.video",
            "video.condition.audio",
            "video.retake",
        ]
        if variant == "fast"
        else (
            [
                "video.generate",
                "video.condition.image",
                "video.interpolate",
                "video.retake",
            ]
            if dev_two_stage_ready
            else ["video.generate", "video.condition.image", "video.retake"]
        )
    )
    if variant == "fast":
        pipeline_variants = ["distilled_two_stage"]
    elif dev_two_stage_ready:
        pipeline_variants = ["one_stage", "two_stage", "two_stage_hq"]
    else:
        pipeline_variants = ["one_stage"]
    return FamilyInspection(
        family=self.family_id,
        variant=variant,
        tasks=tuple(implemented_tasks),
        scheduler_class="media_video_dit",
        metadata={
            "source_provider": source.provider,
            "required_source_roles": list(required_roles),
            "role_candidates": role_candidates,
            "bundle_ready": "bundle" in role_candidates,
            "implemented_tasks": implemented_tasks,
            "implemented_pipeline_variants": pipeline_variants,
            "checkpoint_variants": [
                filename.removesuffix(".safetensors")
                for filename in self._checkpoint_filenames
                if filename in file_paths
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
    if role not in (*self._required_roles, *self._optional_roles, "bundle"):
        raise ValueError(f"LTX does not support conversion role '{role}'")
    if role == "checkpoint":
        allow_patterns = (*self._checkpoint_filenames, "*.json")
    elif role == "spatial_upsampler":
        allow_patterns = (self._spatial_upsampler_filename, "*.json")
    elif role == "distilled_lora":
        allow_patterns = (self._distilled_lora_filename, "*.json")
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
            *self._checkpoint_filenames,
            self._spatial_upsampler_filename,
            self._distilled_lora_filename,
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
    checkpoint_variant = _checkpoint_variant(self, prepared["checkpoint"].source_path)
    required_roles = (
        ("checkpoint", "spatial_upsampler", "text_encoder")
        if checkpoint_variant == "fast"
        else ("checkpoint", "text_encoder")
    )
    implemented_tasks = (
        [
            "video.generate",
            "video.condition.image",
            "video.condition.video",
            "video.condition.audio",
            "video.retake",
        ]
        if checkpoint_variant == "fast"
        else (
            [
                "video.generate",
                "video.condition.image",
                "video.interpolate",
                "video.retake",
            ]
            if {"spatial_upsampler", "distilled_lora"} <= set(prepared)
            else ["video.generate", "video.condition.image", "video.retake"]
        )
    )
    if checkpoint_variant == "fast":
        pipeline_variants = ["distilled_two_stage"]
    elif {"spatial_upsampler", "distilled_lora"} <= set(prepared):
        pipeline_variants = ["one_stage", "two_stage", "two_stage_hq"]
    else:
        pipeline_variants = ["one_stage"]
    conditioning = (
        {"image": True, "video": True, "audio": True, "lora": True}
        if checkpoint_variant == "fast"
        else {"image": True, "video": True, "audio": False, "lora": False}
    )
    component_records, payload_items = self._artifact_components(prepared)
    artifact_digest = self._artifact_digest(component_records, plan)
    primary_provenance = prepared["checkpoint"].provenance
    policy = self._combined_policy(prepared)
    dependencies: dict[str, dict[str, object]] = {
        "checkpoint": {
            "required": True,
            "role": "checkpoint",
            "kind": "file",
        },
        "text_encoder": {
            "required": True,
            "role": "text_encoder",
            "kind": "directory",
            "mode": "strict-local",
        },
        "media_encode": {"required": True, "policy": "runtime-managed"},
    }
    if checkpoint_variant == "fast":
        dependencies["spatial_upsampler"] = {
            "required": True,
            "role": "spatial_upsampler",
            "kind": "file",
        }
    else:
        dependencies["spatial_upsampler"] = {
            "required": False,
            "role": "spatial_upsampler",
            "kind": "file",
        }
        dependencies["distilled_lora"] = {
            "required": False,
            "role": "distilled_lora",
            "kind": "file",
        }
    capability = CapabilityDescriptor(
        model_id=plan.model_id,
        artifact_digest=artifact_digest,
        family=self.family_id,
        family_variant=checkpoint_variant,
        tasks=implemented_tasks,
        modalities_in=["text", "image", "audio"],
        modalities_out=["video", "audio"],
        constraints={
            "width": {"multiple_of": 32},
            "height": {"multiple_of": 32},
            "num_frames": {"formula": "8n+1"},
        },
        conditioning=conditioning,
        profiles_by_task={task: ["bf16"] for task in implemented_tasks},
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
        dependencies=dependencies,
        policy=policy,
        extensions_schema=ExtensionSchemaDescriptor(namespace="ltx", version="1"),
        metadata={
            "artifact_layout": "componentized_payload",
            "precision": plan.precision,
            "checkpoint_variant": checkpoint_variant,
            "primary_component_role": "checkpoint",
            "source_count": len(sources),
            "stage_ids": [
                "prompt_encode",
                "condition_inputs",
                "generate",
                "encode_output",
            ],
            "implemented_surface": {
                "tasks": [
                    *implemented_tasks,
                ],
                "pipeline_variants": pipeline_variants,
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
                    "motion_track_control",
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
        family_variant=checkpoint_variant,
        format_version="0.2.0",
        weight_format="source_packaged_fastpath_assets",
        storage_key=f"ltx/{plan.model_id}/{artifact_digest}",
        capability=capability,
        provenance=primary_provenance,
        components=component_records,
        metadata={
            "artifact_layout": "componentized_payload",
            "checkpoint_variant": checkpoint_variant,
            "primary_component_role": "checkpoint",
            "required_source_roles": list(required_roles),
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


def normalize_capability(
    self: LTXFamilyAdapter, artifact: PortableArtifact
) -> CapabilityDescriptor:
    return capabilities(self, artifact)


def _checkpoint_variant(self: LTXFamilyAdapter, checkpoint_path: Path) -> str:
    if checkpoint_path.name == self._checkpoint_filename:
        return "fast"
    if checkpoint_path.name == self._dev_checkpoint_filename:
        return "dev"
    raise ValueError(
        f"LTX checkpoint '{checkpoint_path.name}' does not match a known checkpoint variant"
    )
