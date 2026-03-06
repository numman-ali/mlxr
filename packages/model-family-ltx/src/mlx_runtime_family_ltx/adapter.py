from __future__ import annotations

import hashlib

from mlx_runtime_core import (
    ConversionPlan,
    ExecutionProfile,
    ExecutionStage,
    FamilyInspection,
    LoadedModelHandle,
    PortableArtifact,
    SourceMaterialization,
    StageResult,
)
from mlx_runtime_schemas import (
    CapabilityDescriptor,
    ExtensionSchemaDescriptor,
    HardwareTier,
    PolicyDescriptor,
    PortableArtifactRecord,
)


class LTXFamilyAdapter:
    family_id = "ltx"

    def inspect_source(self, source: SourceMaterialization) -> FamilyInspection:
        return FamilyInspection(
            family=self.family_id,
            variant="fast",
            tasks=("video.generate", "video.condition.image"),
            scheduler_class="media_video_dit",
            metadata={
                "status": "scaffold",
                "source_provider": source.resolved.provider,
            },
        )

    def convert(
        self, source: SourceMaterialization, plan: ConversionPlan
    ) -> PortableArtifact:
        artifact_digest = self._artifact_digest(source, plan)
        policy = PolicyDescriptor(
            license=source.provenance.license,
            access_state=source.provenance.access_state,
            remote_code_required=source.provenance.remote_code_required,
            remote_code_approved=source.provenance.remote_code_approved,
        )
        capability = CapabilityDescriptor(
            model_id=plan.model_id,
            artifact_digest=artifact_digest,
            family=self.family_id,
            family_variant="fast",
            tasks=["video.generate", "video.condition.image"],
            modalities_in=["text", "image"],
            modalities_out=["video", "audio"],
            constraints={
                "width": {"multiple_of": 32},
                "height": {"multiple_of": 32},
                "num_frames": {"formula": "8n+1"},
            },
            conditioning={"image": True, "video": False, "audio": False, "lora": True},
            profiles_by_task={
                "video.generate": ["bf16", "q8"],
                "video.condition.image": ["bf16", "q8"],
            },
            streaming={
                "progress_events": True,
                "partial_artifacts": True,
                "segment_events": True,
                "token_deltas": False,
            },
            artifacts_out=["mp4", "mov", "wav"],
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
                "media_encode": {"required": True, "policy": "runtime-managed"}
            },
            policy=policy,
            extensions_schema=ExtensionSchemaDescriptor(namespace="ltx", version="1"),
            metadata={
                "status": "scaffold",
                "precision": plan.precision,
                "materialization_mode": source.materialization_mode,
            },
        )
        record = PortableArtifactRecord(
            model_id=plan.model_id,
            artifact_digest=artifact_digest,
            family=self.family_id,
            family_variant="fast",
            format_version="0.1.0",
            weight_format="mlx_safetensors_sharded",
            storage_key=f"ltx/{plan.model_id}/{artifact_digest}",
            capability=capability,
            provenance=source.provenance,
            metadata={"status": "scaffold", "precision": plan.precision},
        )
        return PortableArtifact(record=record, storage_path=None)

    def load(
        self, artifact: PortableArtifact, profile: ExecutionProfile
    ) -> LoadedModelHandle:
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

    def capabilities(self, artifact: PortableArtifact) -> CapabilityDescriptor:
        return artifact.record.capability

    def run_stage(
        self, loaded: LoadedModelHandle, stage: ExecutionStage
    ) -> StageResult:
        raise NotImplementedError(
            f"LTX stage execution is not implemented yet for '{stage.stage_id}'"
        )

    def unload(self, loaded: LoadedModelHandle) -> None:
        return None

    def _artifact_digest(
        self, source: SourceMaterialization, plan: ConversionPlan
    ) -> str:
        hasher = hashlib.sha256()
        hasher.update((source.provenance.resolved_ref or "unresolved").encode("utf-8"))
        hasher.update(plan.model_id.encode("utf-8"))
        hasher.update(plan.precision.encode("utf-8"))
        return f"sha256:{hasher.hexdigest()}"
