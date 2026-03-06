from __future__ import annotations

from pathlib import Path

from mlx_runtime_core import ConversionProfile, LoadedHandle, ModelArtifact, ResolvedSource, RuntimeProfile, SourceInspection, SourceRef
from mlx_runtime_schemas import CapabilityDescriptor, JobRecord, RuntimeEvent


class LTXFamilyAdapter:
    family_id = "ltx"

    def resolve_source(self, ref: SourceRef) -> ResolvedSource:
        return ResolvedSource(ref=ref, local_path=Path(ref.uri))

    def inspect_source(self, source: ResolvedSource) -> SourceInspection:
        return SourceInspection(
            family=self.family_id,
            variant="ltx-2.3-fast",
            tasks=("video.generate",),
            metadata={"status": "scaffold"},
        )

    def convert(self, source: ResolvedSource, profile: ConversionProfile) -> ModelArtifact:
        capability = self.capabilities()
        return ModelArtifact(
            artifact_path=source.local_path,
            format_version="0.1.0",
            weight_format="mlx_safetensors_sharded",
            capability=capability,
            metadata={"status": "scaffold", "precision": profile.precision},
        )

    def load(self, artifact: ModelArtifact, runtime: RuntimeProfile) -> LoadedHandle:
        return LoadedHandle(
            model_id="ltx-2.3-fast-local",
            family=self.family_id,
            capability=artifact.capability,
            metadata={"status": "scaffold", "execution_mode": runtime.execution_mode},
        )

    def capabilities(self, handle: LoadedHandle | None = None) -> CapabilityDescriptor:
        return CapabilityDescriptor(
            model_id="ltx-2.3-fast-local",
            family=self.family_id,
            tasks=["video.generate"],
            modalities_in=["text", "image"],
            modalities_out=["video", "audio"],
            conditioning={"image": True, "video": False, "audio": False, "lora": True},
            profiles=["bf16", "q8", "q6", "q4"],
            streaming={"progress_events": True, "partial_artifacts": True},
            metadata={"status": "scaffold"},
        )

    def execute(self, handle: LoadedHandle, job: JobRecord) -> list[RuntimeEvent]:
        raise NotImplementedError("LTX MLX execution is not implemented yet")

    def unload(self, handle: LoadedHandle) -> None:
        return None
