from __future__ import annotations

import hashlib
import time
from pathlib import Path

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
    ArtifactHandle,
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
        delay = stage.params.get("simulate_delay_seconds", 0.05)
        if isinstance(delay, (float, int)) and delay > 0:
            time.sleep(float(delay))

        if stage.stage_id == "prompt_encode":
            return StageResult(
                metrics={
                    "stage": stage.stage_id,
                    "status": "scaffold",
                    "tokens_estimate": len(str(stage.inputs.get("prompt", "")).split()),
                }
            )
        if stage.stage_id == "condition_inputs":
            conditioning_items = stage.inputs.get("images", [])
            conditioning_count = (
                len(conditioning_items) if isinstance(conditioning_items, list) else 0
            )
            return StageResult(
                metrics={
                    "stage": stage.stage_id,
                    "status": "scaffold",
                    "conditioning_count": conditioning_count,
                }
            )
        if stage.stage_id == "generate":
            return StageResult(
                metrics={
                    "stage": stage.stage_id,
                    "status": "scaffold",
                    "frames_requested": stage.params.get(
                        "num_frames", stage.inputs.get("num_frames")
                    ),
                }
            )
        if stage.stage_id == "encode_output":
            artifact_id = self._require_str(
                stage.params.get("artifact_id"), "artifact_id"
            )
            artifact_format = self._require_str(
                stage.params.get("artifact_format"), "artifact_format"
            )
            output_dir = self._require_str(stage.params.get("output_dir"), "output_dir")
            storage_key = self._require_str(
                stage.params.get("storage_key"), "storage_key"
            )
            filename = f"{artifact_id}.{artifact_format}"
            output_path = Path(output_dir) / filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            payload = (
                "MLXR scaffold output\n"
                f"model_id={loaded.model_id}\n"
                f"family={loaded.family}\n"
                f"artifact_digest={loaded.artifact_digest}\n"
                f"task={loaded.metadata.get('task')}\n"
                f"prompt={stage.inputs.get('prompt', '')}\n"
                f"format={artifact_format}\n"
            ).encode("utf-8")
            output_path.write_bytes(payload)
            return StageResult(
                artifacts=[
                    ArtifactHandle(
                        artifact_id=artifact_id,
                        artifact_format=artifact_format,
                        metadata={
                            "filename": filename,
                            "media_type": self._media_type_for_format(artifact_format),
                            "size_bytes": output_path.stat().st_size,
                            "storage_key": storage_key,
                        },
                    )
                ],
                metrics={
                    "stage": stage.stage_id,
                    "status": "scaffold",
                    "output_bytes": output_path.stat().st_size,
                },
            )

        raise ValueError(
            f"LTX stage execution is not implemented for unsupported stage '{stage.stage_id}'"
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

    def _media_type_for_format(self, artifact_format: str) -> str:
        if artifact_format == "mp4":
            return "video/mp4"
        if artifact_format == "mov":
            return "video/quicktime"
        if artifact_format == "wav":
            return "audio/wav"
        return "application/octet-stream"

    def _require_str(self, value: object, name: str) -> str:
        if not isinstance(value, str) or not value:
            raise ValueError(f"encode_output requires a non-empty {name}")
        return value
