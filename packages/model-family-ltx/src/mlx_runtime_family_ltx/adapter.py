from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

from mlx_runtime_core import (
    ArtifactPayloadItem,
    ConversionPlan,
    ConversionSource,
    ExecutionProfile,
    ExecutionStage,
    FamilyInspection,
    FetchPolicy,
    LoadedModelHandle,
    PortableArtifact,
    StageResult,
)
from mlx_runtime_schemas import (
    ArtifactHandle,
    CapabilityDescriptor,
    ExtensionSchemaDescriptor,
    HardwareTier,
    PolicyDescriptor,
    PortableArtifactComponentRecord,
    PortableArtifactRecord,
    ProvenanceRecord,
    ResolvedSource,
)

from .generation import (
    ConditioningInput,
    GeneratedVideo,
    VideoGenerator,
    create_video_generator,
    encode_mp4_video,
)
from .prompt_encoding import PromptEncoder, PromptEncodingResult, create_prompt_encoder


@dataclass(frozen=True, slots=True)
class PreparedComponent:
    role: str
    kind: str
    source_id: str
    source_path: Path
    provenance: ProvenanceRecord
    resolved_ref: str | None


@dataclass(slots=True)
class LoadedLTXRuntimeState:
    component_paths: dict[str, Path]
    prompt_encoder: PromptEncoder | None = None
    prompt_context: PromptEncodingResult | None = None
    video_generator: VideoGenerator | None = None
    conditioning_inputs: tuple[ConditioningInput, ...] = ()
    generated_video: GeneratedVideo | None = None


class LTXFamilyAdapter:
    family_id = "ltx"
    _checkpoint_filename = "ltx-2.3-22b-distilled.safetensors"
    _spatial_upsampler_filename = "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
    _text_encoder_dirname = "gemma-3-12b-it-qat-q4_0-unquantized"
    _required_roles = ("checkpoint", "spatial_upsampler", "text_encoder")
    _runtime_state_key = "_ltx_runtime_state"

    def inspect_source(self, source: ResolvedSource) -> FamilyInspection:
        role_candidates = self._role_candidates(source)
        return FamilyInspection(
            family=self.family_id,
            variant="fast",
            tasks=("video.generate", "video.condition.image"),
            scheduler_class="media_video_dit",
            metadata={
                "source_provider": source.provider,
                "required_source_roles": list(self._required_roles),
                "role_candidates": role_candidates,
                "bundle_ready": "bundle" in role_candidates,
            },
        )

    def fetch_policy_for_conversion(
        self, role: str, source: ResolvedSource
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
        self, sources: dict[str, ConversionSource], plan: ConversionPlan
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
            tasks=["video.generate", "video.condition.image"],
            modalities_in=["text", "image"],
            modalities_out=["video"],
            constraints={
                "width": {"multiple_of": 32},
                "height": {"multiple_of": 32},
                "num_frames": {"formula": "8n+1"},
            },
            conditioning={"image": True, "video": False, "audio": False, "lora": False},
            profiles_by_task={
                "video.generate": ["bf16"],
                "video.condition.image": ["bf16"],
            },
            streaming={
                "progress_events": True,
                "partial_artifacts": False,
                "segment_events": False,
                "token_deltas": False,
            },
            artifacts_out=["mp4"],
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
        self, artifact: PortableArtifact, profile: ExecutionProfile
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

    def capabilities(self, artifact: PortableArtifact) -> CapabilityDescriptor:
        return artifact.record.capability

    def run_stage(
        self, loaded: LoadedModelHandle, stage: ExecutionStage
    ) -> StageResult:
        delay = stage.params.get("simulate_delay_seconds", 0.05)
        if isinstance(delay, (float, int)) and delay > 0:
            time.sleep(float(delay))

        runtime_state = self._runtime_state(loaded)
        if stage.stage_id == "prompt_encode":
            if runtime_state is None:
                return StageResult(
                    metrics={
                        "stage": stage.stage_id,
                        "status": "placeholder",
                        "tokens_estimate": len(
                            str(stage.inputs.get("prompt", "")).split()
                        ),
                    }
                )
            prompt = self._prompt_text(stage.inputs)
            self._reject_unsupported_negative_prompt(stage.inputs)
            prompt_encoder = self._prompt_encoder(runtime_state)
            prompt_context = prompt_encoder.encode(
                prompt,
                max_length=1024,
                return_audio_context=True,
            )
            runtime_state.prompt_context = prompt_context
            return StageResult(
                metrics={
                    "stage": stage.stage_id,
                    "status": "encoded",
                    "prompt_chars": len(prompt),
                    "token_count": prompt_context.token_count,
                    "sequence_length": prompt_context.sequence_length,
                    "video_context_shape": list(prompt_context.video_context_shape),
                    "audio_context_shape": (
                        list(prompt_context.audio_context_shape)
                        if prompt_context.audio_context_shape is not None
                        else None
                    ),
                    "attention_mask_shape": list(prompt_context.attention_mask_shape),
                    "audio_context_available": prompt_context.audio_context is not None,
                }
            )
        if stage.stage_id == "condition_inputs":
            prepared_inputs = self._prepared_conditioning_inputs(stage)
            if runtime_state is None:
                return StageResult(
                    metrics={
                        "stage": stage.stage_id,
                        "status": "placeholder",
                        "conditioning_count": len(prepared_inputs),
                    }
                )
            runtime_state.conditioning_inputs = prepared_inputs
            task = self._stage_task(stage)
            if task == "video.condition.image" and not prepared_inputs:
                raise ValueError(
                    "video.condition.image requires at least one resolved conditioning image"
                )
            return StageResult(
                metrics={
                    "stage": stage.stage_id,
                    "status": "prepared" if prepared_inputs else "skipped",
                    "conditioning_count": len(prepared_inputs),
                    "frame_indices": [item.frame_index for item in prepared_inputs],
                }
            )
        if stage.stage_id == "generate":
            if runtime_state is None:
                return StageResult(
                    metrics={
                        "stage": stage.stage_id,
                        "status": "placeholder",
                        "frames_requested": stage.params.get(
                            "num_frames", stage.inputs.get("num_frames")
                        ),
                    }
                )
            if runtime_state.prompt_context is None:
                raise ValueError(
                    "LTX generate stage requires prompt_encode to run successfully first"
                )
            generator = self._video_generator(runtime_state)
            generated_video = generator.generate(
                prompt_context=runtime_state.prompt_context,
                conditioning_inputs=runtime_state.conditioning_inputs,
                width=self._stage_dimension(stage.params.get("width"), name="width"),
                height=self._stage_dimension(stage.params.get("height"), name="height"),
                num_frames=self._num_frames(stage),
                fps=self._fps(stage),
                seed=self._seed(stage),
            )
            runtime_state.generated_video = generated_video
            return StageResult(
                metrics={
                    "stage": stage.stage_id,
                    "status": "generated",
                    "frames_generated": int(generated_video.frames.shape[0]),
                    "width": int(generated_video.frames.shape[2]),
                    "height": int(generated_video.frames.shape[1]),
                    "fps": generated_video.fps,
                    "seed": generated_video.seed,
                    "backend": generated_video.backend,
                    "conditioning_count": generated_video.conditioning_count,
                    "prompt_signature": generated_video.prompt_signature,
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
            if artifact_format != "mp4":
                raise ValueError(
                    f"LTX encode_output only supports runtime-managed mp4 artifacts, got '{artifact_format}'"
                )
            if runtime_state is None or runtime_state.generated_video is None:
                raise ValueError(
                    "LTX encode_output requires generate to run successfully first"
                )
            encode_mp4_video(runtime_state.generated_video, output_path)
            generated_video = runtime_state.generated_video
            runtime_state.generated_video = None
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
                    "status": "encoded",
                    "output_bytes": output_path.stat().st_size,
                    "frames_encoded": int(generated_video.frames.shape[0]),
                    "fps": generated_video.fps,
                    "backend": generated_video.backend,
                },
            )

        raise ValueError(
            f"LTX stage execution is not implemented for unsupported stage '{stage.stage_id}'"
        )

    def unload(self, loaded: LoadedModelHandle) -> None:
        runtime_state = self._runtime_state(loaded)
        if runtime_state is not None:
            if runtime_state.prompt_encoder is not None:
                runtime_state.prompt_encoder.close()
                runtime_state.prompt_encoder = None
            if runtime_state.video_generator is not None:
                runtime_state.video_generator.close()
                runtime_state.video_generator = None
            runtime_state.prompt_context = None
            runtime_state.conditioning_inputs = ()
            runtime_state.generated_video = None
            loaded.metadata.pop(self._runtime_state_key, None)
        return None

    def _runtime_state(self, loaded: LoadedModelHandle) -> LoadedLTXRuntimeState | None:
        state = loaded.metadata.get(self._runtime_state_key)
        if isinstance(state, LoadedLTXRuntimeState):
            return state
        return None

    def _prompt_encoder(self, runtime_state: LoadedLTXRuntimeState) -> PromptEncoder:
        if runtime_state.prompt_encoder is None:
            runtime_state.prompt_encoder = create_prompt_encoder(
                checkpoint_path=runtime_state.component_paths["checkpoint"],
                text_encoder_path=runtime_state.component_paths["text_encoder"],
            )
        return runtime_state.prompt_encoder

    def _video_generator(self, runtime_state: LoadedLTXRuntimeState) -> VideoGenerator:
        if runtime_state.video_generator is None:
            runtime_state.video_generator = create_video_generator(
                checkpoint_path=runtime_state.component_paths["checkpoint"],
                spatial_upsampler_path=runtime_state.component_paths[
                    "spatial_upsampler"
                ],
            )
        return runtime_state.video_generator

    def _prompt_text(self, inputs: dict[str, object]) -> str:
        prompt = inputs.get("prompt", "")
        if not isinstance(prompt, str):
            raise ValueError("LTX prompt_encode expects inputs.prompt to be a string")
        return prompt

    def _prepared_conditioning_inputs(
        self, stage: ExecutionStage
    ) -> tuple[ConditioningInput, ...]:
        resolved_inputs = stage.params.get("resolved_inputs")
        if resolved_inputs is None:
            return ()
        if not isinstance(resolved_inputs, dict):
            raise ValueError("LTX stage params.resolved_inputs must be an object")
        raw_images = resolved_inputs.get("images", [])
        if not isinstance(raw_images, list):
            raise ValueError("LTX resolved conditioning inputs must be a list")

        num_frames = self._num_frames(stage)
        prepared: list[ConditioningInput] = []
        for entry in raw_images:
            if not isinstance(entry, dict):
                raise ValueError(
                    "LTX resolved conditioning image entries must be objects"
                )
            handle_id = self._require_str(entry.get("input_handle"), "input_handle")
            payload_path = Path(
                self._require_str(entry.get("payload_path"), "payload_path")
            )
            if not payload_path.is_file():
                raise ValueError(
                    f"LTX resolved conditioning payload '{payload_path}' does not exist"
                )
            frame_index = entry.get("frame_index", 0)
            if not isinstance(frame_index, int) or frame_index < 0:
                raise ValueError(
                    "LTX conditioning frame_index must be a non-negative integer"
                )
            if frame_index >= num_frames:
                raise ValueError(
                    f"LTX conditioning frame_index {frame_index} is outside num_frames={num_frames}"
                )
            strength = entry.get("strength", 1.0)
            if not isinstance(strength, (int, float)):
                raise ValueError("LTX conditioning strength must be numeric")
            strength_value = float(strength)
            if not 0.0 <= strength_value <= 1.0:
                raise ValueError(
                    "LTX conditioning strength must be between 0.0 and 1.0"
                )
            media_type = entry.get("media_type")
            filename = entry.get("filename")
            prepared.append(
                ConditioningInput(
                    handle_id=handle_id,
                    payload_path=payload_path,
                    frame_index=frame_index,
                    strength=strength_value,
                    media_type=media_type if isinstance(media_type, str) else None,
                    filename=filename if isinstance(filename, str) else None,
                )
            )
        return tuple(prepared)

    def _stage_dimension(self, value: object, *, name: str) -> int:
        if value is None:
            return 768 if name == "width" else 512
        if not isinstance(value, int) or value < 32:
            raise ValueError(f"LTX {name} must be an integer >= 32")
        return value

    def _num_frames(self, stage: ExecutionStage) -> int:
        value = stage.params.get("num_frames", stage.inputs.get("num_frames"))
        if value is None:
            return 121
        if not isinstance(value, int) or value < 1:
            raise ValueError("LTX num_frames must be a positive integer")
        return value

    def _fps(self, stage: ExecutionStage) -> int:
        value = stage.params.get("fps", stage.inputs.get("fps"))
        if value is None:
            return 24
        if not isinstance(value, int) or value < 1:
            raise ValueError("LTX fps must be a positive integer")
        return value

    def _seed(self, stage: ExecutionStage) -> int | None:
        value = stage.params.get("seed", stage.inputs.get("seed"))
        if value is None:
            return None
        if not isinstance(value, int):
            raise ValueError("LTX seed must be an integer when provided")
        return value

    def _stage_task(self, stage: ExecutionStage) -> str:
        value = stage.params.get("task")
        if isinstance(value, str) and value:
            return value
        return "video.generate"

    def _reject_unsupported_negative_prompt(self, inputs: dict[str, object]) -> None:
        negative_prompt = inputs.get("negative_prompt")
        if negative_prompt in {None, ""}:
            return
        if not isinstance(negative_prompt, str):
            raise ValueError(
                "LTX prompt_encode expects inputs.negative_prompt to be a string when provided"
            )
        if negative_prompt.strip():
            raise ValueError(
                "LTX fast-path prompt encoding does not support negative_prompt yet"
            )

    def _role_candidates(self, source: ResolvedSource) -> list[str]:
        file_paths = {record.path for record in source.files}
        candidates: list[str] = []
        if self._checkpoint_filename in file_paths:
            candidates.append("checkpoint")
        if self._spatial_upsampler_filename in file_paths:
            candidates.append("spatial_upsampler")
        if self._looks_like_text_encoder_snapshot(file_paths):
            candidates.append("text_encoder")
        if (
            self._checkpoint_filename in file_paths
            and self._spatial_upsampler_filename in file_paths
            and any(
                path.startswith(f"{self._text_encoder_dirname}/")
                or path == self._text_encoder_dirname
                for path in file_paths
            )
        ):
            candidates.append("bundle")
        return candidates

    def _prepare_components(
        self, sources: dict[str, ConversionSource]
    ) -> dict[str, PreparedComponent]:
        if set(sources) == {"bundle"}:
            return self._prepare_bundle_components(sources["bundle"])

        unexpected_roles = sorted(
            role for role in sources if role not in self._required_roles
        )
        if unexpected_roles:
            raise ValueError(
                f"LTX conversion does not support source roles: {', '.join(unexpected_roles)}"
            )
        missing_roles = [role for role in self._required_roles if role not in sources]
        if missing_roles:
            raise ValueError(
                f"LTX conversion requires source roles: {', '.join(missing_roles)}"
            )

        checkpoint_source = sources["checkpoint"]
        upsampler_source = sources["spatial_upsampler"]
        text_encoder_source = sources["text_encoder"]
        return {
            "checkpoint": PreparedComponent(
                role="checkpoint",
                kind="file",
                source_id=checkpoint_source.source_id,
                source_path=self._resolve_required_file(
                    checkpoint_source.materialization.local_path,
                    self._checkpoint_filename,
                    role="checkpoint",
                ),
                provenance=checkpoint_source.materialization.provenance,
                resolved_ref=checkpoint_source.materialization.provenance.resolved_ref,
            ),
            "spatial_upsampler": PreparedComponent(
                role="spatial_upsampler",
                kind="file",
                source_id=upsampler_source.source_id,
                source_path=self._resolve_required_file(
                    upsampler_source.materialization.local_path,
                    self._spatial_upsampler_filename,
                    role="spatial_upsampler",
                ),
                provenance=upsampler_source.materialization.provenance,
                resolved_ref=upsampler_source.materialization.provenance.resolved_ref,
            ),
            "text_encoder": PreparedComponent(
                role="text_encoder",
                kind="directory",
                source_id=text_encoder_source.source_id,
                source_path=self._resolve_text_encoder_dir(
                    text_encoder_source.materialization.local_path,
                    role="text_encoder",
                ),
                provenance=text_encoder_source.materialization.provenance,
                resolved_ref=text_encoder_source.materialization.provenance.resolved_ref,
            ),
        }

    def _prepare_bundle_components(
        self, bundle_source: ConversionSource
    ) -> dict[str, PreparedComponent]:
        bundle_root = bundle_source.materialization.local_path
        if bundle_root is None or not bundle_root.is_dir():
            raise ValueError(
                "LTX bundle conversion requires a directory source containing the fast-path assets"
            )
        text_encoder_root = bundle_root / self._text_encoder_dirname
        if not text_encoder_root.exists():
            raise ValueError(
                f"LTX bundle source is missing '{self._text_encoder_dirname}'"
            )
        self._validate_text_encoder_dir(text_encoder_root, role="bundle")
        return {
            "checkpoint": PreparedComponent(
                role="checkpoint",
                kind="file",
                source_id=bundle_source.source_id,
                source_path=self._resolve_required_file(
                    bundle_root, self._checkpoint_filename, role="bundle"
                ),
                provenance=bundle_source.materialization.provenance,
                resolved_ref=bundle_source.materialization.provenance.resolved_ref,
            ),
            "spatial_upsampler": PreparedComponent(
                role="spatial_upsampler",
                kind="file",
                source_id=bundle_source.source_id,
                source_path=self._resolve_required_file(
                    bundle_root, self._spatial_upsampler_filename, role="bundle"
                ),
                provenance=bundle_source.materialization.provenance,
                resolved_ref=bundle_source.materialization.provenance.resolved_ref,
            ),
            "text_encoder": PreparedComponent(
                role="text_encoder",
                kind="directory",
                source_id=bundle_source.source_id,
                source_path=text_encoder_root,
                provenance=bundle_source.materialization.provenance,
                resolved_ref=bundle_source.materialization.provenance.resolved_ref,
            ),
        }

    def _artifact_components(
        self, prepared: dict[str, PreparedComponent]
    ) -> tuple[list[PortableArtifactComponentRecord], list[ArtifactPayloadItem]]:
        components: list[PortableArtifactComponentRecord] = []
        payload_items: list[ArtifactPayloadItem] = []
        for role in self._required_roles:
            component = prepared[role]
            if component.kind == "file":
                relative_path = Path("payload") / role / component.source_path.name
                payload_items.append(
                    ArtifactPayloadItem(
                        source_path=component.source_path,
                        relative_path=relative_path,
                    )
                )
                components.append(
                    PortableArtifactComponentRecord(
                        role=role,
                        kind="file",
                        relative_path=relative_path.as_posix(),
                        source_id=component.source_id,
                        resolved_ref=component.resolved_ref,
                        size_bytes=component.source_path.stat().st_size,
                        component_digest=self._file_digest(component.source_path),
                        provenance=component.provenance,
                        metadata={"filename": component.source_path.name},
                    )
                )
                continue

            relative_root = Path("payload") / role
            for file_path in self._directory_files(component.source_path):
                payload_items.append(
                    ArtifactPayloadItem(
                        source_path=file_path,
                        relative_path=relative_root
                        / file_path.relative_to(component.source_path),
                    )
                )
            components.append(
                PortableArtifactComponentRecord(
                    role=role,
                    kind="directory",
                    relative_path=relative_root.as_posix(),
                    source_id=component.source_id,
                    resolved_ref=component.resolved_ref,
                    size_bytes=self._directory_size(component.source_path),
                    component_digest=self._directory_digest(component.source_path),
                    provenance=component.provenance,
                    metadata={"dirname": component.source_path.name},
                )
            )
        return components, payload_items

    def _artifact_digest(
        self,
        components: list[PortableArtifactComponentRecord],
        plan: ConversionPlan,
    ) -> str:
        payload = {
            "family": self.family_id,
            "model_id": plan.model_id,
            "precision": plan.precision,
            "target_format": plan.target_format,
            "components": [
                {
                    "role": component.role,
                    "kind": component.kind,
                    "relative_path": component.relative_path,
                    "component_digest": component.component_digest,
                }
                for component in components
            ],
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return f"sha256:{digest}"

    def _combined_policy(
        self, prepared: dict[str, PreparedComponent]
    ) -> PolicyDescriptor:
        provenances = [component.provenance for component in prepared.values()]
        remote_code_required = any(
            provenance.remote_code_required for provenance in provenances
        )
        remote_code_approved = remote_code_required and all(
            (not provenance.remote_code_required or provenance.remote_code_approved)
            for provenance in provenances
        )
        return PolicyDescriptor(
            license=prepared["checkpoint"].provenance.license,
            access_state=prepared["checkpoint"].provenance.access_state,
            remote_code_required=remote_code_required,
            remote_code_approved=remote_code_approved,
        )

    def _component_paths(
        self,
        artifact_root: Path,
        components: list[PortableArtifactComponentRecord],
    ) -> dict[str, Path]:
        component_paths: dict[str, Path] = {}
        for component in components:
            component_path = artifact_root / component.relative_path
            if component.kind == "file":
                if not component_path.is_file():
                    raise ValueError(
                        f"LTX artifact is missing required file component '{component.role}'"
                    )
            elif component.kind == "directory":
                if not component_path.is_dir():
                    raise ValueError(
                        f"LTX artifact is missing required directory component '{component.role}'"
                    )
            component_paths[component.role] = component_path
        missing_roles = [
            role for role in self._required_roles if role not in component_paths
        ]
        if missing_roles:
            raise ValueError(
                f"LTX artifact is missing required component roles: {', '.join(missing_roles)}"
            )
        self._validate_text_encoder_dir(
            component_paths["text_encoder"], role="artifact"
        )
        return component_paths

    def _resolve_required_file(
        self, local_path: Path | None, filename: str, *, role: str
    ) -> Path:
        if local_path is None:
            raise ValueError(
                f"LTX {role} source is missing a local materialization path"
            )
        if local_path.is_file():
            if local_path.name != filename:
                raise ValueError(
                    f"LTX {role} source must point to '{filename}', got '{local_path.name}'"
                )
            return local_path
        candidate = local_path / filename
        if candidate.is_file():
            return candidate
        raise ValueError(f"LTX {role} source is missing '{filename}'")

    def _resolve_text_encoder_dir(self, local_path: Path | None, *, role: str) -> Path:
        if local_path is None:
            raise ValueError(
                f"LTX {role} source is missing a local text-encoder directory"
            )
        if local_path.is_file():
            raise ValueError(f"LTX {role} source must be a directory")

        candidates = [local_path]
        named_child = local_path / self._text_encoder_dirname
        if named_child.is_dir():
            candidates.insert(0, named_child)

        for candidate in candidates:
            try:
                self._validate_text_encoder_dir(candidate, role=role)
            except ValueError:
                continue
            return candidate
        raise ValueError(
            f"LTX {role} source is missing a valid Gemma text-encoder directory"
        )

    def _validate_text_encoder_dir(self, text_encoder_root: Path, *, role: str) -> None:
        if not text_encoder_root.is_dir():
            raise ValueError(f"LTX {role} text encoder source must be a directory")
        config_path = text_encoder_root / "config.json"
        if not config_path.is_file():
            raise ValueError(f"LTX {role} text encoder source is missing 'config.json'")
        tokenizer_candidates = (
            text_encoder_root / "tokenizer.json",
            text_encoder_root / "tokenizer.model",
            text_encoder_root / "tokenizer_config.json",
        )
        if not any(candidate.is_file() for candidate in tokenizer_candidates):
            raise ValueError("LTX text encoder source must include tokenizer metadata")
        if not any(
            file_path.suffix == ".safetensors"
            for file_path in self._directory_files(text_encoder_root)
        ):
            raise ValueError(
                "LTX text encoder source must include at least one safetensors weight file"
            )

    def _looks_like_text_encoder_snapshot(self, file_paths: set[str]) -> bool:
        has_config = "config.json" in file_paths
        has_tokenizer = any(
            name in file_paths
            for name in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json")
        )
        has_weights = any(path.endswith(".safetensors") for path in file_paths)
        return has_config and has_tokenizer and has_weights

    def _directory_files(self, root: Path) -> tuple[Path, ...]:
        return tuple(sorted(path for path in root.rglob("*") if path.is_file()))

    def _directory_size(self, root: Path) -> int:
        return sum(path.stat().st_size for path in self._directory_files(root))

    def _file_digest(self, source_path: Path) -> str:
        hasher = hashlib.sha256()
        hasher.update(source_path.read_bytes())
        return f"sha256:{hasher.hexdigest()}"

    def _directory_digest(self, root: Path) -> str:
        hasher = hashlib.sha256()
        for file_path in self._directory_files(root):
            relative_path = file_path.relative_to(root).as_posix()
            hasher.update(relative_path.encode("utf-8"))
            hasher.update(file_path.read_bytes())
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
            raise ValueError(f"LTX stage data requires a non-empty {name}")
        return value
