from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path

from mlxr.core.runtime import (
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
from mlxr.core.schemas import (
    ArtifactHandle,
    CapabilityDescriptor,
    HardwareTier,
    PolicyDescriptor,
    PortableArtifactComponentRecord,
    PortableArtifactRecord,
    ProvenanceRecord,
    ResolvedSource,
)

from .family_options import (
    cfg_normalization_from_extensions,
    cfg_truncation_from_extensions,
    max_sequence_length_from_extensions,
)
from .generation import (
    GeneratedImage,
    ImageGenerator,
    create_image_generator,
    encode_jpg_image,
    encode_png_image,
)
from .prompt_encoding import PromptEncoder, PromptEncodingResult, create_prompt_encoder

_REQUIRED_COMPONENTS = (
    "transformer",
    "vae",
    "text_encoder",
    "tokenizer",
    "scheduler",
)
_RELEASED_VARIANTS = ("z-image", "z-image-turbo")
_BLOCKED_VARIANTS = ("z-image-omni-base", "z-image-edit")
_OUTPUT_FORMATS = ("png", "jpg")
_SCHEDULER_CLASS = "image_diffusion"
_WEIGHT_FORMAT = "diffusers_component_bundle"
_FORMAT_VERSION = "0.1.0"
_NATIVE_RUNTIME_STATUS = "experimental_native_mlx_backend"


class ZImageFamilyAdapter:
    family_id = "z_image"
    _required_roles = ("bundle",)
    _required_components = _REQUIRED_COMPONENTS

    def inspect_source(self, source: ResolvedSource) -> FamilyInspection:
        file_paths = {record.path for record in source.files}
        missing = [
            component
            for component in self._required_components
            if not _has_component_files(file_paths, component)
        ]
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(
                "Z-Image source inspection requires a diffusers-style bundle with "
                f"{missing_text} component directories"
            )
        variant = _variant_from_source(source)
        return FamilyInspection(
            family=self.family_id,
            variant=variant,
            tasks=("image.generate",),
            scheduler_class=_SCHEDULER_CLASS,
            metadata={
                "stage_ids": ["prompt_encode", "generate", "encode_output"],
                "required_components": list(self._required_components),
                "released_variants": list(_RELEASED_VARIANTS),
                "blocked_variants": list(_BLOCKED_VARIANTS),
                "upstream_remote_code_pressure": True,
                "native_runtime_status": _NATIVE_RUNTIME_STATUS,
            },
        )

    def fetch_policy_for_conversion(
        self, role: str, source: ResolvedSource
    ) -> FetchPolicy:
        del source
        if role != "bundle":
            raise ValueError(
                "Z-Image conversion currently requires a single 'bundle' source role"
            )
        allow_patterns = ["model_index.json"]
        for component in self._required_components:
            allow_patterns.append(f"{component}/**")
        return FetchPolicy(
            allow_patterns=tuple(allow_patterns),
            eager=False,
            options={"source_contract": "diffusers_component_bundle"},
        )

    def convert(
        self, sources: dict[str, ConversionSource], plan: ConversionPlan
    ) -> PortableArtifact:
        bundle_source = _require_bundle_source(sources)
        bundle_root = bundle_source.materialization.local_path
        if bundle_root is None or not bundle_root.is_dir():
            raise ValueError(
                "Z-Image conversion requires a materialized local bundle directory"
            )

        component_files = _collect_component_files(
            bundle_root, self._required_components
        )
        artifact_digest = _artifact_digest(
            model_id=plan.model_id,
            precision=plan.precision,
            variant=_variant_from_path(bundle_root),
            component_files=component_files,
        )
        capability = _capability_descriptor(
            model_id=plan.model_id,
            artifact_digest=artifact_digest,
            variant=_variant_from_path(bundle_root),
            provenance=bundle_source.materialization.provenance,
        )
        components: list[PortableArtifactComponentRecord] = []
        payload_items: list[ArtifactPayloadItem] = []
        for component, files in component_files.items():
            components.append(
                PortableArtifactComponentRecord(
                    role=component,
                    kind="directory",
                    relative_path=f"payload/{component}",
                    source_id=bundle_source.source_id,
                    resolved_ref=bundle_source.materialization.resolved.pinned_ref,
                    size_bytes=sum(file_path.stat().st_size for file_path in files),
                    component_digest=_component_digest(bundle_root, files),
                    provenance=bundle_source.materialization.provenance,
                    metadata={"file_count": len(files)},
                )
            )
            for file_path in files:
                payload_items.append(
                    ArtifactPayloadItem(
                        source_path=file_path,
                        relative_path=Path("payload")
                        / file_path.relative_to(bundle_root),
                    )
                )

        record = PortableArtifactRecord(
            model_id=plan.model_id,
            artifact_digest=artifact_digest,
            family=self.family_id,
            family_variant=_variant_from_path(bundle_root),
            format_version=_FORMAT_VERSION,
            weight_format=_WEIGHT_FORMAT,
            storage_key="",
            capability=capability,
            provenance=bundle_source.materialization.provenance,
            components=components,
            metadata={
                "required_components": list(self._required_components),
                "released_variants": list(_RELEASED_VARIANTS),
                "blocked_variants": list(_BLOCKED_VARIANTS),
                "native_runtime_status": _NATIVE_RUNTIME_STATUS,
            },
        )
        return PortableArtifact(record=record, payload_items=tuple(payload_items))

    def load(
        self, artifact: PortableArtifact, profile: ExecutionProfile
    ) -> LoadedModelHandle:
        if profile.task != "image.generate":
            raise ValueError(
                f"Z-Image load only supports 'image.generate', got '{profile.task}'"
            )
        if artifact.storage_path is None:
            raise ValueError(
                "Z-Image load requires a materialized artifact storage path"
            )

        component_paths: dict[str, str] = {}
        for component in self._required_components:
            component_root = artifact.storage_path / "payload" / component
            if not component_root.is_dir():
                raise ValueError(
                    "Z-Image artifact is missing required component directory "
                    f"'{component_root}'"
                )
            component_paths[component] = str(component_root)

        return LoadedModelHandle(
            model_id=artifact.record.model_id,
            family=self.family_id,
            artifact_digest=artifact.record.artifact_digest,
            capability=self.capabilities(artifact),
            metadata={
                "component_paths": component_paths,
                "requested_profile": profile.profile,
                "execution_status": _NATIVE_RUNTIME_STATUS,
            },
        )

    def capabilities(self, artifact: PortableArtifact) -> CapabilityDescriptor:
        return artifact.record.capability.model_copy(
            update={
                "model_id": artifact.record.model_id,
                "artifact_digest": artifact.record.artifact_digest,
                "family_variant": artifact.record.family_variant,
            }
        )

    def run_stage(
        self, loaded: LoadedModelHandle, stage: ExecutionStage
    ) -> StageResult:
        if stage.stage_id == "prompt_encode":
            prompt_encoder = _prompt_encoder(loaded)
            prompt = _require_prompt(stage.inputs)
            negative_prompt = _optional_text(stage.inputs.get("negative_prompt"))
            prompt_context = prompt_encoder.encode(
                prompt,
                max_length=_max_sequence_length(stage),
                negative_prompt=negative_prompt,
            )
            loaded.metadata["prompt_context"] = prompt_context
            loaded.metadata["execution_status"] = "prompt_encode_ready"
            return StageResult(
                metrics={
                    "stage": stage.stage_id,
                    "status": "encoded",
                    "token_count": prompt_context.token_count,
                    "sequence_length": prompt_context.sequence_length,
                    "hidden_size": prompt_context.hidden_size,
                    "negative_prompt_present": (
                        prompt_context.negative_prompt_embeddings is not None
                    ),
                    "config_source": prompt_context.config_source,
                }
            )
        if stage.stage_id == "generate":
            prompt_context_value = _prompt_context(loaded)
            if prompt_context_value is None:
                raise ValueError(
                    "Z-Image generate stage requires prompt_encode to run successfully first"
                )
            prompt_context = prompt_context_value
            image_generator = _image_generator(loaded)
            generated_image = image_generator.generate(
                prompt_context=prompt_context,
                width=_stage_dimension(stage.params.get("width"), name="width"),
                height=_stage_dimension(stage.params.get("height"), name="height"),
                num_inference_steps=_stage_positive_int(
                    stage.params.get("num_inference_steps"),
                    name="num_inference_steps",
                    default=8,
                ),
                guidance_scale=_stage_non_negative_float(
                    stage.params.get("guidance_scale"),
                    name="guidance_scale",
                    default=0.0,
                ),
                cfg_normalization=_cfg_normalization(stage),
                cfg_truncation=_cfg_truncation(stage),
                seed=_stage_seed(stage.params.get("seed")),
            )
            loaded.metadata["generated_image"] = generated_image
            loaded.metadata["execution_status"] = "generated_image_ready"
            return StageResult(
                metrics={
                    "stage": stage.stage_id,
                    "status": "generated",
                    "width": int(generated_image.pixels.shape[1]),
                    "height": int(generated_image.pixels.shape[0]),
                    "seed": generated_image.seed,
                    "backend": generated_image.backend,
                    "prompt_signature": generated_image.prompt_signature,
                    **generated_image.metadata,
                }
            )
        if stage.stage_id == "encode_output":
            generated_image_value = _generated_image(loaded)
            if generated_image_value is None:
                raise ValueError(
                    "Z-Image encode_output requires generate to run successfully first"
                )
            generated_image = generated_image_value
            artifact_id = str(stage.params.get("artifact_id", ""))
            artifact_format = str(stage.params.get("artifact_format", ""))
            output_dir = _require_output_dir(stage.params.get("output_dir"))
            storage_key = _require_storage_key(stage.params.get("storage_key"))
            filename = f"{artifact_id}.{artifact_format}"
            output_path = output_dir / filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if artifact_format == "png":
                encode_png_image(generated_image, output_path)
            elif artifact_format == "jpg":
                encode_jpg_image(generated_image, output_path)
            else:
                raise ValueError(
                    "Z-Image encode_output only supports runtime-managed png or jpg "
                    f"artifacts, got '{artifact_format}'"
                )
            loaded.metadata.pop("generated_image", None)
            return StageResult(
                artifacts=[
                    ArtifactHandle(
                        artifact_id=artifact_id,
                        artifact_format=artifact_format,
                        metadata={
                            "filename": filename,
                            "media_type": _media_type_for_format(artifact_format),
                            "size_bytes": output_path.stat().st_size,
                            "storage_key": storage_key,
                            "width": int(generated_image.pixels.shape[1]),
                            "height": int(generated_image.pixels.shape[0]),
                        },
                    )
                ],
                metrics={
                    "stage": stage.stage_id,
                    "status": "encoded",
                    "output_bytes": output_path.stat().st_size,
                    "artifact_format": artifact_format,
                    "width": int(generated_image.pixels.shape[1]),
                    "height": int(generated_image.pixels.shape[0]),
                    "backend": generated_image.backend,
                },
            )
        raise ValueError(
            f"Z-Image stage execution is not implemented for unsupported stage '{stage.stage_id}'"
        )

    def unload(self, loaded: LoadedModelHandle) -> None:
        prompt_encoder = loaded.metadata.pop("prompt_encoder", None)
        if hasattr(prompt_encoder, "close"):
            prompt_encoder.close()
        image_generator = loaded.metadata.pop("image_generator", None)
        if hasattr(image_generator, "close"):
            image_generator.close()
        loaded.metadata.pop("prompt_context", None)
        loaded.metadata.pop("generated_image", None)
        return None


def _require_prompt(inputs: dict[str, object]) -> str:
    prompt = inputs.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Z-Image prompt_encode requires a non-empty prompt")
    return prompt


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("Z-Image prompt inputs must use string prompt fields")
    normalized = value.strip()
    return normalized or None


def _prompt_context(loaded: LoadedModelHandle) -> PromptEncodingResult | None:
    context = loaded.metadata.get("prompt_context")
    return context if isinstance(context, PromptEncodingResult) else None


def _prompt_encoder(loaded: LoadedModelHandle) -> PromptEncoder:
    existing = loaded.metadata.get("prompt_encoder")
    if isinstance(existing, PromptEncoder):
        return existing
    component_paths = loaded.metadata.get("component_paths")
    if not isinstance(component_paths, dict):
        raise ValueError("Z-Image runtime state is missing component_paths")
    text_encoder_path = component_paths.get("text_encoder")
    tokenizer_path = component_paths.get("tokenizer")
    if not isinstance(text_encoder_path, str) or not isinstance(tokenizer_path, str):
        raise ValueError(
            "Z-Image runtime state is missing text_encoder/tokenizer component paths"
        )
    prompt_encoder = create_prompt_encoder(
        text_encoder_path=Path(text_encoder_path),
        tokenizer_path=Path(tokenizer_path),
    )
    loaded.metadata["prompt_encoder"] = prompt_encoder
    return prompt_encoder


def _generated_image(loaded: LoadedModelHandle) -> GeneratedImage | None:
    generated = loaded.metadata.get("generated_image")
    return generated if isinstance(generated, GeneratedImage) else None


def _cfg_normalization(stage: ExecutionStage) -> float:
    return cfg_normalization_from_extensions(stage.params.get("family_extensions"))


def _cfg_truncation(stage: ExecutionStage) -> float:
    return cfg_truncation_from_extensions(stage.params.get("family_extensions"))


def _max_sequence_length(stage: ExecutionStage) -> int:
    return max_sequence_length_from_extensions(stage.params.get("family_extensions"))


def _image_generator(loaded: LoadedModelHandle) -> ImageGenerator:
    existing = loaded.metadata.get("image_generator")
    if isinstance(existing, ImageGenerator):
        return existing
    component_paths = loaded.metadata.get("component_paths")
    if not isinstance(component_paths, dict):
        raise ValueError("Z-Image runtime state is missing component_paths")
    transformer_path = component_paths.get("transformer")
    vae_path = component_paths.get("vae")
    scheduler_path = component_paths.get("scheduler")
    if not isinstance(transformer_path, str) or not isinstance(vae_path, str):
        raise ValueError(
            "Z-Image runtime state is missing transformer/vae component paths"
        )
    if not isinstance(scheduler_path, str):
        raise ValueError("Z-Image runtime state is missing scheduler component path")
    image_generator = create_image_generator(
        transformer_path=Path(transformer_path),
        vae_path=Path(vae_path),
        scheduler_path=Path(scheduler_path),
    )
    loaded.metadata["image_generator"] = image_generator
    return image_generator


def _require_bundle_source(sources: dict[str, ConversionSource]) -> ConversionSource:
    if set(sources) != {"bundle"}:
        received = ", ".join(sorted(sources))
        raise ValueError(
            "Z-Image conversion requires exactly one 'bundle' source role, "
            f"got: {received or '<none>'}"
        )
    return sources["bundle"]


def _has_component_files(file_paths: set[str], component: str) -> bool:
    prefix = f"{component}/"
    return any(path.startswith(prefix) for path in file_paths)


def _collect_component_files(
    bundle_root: Path, components: Iterable[str]
) -> dict[str, tuple[Path, ...]]:
    collected: dict[str, tuple[Path, ...]] = {}
    for component in components:
        component_root = bundle_root / component
        if not component_root.is_dir():
            raise ValueError(
                f"Z-Image bundle is missing required component directory '{component}'"
            )
        files = tuple(
            candidate
            for candidate in sorted(component_root.rglob("*"))
            if candidate.is_file()
        )
        if not files:
            raise ValueError(
                f"Z-Image bundle component '{component}' does not contain any files"
            )
        collected[component] = files
    return collected


def _artifact_digest(
    *,
    model_id: str,
    precision: str,
    variant: str | None,
    component_files: dict[str, tuple[Path, ...]],
) -> str:
    hasher = hashlib.sha256()
    hasher.update(model_id.encode("utf-8"))
    hasher.update(precision.encode("utf-8"))
    hasher.update((variant or "unknown").encode("utf-8"))
    for component in sorted(component_files):
        hasher.update(component.encode("utf-8"))
        for file_path in component_files[component]:
            hasher.update(file_path.name.encode("utf-8"))
            hasher.update(str(file_path.stat().st_size).encode("utf-8"))
    return f"sha256:{hasher.hexdigest()}"


def _component_digest(bundle_root: Path, files: tuple[Path, ...]) -> str:
    hasher = hashlib.sha256()
    for file_path in files:
        hasher.update(str(file_path.relative_to(bundle_root)).encode("utf-8"))
        hasher.update(str(file_path.stat().st_size).encode("utf-8"))
    return f"sha256:{hasher.hexdigest()}"


def _capability_descriptor(
    *,
    model_id: str,
    artifact_digest: str,
    variant: str | None,
    provenance: ProvenanceRecord,
) -> CapabilityDescriptor:
    return CapabilityDescriptor(
        model_id=model_id,
        artifact_digest=artifact_digest,
        family="z_image",
        family_variant=variant,
        tasks=["image.generate"],
        modalities_in=["text"],
        modalities_out=["image"],
        constraints={
            "width": {"multiple_of": 16},
            "height": {"multiple_of": 16},
        },
        conditioning={
            "image": False,
            "video": False,
            "audio": False,
            "lora": False,
        },
        profiles_by_task={"image.generate": ["default"]},
        artifacts_out=list(_OUTPUT_FORMATS),
        scheduler_class=_SCHEDULER_CLASS,
        hardware_tiers=[
            HardwareTier(
                tier="apple-silicon-mlx",
                notes=(
                    "Native MLX still-image generation is implemented for the released "
                    "family rows; broader checkpoint validation is still in progress."
                ),
            )
        ],
        policy=PolicyDescriptor(
            license=provenance.license,
            access_state=provenance.access_state,
            remote_code_required=provenance.remote_code_required,
            remote_code_approved=provenance.remote_code_approved,
        ),
        metadata={
            "stage_ids": ["prompt_encode", "generate", "encode_output"],
            "required_components": list(_REQUIRED_COMPONENTS),
            "released_variants": list(_RELEASED_VARIANTS),
            "blocked_variants": list(_BLOCKED_VARIANTS),
            "upstream_remote_code_pressure": True,
            "native_runtime_status": _NATIVE_RUNTIME_STATUS,
        },
    )


def _variant_from_source(source: ResolvedSource) -> str | None:
    locator_candidates = [
        source.locator.get("path"),
        source.locator.get("repo"),
        source.metadata.get("repo_id"),
    ]
    return _variant_from_strings(
        candidate for candidate in locator_candidates if isinstance(candidate, str)
    )


def _variant_from_path(path: Path) -> str | None:
    return _variant_from_strings((str(path), path.name))


def _variant_from_strings(candidates: Iterable[str]) -> str | None:
    normalized = " ".join(candidate.lower() for candidate in candidates)
    if "turbo" in normalized:
        return "z-image-turbo"
    if "omni" in normalized:
        return "z-image-omni-base"
    if "edit" in normalized:
        return "z-image-edit"
    if "z-image" in normalized or "z_image" in normalized:
        return "z-image"
    return None


def _stage_dimension(value: object, *, name: str) -> int:
    if value is None:
        return 1024
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"Z-Image {name} must be a positive integer")
    return value


def _stage_positive_int(value: object, *, name: str, default: int) -> int:
    if value is None:
        return default
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"Z-Image {name} must be a positive integer")
    return value


def _stage_non_negative_float(value: object, *, name: str, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, int):
        return float(value)
    if not isinstance(value, float) or value < 0:
        raise ValueError(f"Z-Image {name} must be a non-negative number")
    return value


def _stage_seed(value: object) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int):
        raise ValueError("Z-Image seed must be an integer when provided")
    return value


def _require_output_dir(value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("Z-Image encode_output requires output_dir")
    return Path(value)


def _require_storage_key(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("Z-Image encode_output requires storage_key")
    return value


def _media_type_for_format(artifact_format: str) -> str:
    if artifact_format == "png":
        return "image/png"
    if artifact_format == "jpg":
        return "image/jpeg"
    return "application/octet-stream"
