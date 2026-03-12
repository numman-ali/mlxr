from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path

import mlx.core as mx
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
    ExtensionSchemaDescriptor,
    PolicyDescriptor,
    PortableArtifactComponentRecord,
    PortableArtifactRecord,
    ResolvedSource,
)

from .family_options import (
    prompt_enhance_mode_from_extensions,
    quantize_bits_from_extensions,
    scheduler_preset_from_extensions,
)
from .generation import (
    GeneratedImage,
    ImageGenerator,
    create_image_generator,
    encode_jpg_image,
    encode_png_image,
)
from .prompt_encoding import PromptEncoder, PromptEncodingResult, create_prompt_encoder

_SUPPORTED_GENERATION_VARIANTS = ("qwen-image", "qwen-image-2512")
_SUPPORTED_EDIT_VARIANTS = (
    "qwen-image-edit",
    "qwen-image-edit-2509",
    "qwen-image-edit-2511",
)
_BLOCKED_VARIANTS = ("qwen-image-layered", "qwen-image-2.0")
_OUTPUT_FORMATS = ("png", "jpg")
_SCHEDULER_CLASS = "image_diffusion"
_WEIGHT_FORMAT = "diffusers_component_bundle"
_FORMAT_VERSION = "0.1.0"
_NATIVE_RUNTIME_STATUS = "owned_mlx_backend_required"
_MAX_SEQUENCE_LENGTH = 512


class QwenImageFamilyAdapter:
    family_id = "qwen_image"
    _required_role = "bundle"

    def inspect_source(self, source: ResolvedSource) -> FamilyInspection:
        file_paths = {record.path for record in source.files}
        if "model_index.json" not in file_paths:
            raise ValueError("Qwen-Image source inspection requires model_index.json")
        variant = _variant_from_source(source)
        if variant in _BLOCKED_VARIANTS:
            raise ValueError(
                f"Qwen-Image variant '{variant}' is not yet enabled in MLXR"
            )
        required_components = _required_components(variant, file_paths)
        missing = [
            component
            for component in required_components
            if not _has_component_files(file_paths, component)
        ]
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(
                "Qwen-Image source inspection requires a diffusers-style bundle with "
                f"{missing_text} component directories"
            )
        tasks = _tasks_for_variant(variant)
        return FamilyInspection(
            family=self.family_id,
            variant=variant,
            tasks=tasks,
            scheduler_class=_SCHEDULER_CLASS,
            metadata={
                "stage_ids": ["prompt_encode", "generate", "encode_output"],
                "required_components": list(required_components),
                "supported_generation_variants": list(_SUPPORTED_GENERATION_VARIANTS),
                "supported_edit_variants": list(_SUPPORTED_EDIT_VARIANTS),
                "blocked_variants": list(_BLOCKED_VARIANTS),
                "native_runtime_status": _NATIVE_RUNTIME_STATUS,
            },
        )

    def fetch_policy_for_conversion(
        self, role: str, source: ResolvedSource
    ) -> FetchPolicy:
        if role != self._required_role:
            raise ValueError(
                "Qwen-Image conversion currently requires a single 'bundle' source role"
            )
        variant = _variant_from_source(source)
        required_components = _required_components(
            variant, {record.path for record in source.files}
        )
        allow_patterns = ["model_index.json"]
        for component in required_components:
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
                "Qwen-Image conversion requires a materialized local bundle directory"
            )
        variant = _variant_from_conversion_source(bundle_source)
        required_components = _required_components(
            variant, _relative_file_paths(bundle_root)
        )
        component_files = _collect_component_files(bundle_root, required_components)
        artifact_digest = _artifact_digest(
            model_id=plan.model_id,
            precision=plan.precision,
            variant=variant,
            component_files=component_files,
        )
        capability = _capability_descriptor(
            model_id=plan.model_id,
            artifact_digest=artifact_digest,
            variant=variant,
            policy=_policy_for_variant(variant),
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
            family_variant=variant,
            format_version=_FORMAT_VERSION,
            weight_format=_WEIGHT_FORMAT,
            storage_key="",
            capability=capability,
            provenance=bundle_source.materialization.provenance,
            components=components,
            metadata={
                "required_components": list(required_components),
                "supported_generation_variants": list(_SUPPORTED_GENERATION_VARIANTS),
                "supported_edit_variants": list(_SUPPORTED_EDIT_VARIANTS),
                "blocked_variants": list(_BLOCKED_VARIANTS),
                "native_runtime_status": _NATIVE_RUNTIME_STATUS,
            },
        )
        return PortableArtifact(record=record, payload_items=tuple(payload_items))

    def load(
        self, artifact: PortableArtifact, profile: ExecutionProfile
    ) -> LoadedModelHandle:
        if profile.task not in artifact.record.capability.tasks:
            raise ValueError(
                f"Qwen-Image artifact does not support task '{profile.task}'"
            )
        if artifact.storage_path is None:
            raise ValueError(
                "Qwen-Image load requires a materialized artifact storage path"
            )
        required_components = _required_components(
            artifact.record.family_variant or "",
            _relative_file_paths(artifact.storage_path / "payload"),
            payload_root=artifact.storage_path / "payload",
        )
        component_paths: dict[str, str] = {}
        for component in required_components:
            component_root = artifact.storage_path / "payload" / component
            if not component_root.is_dir():
                raise ValueError(
                    "Qwen-Image artifact is missing required component directory "
                    f"'{component_root}'"
                )
            component_paths[component] = str(component_root)
        return LoadedModelHandle(
            model_id=artifact.record.model_id,
            family=self.family_id,
            artifact_digest=artifact.record.artifact_digest,
            capability=self.capabilities(artifact),
            metadata={
                "artifact_storage_path": str(artifact.storage_path),
                "component_paths": component_paths,
                "requested_profile": profile.profile,
                "requested_task": profile.task,
                "family_variant": artifact.record.family_variant,
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
            image_paths = _resolved_prompt_image_paths(stage.params)
            prompt_context = prompt_encoder.encode(
                prompt,
                max_length=_MAX_SEQUENCE_LENGTH,
                negative_prompt=negative_prompt,
                image_paths=image_paths,
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
            if (
                prompt_enhance_mode_from_extensions(
                    stage.params.get("family_extensions")
                )
                != "none"
            ):
                raise ValueError(
                    "Qwen-Image prompt enhancement is not implemented in MLXR yet"
                )
            prompt_context_value = _prompt_context(loaded)
            if prompt_context_value is None:
                raise ValueError(
                    "Qwen-Image generate stage requires prompt_encode to run successfully first"
                )
            prompt_context = prompt_context_value
            image_generator = _image_generator(loaded, stage)
            generated_image = image_generator.generate(
                prompt_context=prompt_context,
                task=_stage_task(loaded, stage),
                width=_optional_stage_dimension(stage.params.get("width")),
                height=_optional_stage_dimension(stage.params.get("height")),
                num_inference_steps=_optional_positive_int(
                    stage.params.get("num_inference_steps")
                ),
                guidance_scale=_optional_non_negative_float(
                    stage.params.get("guidance_scale")
                ),
                seed=_stage_seed(stage.params.get("seed")),
                image_paths=_resolved_image_paths(stage),
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
                    "Qwen-Image encode_output requires generate to run successfully first"
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
                    "Qwen-Image encode_output only supports runtime-managed png or jpg "
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
            "Qwen-Image stage execution is not implemented for unsupported stage "
            f"'{stage.stage_id}'"
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
        loaded.metadata["execution_status"] = "unloaded"


def _relative_file_paths(root: Path) -> set[str]:
    return {
        str(file_path.relative_to(root))
        for file_path in root.rglob("*")
        if file_path.is_file()
    }


def _variant_from_source(source: ResolvedSource) -> str:
    locator_candidates = (
        source.locator.get("variant"),
        source.locator.get("family_variant"),
        source.locator.get("repo"),
        source.locator.get("repo_id"),
        source.locator.get("path"),
        source.pinned_ref,
    )
    for raw_value in locator_candidates:
        if isinstance(raw_value, str) and raw_value.strip():
            return _normalize_variant_name(raw_value)
    return "qwen-image-2512"


def _variant_from_path(path: Path) -> str:
    return _normalize_variant_name(path.name)


def _variant_from_conversion_source(source: ConversionSource) -> str:
    locator_candidates = (
        source.materialization.resolved.locator.get("variant"),
        source.source.locator.get("variant"),
    )
    for raw_value in locator_candidates:
        if isinstance(raw_value, str) and raw_value.strip():
            return _normalize_variant_name(raw_value)
    local_path = source.materialization.local_path
    if local_path is not None:
        return _variant_from_path(local_path)
    return _variant_from_source(source.materialization.resolved)


def _normalize_variant_name(raw_value: str) -> str:
    tail = raw_value.rstrip("/").split("/")[-1]
    return tail.replace("_", "-").lower()


def _tasks_for_variant(variant: str) -> tuple[str, ...]:
    if variant in _SUPPORTED_GENERATION_VARIANTS:
        return ("image.generate",)
    if variant in _SUPPORTED_EDIT_VARIANTS:
        return ("image.edit",)
    raise ValueError(f"Unsupported Qwen-Image variant '{variant}'")


def _required_components(
    variant: str, file_paths: set[str], *, payload_root: Path | None = None
) -> tuple[str, ...]:
    del payload_root
    components = ["transformer", "vae", "text_encoder", "tokenizer", "scheduler"]
    if variant in _SUPPORTED_EDIT_VARIANTS and (
        any(path.startswith("processor/") for path in file_paths)
        or variant.endswith("2511")
    ):
        components.append("processor")
    return tuple(components)


def _has_component_files(file_paths: set[str], component: str) -> bool:
    prefix = f"{component}/"
    return any(path.startswith(prefix) for path in file_paths)


def _collect_component_files(
    bundle_root: Path, required_components: Iterable[str]
) -> dict[str, tuple[Path, ...]]:
    collected: dict[str, tuple[Path, ...]] = {}
    for component in required_components:
        component_root = bundle_root / component
        files = tuple(
            path for path in sorted(component_root.rglob("*")) if path.is_file()
        )
        if not files:
            raise ValueError(
                f"Qwen-Image bundle is missing files for component '{component}'"
            )
        collected[component] = files
    return collected


def _require_bundle_source(
    sources: dict[str, ConversionSource],
) -> ConversionSource:
    bundle_source = sources.get("bundle")
    if bundle_source is None:
        raise ValueError("Qwen-Image conversion requires a 'bundle' source")
    return bundle_source


def _artifact_digest(
    *,
    model_id: str,
    precision: str,
    variant: str,
    component_files: dict[str, tuple[Path, ...]],
) -> str:
    hasher = hashlib.sha256()
    hasher.update(model_id.encode("utf-8"))
    hasher.update(precision.encode("utf-8"))
    hasher.update(variant.encode("utf-8"))
    for component in sorted(component_files):
        hasher.update(component.encode("utf-8"))
        for file_path in component_files[component]:
            hasher.update(
                str(file_path.relative_to(file_path.parents[1])).encode("utf-8")
            )
            hasher.update(file_path.read_bytes())
    return f"sha256:{hasher.hexdigest()}"


def _component_digest(bundle_root: Path, files: tuple[Path, ...]) -> str:
    hasher = hashlib.sha256()
    for file_path in files:
        hasher.update(str(file_path.relative_to(bundle_root)).encode("utf-8"))
        hasher.update(file_path.read_bytes())
    return f"sha256:{hasher.hexdigest()}"


def _policy_for_variant(variant: str) -> PolicyDescriptor:
    return PolicyDescriptor(
        license="apache-2.0",
        access_state="public",
        remote_code_required=False,
        remote_code_approved=False,
        redistribution_state="unknown",
    )


def _capability_descriptor(
    *,
    model_id: str,
    artifact_digest: str,
    variant: str,
    policy: PolicyDescriptor,
) -> CapabilityDescriptor:
    tasks = list(_tasks_for_variant(variant))
    conditioning = (
        {"image": {"required": True, "max_references": None}}
        if tasks == ["image.edit"]
        else {}
    )
    modalities_in = ["text"] if tasks == ["image.generate"] else ["text", "image"]
    return CapabilityDescriptor(
        model_id=model_id,
        artifact_digest=artifact_digest,
        family="qwen_image",
        family_variant=variant,
        tasks=tasks,
        modalities_in=modalities_in,
        modalities_out=["image"],
        constraints={
            "width": {"multiple_of": 16},
            "height": {"multiple_of": 16},
        },
        conditioning=conditioning,
        profiles_by_task={task: ["default"] for task in tasks},
        streaming={"artifacts": False},
        artifacts_out=list(_OUTPUT_FORMATS),
        scheduler_class=_SCHEDULER_CLASS,
        policy=policy,
        extensions_schema=ExtensionSchemaDescriptor(
            namespace="qwen_image",
            version="0.1.0",
        ),
        metadata={
            "native_runtime_status": _NATIVE_RUNTIME_STATUS,
            "stage_ids": ["prompt_encode", "generate", "encode_output"],
            "primary_generation_variant": "qwen-image-2512",
            "primary_edit_variant": "qwen-image-edit-2511",
        },
    )


def _require_prompt(inputs: dict[str, object]) -> str:
    prompt = inputs.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Qwen-Image generate requires a non-empty prompt")
    return prompt


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("Expected text input to be a string when provided")
    return value


def _prompt_context(loaded: LoadedModelHandle) -> PromptEncodingResult | None:
    context = loaded.metadata.get("prompt_context")
    return context if isinstance(context, PromptEncodingResult) else None


def _prompt_encoder(loaded: LoadedModelHandle) -> PromptEncoder:
    existing = loaded.metadata.get("prompt_encoder")
    if isinstance(existing, PromptEncoder):
        return existing
    component_paths = loaded.metadata.get("component_paths")
    if not isinstance(component_paths, dict):
        raise ValueError("Qwen-Image runtime state is missing component_paths")
    text_encoder_path = component_paths.get("text_encoder")
    tokenizer_path = component_paths.get("tokenizer")
    processor_path = component_paths.get("processor")
    task = loaded.metadata.get("requested_task")
    if not isinstance(task, str) or not task:
        if loaded.capability.tasks:
            task = loaded.capability.tasks[0]
        else:
            raise ValueError("Qwen-Image runtime state is missing requested_task")
    if not isinstance(text_encoder_path, str) or not isinstance(tokenizer_path, str):
        raise ValueError(
            "Qwen-Image runtime state is missing text_encoder/tokenizer component paths"
        )
    prompt_encoder = create_prompt_encoder(
        text_encoder_path=Path(text_encoder_path),
        tokenizer_path=Path(tokenizer_path),
        processor_path=Path(processor_path)
        if isinstance(processor_path, str)
        else None,
        task=task,
    )
    loaded.metadata["prompt_encoder"] = prompt_encoder
    return prompt_encoder


def _resolved_prompt_image_paths(params: dict[str, object]) -> tuple[Path, ...]:
    resolved_inputs = params.get("resolved_inputs")
    if not isinstance(resolved_inputs, dict):
        return ()
    raw_images = resolved_inputs.get("images")
    if not isinstance(raw_images, list):
        return ()
    image_paths: list[Path] = []
    for item in raw_images:
        if not isinstance(item, dict):
            continue
        payload_path = item.get("payload_path")
        if isinstance(payload_path, str) and payload_path:
            image_paths.append(Path(payload_path))
    return tuple(image_paths)


def _stage_task(loaded: LoadedModelHandle, stage: ExecutionStage) -> str:
    task = stage.params.get("task")
    if not isinstance(task, str) or not task:
        task = loaded.metadata.get("requested_task")
    if not isinstance(task, str) or not task:
        raise ValueError("Qwen-Image stage execution requires a task")
    return task


def _optional_stage_dimension(value: object) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or value <= 0:
        raise ValueError("Stage dimension values must be positive integers")
    return value


def _optional_positive_int(value: object) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or value <= 0:
        raise ValueError("num_inference_steps must be a positive integer when provided")
    return value


def _optional_non_negative_float(value: object) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or float(value) < 0.0:
        raise ValueError("guidance_scale must be non-negative when provided")
    return float(value)


def _stage_seed(value: object) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int):
        raise ValueError("seed must be an integer when provided")
    return value


def _image_generator(
    loaded: LoadedModelHandle, stage: ExecutionStage
) -> ImageGenerator:
    cached = loaded.metadata.get("image_generator")
    if isinstance(cached, ImageGenerator):
        return cached
    prompt_encoder = loaded.metadata.get("prompt_encoder")
    close = getattr(prompt_encoder, "close", None)
    if callable(close):
        close()
        loaded.metadata.pop("prompt_encoder", None)
        mx.clear_cache()
    artifact_storage = loaded.metadata.get("artifact_storage_path")
    if not isinstance(artifact_storage, str) or not artifact_storage:
        raise ValueError("Qwen-Image loaded handle is missing artifact storage path")
    family_variant = loaded.metadata.get("family_variant")
    if not isinstance(family_variant, str) or not family_variant:
        raise ValueError("Qwen-Image loaded handle is missing family_variant")
    family_extensions = stage.params.get("family_extensions")
    quantize_bits = quantize_bits_from_extensions(family_extensions)
    scheduler_preset = scheduler_preset_from_extensions(family_extensions)
    resolved_loras = _resolved_loras(stage)
    lora_paths: list[Path] = []
    lora_scales: list[float] = []
    for item in resolved_loras:
        path = item["path"]
        scale = item["scale"]
        if not isinstance(path, Path):
            raise ValueError("Resolved LoRA path must be a Path")
        if not isinstance(scale, float):
            raise ValueError("Resolved LoRA scale must be a float")
        lora_paths.append(path)
        lora_scales.append(scale)
    generator = create_image_generator(
        variant=family_variant,
        model_root=Path(artifact_storage) / "payload",
        task=_stage_task(loaded, stage),
        quantize_bits=quantize_bits,
        scheduler_preset=scheduler_preset,
        lora_paths=tuple(lora_paths),
        lora_scales=tuple(lora_scales),
    )
    loaded.metadata["image_generator"] = generator
    return generator


def _resolved_image_paths(stage: ExecutionStage) -> tuple[Path, ...]:
    resolved_inputs = stage.params.get("resolved_inputs")
    if not isinstance(resolved_inputs, dict):
        return ()
    raw_images = resolved_inputs.get("images")
    if not isinstance(raw_images, list):
        return ()
    image_paths: list[Path] = []
    for item in raw_images:
        if not isinstance(item, dict):
            raise ValueError("Resolved image inputs must be objects")
        payload_path = item.get("payload_path")
        if not isinstance(payload_path, str) or not payload_path:
            raise ValueError("Resolved image inputs require payload_path")
        image_paths.append(Path(payload_path))
    return tuple(image_paths)


def _resolved_loras(stage: ExecutionStage) -> tuple[dict[str, object], ...]:
    resolved_inputs = stage.params.get("resolved_inputs")
    if not isinstance(resolved_inputs, dict):
        return ()
    raw_loras = resolved_inputs.get("loras")
    if not isinstance(raw_loras, list):
        return ()
    results: list[dict[str, object]] = []
    for item in raw_loras:
        if not isinstance(item, dict):
            raise ValueError("Resolved LoRA inputs must be objects")
        payload_path = item.get("payload_path")
        if not isinstance(payload_path, str) or not payload_path:
            raise ValueError("Resolved LoRA inputs require payload_path")
        strength = item.get("strength", 1.0)
        if not isinstance(strength, (int, float)) or float(strength) <= 0.0:
            raise ValueError("Resolved LoRA strength must be > 0.0")
        results.append({"path": Path(payload_path), "scale": float(strength)})
    return tuple(results)


def _generated_image(loaded: LoadedModelHandle) -> GeneratedImage | None:
    generated = loaded.metadata.get("generated_image")
    return generated if isinstance(generated, GeneratedImage) else None


def _require_output_dir(value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("Qwen-Image encode_output requires a non-empty output_dir")
    return Path(value)


def _require_storage_key(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("Qwen-Image encode_output requires a non-empty storage_key")
    return value


def _media_type_for_format(artifact_format: str) -> str:
    if artifact_format == "png":
        return "image/png"
    if artifact_format == "jpg":
        return "image/jpeg"
    raise ValueError(f"Unsupported Qwen-Image artifact format '{artifact_format}'")
