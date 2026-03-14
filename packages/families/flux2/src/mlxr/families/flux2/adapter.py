from __future__ import annotations

import hashlib
import json
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
    ExtensionSchemaDescriptor,
    PolicyDescriptor,
    PortableArtifactComponentRecord,
    PortableArtifactRecord,
    ResolvedSource,
)

from .family_options import (
    prompt_upsampling_mode_from_extensions,
    quantize_bits_from_extensions,
)
from .generation import (
    GeneratedImage,
    ImageGenerator,
    create_image_generator,
    encode_jpg_image,
    encode_png_image,
)

_SUPPORTED_VARIANTS = (
    "flux.2-klein-4b",
    "flux.2-klein-9b",
    "flux.2-klein-9b-kv",
    "flux.2-klein-base-4b",
    "flux.2-klein-base-9b",
    "flux.2-dev",
)
_OUTPUT_FORMATS = ("png", "jpg")
_REQUIRED_COMPONENTS = ("transformer", "vae", "text_encoder", "tokenizer", "scheduler")
_SCHEDULER_CLASS = "image_diffusion"
_WEIGHT_FORMAT = "diffusers_component_bundle"
_FORMAT_VERSION = "0.1.0"
_NATIVE_RUNTIME_STATUS = "experimental_native_mlx_backend"
_COMPONENT_REQUIRED_FILES: dict[str, tuple[tuple[str, ...], ...]] = {
    "transformer": (
        ("transformer/config.json",),
        (
            "transformer/diffusion_pytorch_model.safetensors",
            "transformer/diffusion_pytorch_model.safetensors.index.json",
        ),
    ),
    "vae": (
        ("vae/config.json",),
        ("vae/diffusion_pytorch_model.safetensors",),
    ),
    "text_encoder": (
        ("text_encoder/config.json",),
        (
            "text_encoder/model.safetensors",
            "text_encoder/model.safetensors.index.json",
        ),
    ),
    "tokenizer": (
        ("tokenizer/tokenizer.json",),
        ("tokenizer/tokenizer_config.json",),
    ),
    "scheduler": (("scheduler/scheduler_config.json",),),
}


class Flux2FamilyAdapter:
    family_id = "flux2"
    _required_role = "bundle"

    def inspect_source(self, source: ResolvedSource) -> FamilyInspection:
        file_paths = {record.path for record in source.files}
        if "model_index.json" not in file_paths:
            raise ValueError("FLUX.2 source inspection requires model_index.json")
        variant = _variant_from_source(source)
        if variant not in _SUPPORTED_VARIANTS:
            raise ValueError(f"Unsupported FLUX.2 variant '{variant}'")
        missing = [
            component
            for component in _REQUIRED_COMPONENTS
            if not _has_required_component_files(file_paths, component)
        ]
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(
                "FLUX.2 source inspection requires a diffusers-style bundle with "
                f"{missing_text} component directories"
            )
        return FamilyInspection(
            family=self.family_id,
            variant=variant,
            tasks=("image.generate", "image.edit"),
            scheduler_class=_SCHEDULER_CLASS,
            metadata={
                "stage_ids": ["generate", "encode_output"],
                "required_components": list(_REQUIRED_COMPONENTS),
                "supported_variants": list(_SUPPORTED_VARIANTS),
                "native_runtime_status": _NATIVE_RUNTIME_STATUS,
                "primary_klein_variant": "flux.2-klein-9b",
                "edit_optimized_klein_variant": "flux.2-klein-9b-kv",
            },
        )

    def fetch_policy_for_conversion(
        self, role: str, source: ResolvedSource
    ) -> FetchPolicy:
        if role != self._required_role:
            raise ValueError(
                "FLUX.2 conversion currently requires a single 'bundle' source role"
            )
        allow_patterns = ["model_index.json"]
        for component in _REQUIRED_COMPONENTS:
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
                "FLUX.2 conversion requires a materialized local bundle directory"
            )
        variant = _variant_from_conversion_source(bundle_source)
        if variant not in _SUPPORTED_VARIANTS:
            raise ValueError(f"Unsupported FLUX.2 variant '{variant}'")
        component_files = _collect_component_files(bundle_root, _REQUIRED_COMPONENTS)
        _validate_component_materialization(bundle_root, component_files)
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
                "required_components": list(_REQUIRED_COMPONENTS),
                "supported_variants": list(_SUPPORTED_VARIANTS),
                "native_runtime_status": _NATIVE_RUNTIME_STATUS,
                "edit_optimized_klein_variant": "flux.2-klein-9b-kv",
            },
        )
        return PortableArtifact(record=record, payload_items=tuple(payload_items))

    def load(
        self, artifact: PortableArtifact, profile: ExecutionProfile
    ) -> LoadedModelHandle:
        if profile.task not in artifact.record.capability.tasks:
            raise ValueError(f"FLUX.2 artifact does not support task '{profile.task}'")
        if artifact.storage_path is None:
            raise ValueError(
                "FLUX.2 load requires a materialized artifact storage path"
            )
        component_paths: dict[str, str] = {}
        for component in _REQUIRED_COMPONENTS:
            component_root = artifact.storage_path / "payload" / component
            if not component_root.is_dir():
                raise ValueError(
                    f"FLUX.2 artifact is missing required component directory '{component_root}'"
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

    def normalize_capability(self, artifact: PortableArtifact) -> CapabilityDescriptor:
        return self.capabilities(artifact)

    def run_stage(
        self, loaded: LoadedModelHandle, stage: ExecutionStage
    ) -> StageResult:
        if stage.stage_id == "generate":
            if (
                prompt_upsampling_mode_from_extensions(
                    stage.params.get("family_extensions")
                )
                != "none"
            ):
                raise ValueError(
                    "FLUX.2 prompt upsampling is not implemented in MLXR yet"
                )
            image_generator = _image_generator(loaded, stage)
            prompt = _require_prompt(stage.inputs)
            generated_image = image_generator.generate(
                prompt=prompt,
                task=_stage_task(loaded, stage),
                width=_optional_stage_dimension(stage.params.get("width")),
                height=_optional_stage_dimension(stage.params.get("height")),
                num_inference_steps=_optional_positive_int(
                    stage.params.get("num_inference_steps")
                ),
                guidance_scale=_optional_non_negative_float(
                    stage.params.get("guidance_scale")
                ),
                negative_prompt=_optional_text(stage.inputs.get("negative_prompt")),
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
                    "FLUX.2 encode_output requires generate to run successfully first"
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
                    "FLUX.2 encode_output only supports runtime-managed png or jpg "
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
            f"FLUX.2 stage execution is not implemented for unsupported stage '{stage.stage_id}'"
        )

    def unload(self, loaded: LoadedModelHandle) -> None:
        image_generator = loaded.metadata.pop("image_generator", None)
        if hasattr(image_generator, "close"):
            image_generator.close()
        loaded.metadata.pop("generated_image", None)
        loaded.metadata["execution_status"] = "unloaded"


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
    return "flux.2-klein-9b"


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


def _has_required_component_files(file_paths: set[str], component: str) -> bool:
    requirements = _COMPONENT_REQUIRED_FILES.get(component)
    if requirements is None:
        return False
    return all(
        any(candidate in file_paths for candidate in group) for group in requirements
    )


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
                f"FLUX.2 bundle is missing files for component '{component}'"
            )
        collected[component] = files
    return collected


def _validate_component_materialization(
    bundle_root: Path,
    component_files: dict[str, tuple[Path, ...]],
) -> None:
    available_paths = {
        str(file_path.relative_to(bundle_root))
        for files in component_files.values()
        for file_path in files
    }
    missing_components = [
        component
        for component in _REQUIRED_COMPONENTS
        if not _has_required_component_files(available_paths, component)
    ]
    if missing_components:
        missing_text = ", ".join(sorted(missing_components))
        raise ValueError(
            f"FLUX.2 bundle is missing required materialized files for {missing_text}"
        )
    _validate_index_backed_component(
        bundle_root,
        component="transformer",
        index_relative_path="transformer/diffusion_pytorch_model.safetensors.index.json",
    )
    _validate_index_backed_component(
        bundle_root,
        component="text_encoder",
        index_relative_path="text_encoder/model.safetensors.index.json",
    )


def _validate_index_backed_component(
    bundle_root: Path,
    *,
    component: str,
    index_relative_path: str,
) -> None:
    index_path = bundle_root / index_relative_path
    if not index_path.exists():
        return
    raw_index = json.loads(index_path.read_text("utf-8"))
    weight_map = raw_index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"Invalid FLUX.2 weight index at '{index_relative_path}'")
    missing_files = sorted(
        {
            filename
            for filename in weight_map.values()
            if not (bundle_root / component / str(filename)).is_file()
        }
    )
    if missing_files:
        missing_text = ", ".join(str(name) for name in missing_files)
        raise ValueError(
            f"FLUX.2 bundle is incomplete for '{component}'; missing shard files: {missing_text}"
        )


def _require_bundle_source(
    sources: dict[str, ConversionSource],
) -> ConversionSource:
    bundle_source = sources.get("bundle")
    if bundle_source is None:
        raise ValueError("FLUX.2 conversion requires a 'bundle' source")
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
    if variant in {"flux.2-klein-4b", "flux.2-klein-base-4b"}:
        license_name = "apache-2.0"
    else:
        license_name = "flux-non-commercial"
    return PolicyDescriptor(
        license=license_name,
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
    constraints: dict[str, object] = {
        "width": {"multiple_of": 16},
        "height": {"multiple_of": 16},
    }
    if "klein" in variant and "base" not in variant:
        constraints["num_inference_steps"] = {"fixed": 4}
        constraints["guidance_scale"] = {"fixed": 1.0}
    return CapabilityDescriptor(
        model_id=model_id,
        artifact_digest=artifact_digest,
        family="flux2",
        family_variant=variant,
        tasks=["image.generate", "image.edit"],
        modalities_in=["text", "image"],
        modalities_out=["image"],
        constraints=constraints,
        conditioning={
            "image": {"required_for": ["image.edit"], "max_references": None}
        },
        profiles_by_task={"image.generate": ["default"], "image.edit": ["default"]},
        streaming={"artifacts": False},
        artifacts_out=list(_OUTPUT_FORMATS),
        scheduler_class=_SCHEDULER_CLASS,
        policy=policy,
        extensions_schema=ExtensionSchemaDescriptor(namespace="flux2", version="0.1.0"),
        metadata={
            "native_runtime_status": _NATIVE_RUNTIME_STATUS,
            "stage_ids": ["generate", "encode_output"],
            "primary_klein_variant": "flux.2-klein-9b",
            "edit_optimized_klein_variant": "flux.2-klein-9b-kv",
            "lighter_klein_variant": "flux.2-klein-4b",
            "high_end_variant": "flux.2-dev",
        },
    )


def _require_prompt(inputs: dict[str, object]) -> str:
    prompt = inputs.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("FLUX.2 generate requires a non-empty prompt")
    return prompt


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("Expected text input to be a string when provided")
    return value


def _stage_task(loaded: LoadedModelHandle, stage: ExecutionStage) -> str:
    task = stage.params.get("task")
    if not isinstance(task, str) or not task:
        task = loaded.metadata.get("requested_task")
    if not isinstance(task, str) or not task:
        raise ValueError("FLUX.2 stage execution requires a task")
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
    artifact_storage = loaded.metadata.get("artifact_storage_path")
    if not isinstance(artifact_storage, str) or not artifact_storage:
        raise ValueError("FLUX.2 loaded handle is missing artifact storage path")
    family_variant = loaded.metadata.get("family_variant")
    if not isinstance(family_variant, str) or not family_variant:
        raise ValueError("FLUX.2 loaded handle is missing family_variant")
    resolved_loras = _resolved_loras(stage)
    family_extensions = stage.params.get("family_extensions")
    quantize_bits = quantize_bits_from_extensions(family_extensions)
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
        raise ValueError("FLUX.2 encode_output requires a non-empty output_dir")
    return Path(value)


def _require_storage_key(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("FLUX.2 encode_output requires a non-empty storage_key")
    return value


def _media_type_for_format(artifact_format: str) -> str:
    if artifact_format == "png":
        return "image/png"
    if artifact_format == "jpg":
        return "image/jpeg"
    raise ValueError(f"Unsupported FLUX.2 artifact format '{artifact_format}'")
