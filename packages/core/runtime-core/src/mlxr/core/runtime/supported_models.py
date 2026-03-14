from __future__ import annotations

from dataclasses import dataclass

from mlxr.core.schemas import (
    ArtifactConversionRequest,
    RecommendationTier,
    SourceAuth,
    SourceRef,
    SupportedModelDescriptor,
    SupportLevel,
)

_DEFAULT_REVISION = "main"


@dataclass(frozen=True, slots=True)
class SupportedModelRecipe:
    model_id: str
    display_name: str
    family: str
    family_variant: str | None
    recommendation_tier: RecommendationTier
    support_level: SupportLevel
    tasks: tuple[str, ...]
    provider: str
    source_summary: str
    license: str | None
    access_state: str
    notes: str | None = None
    source_ref: SourceRef | None = None
    source_bindings: dict[str, SourceRef] | None = None

    def to_descriptor(self, *, installed: bool) -> SupportedModelDescriptor:
        return SupportedModelDescriptor(
            model_id=self.model_id,
            display_name=self.display_name,
            family=self.family,
            family_variant=self.family_variant,
            recommendation_tier=self.recommendation_tier,
            support_level=self.support_level,
            tasks=list(self.tasks),
            provider=self.provider,
            source_summary=self.source_summary,
            license=self.license,
            access_state=self.access_state,
            installed=installed,
            installable=True,
            notes=self.notes,
        )

    def to_conversion_request(
        self, *, registered_source_ids: dict[str, str]
    ) -> ArtifactConversionRequest:
        if self.source_ref is not None:
            source_id = registered_source_ids.get("bundle")
            if source_id is None:
                raise ValueError(
                    f"Supported model '{self.model_id}' is missing registered bundle source"
                )
            return ArtifactConversionRequest(
                source_id=source_id, model_id=self.model_id
            )
        if self.source_bindings is not None:
            return ArtifactConversionRequest(
                source_bindings=dict(registered_source_ids),
                family=self.family,
                model_id=self.model_id,
            )
        raise ValueError(
            f"Supported model '{self.model_id}' is missing source configuration"
        )


def _hf_source_ref(
    *, repo: str, family_hint: str, role: str | None = None
) -> SourceRef:
    locator: dict[str, str] = {"repo": repo, "revision": _DEFAULT_REVISION}
    if role is not None:
        locator["role"] = role
    return SourceRef(
        provider="huggingface",
        locator=locator,
        auth=SourceAuth(token_ref="hf-default"),
        family_hint=family_hint,
    )


SUPPORTED_MODEL_RECIPES: tuple[SupportedModelRecipe, ...] = (
    SupportedModelRecipe(
        model_id="ltx-2.3-fast-local",
        display_name="LTX 2.3 Fast",
        family="ltx",
        family_variant=None,
        recommendation_tier="recommended",
        support_level="promoted",
        tasks=("video.generate", "video.condition.image", "video.condition.audio"),
        provider="huggingface",
        source_summary="Lightricks/LTX-2.3 + Gemma 3 text encoder",
        license="other",
        access_state="gated",
        source_bindings={
            "checkpoint": _hf_source_ref(
                repo="Lightricks/LTX-2.3", family_hint="ltx", role="checkpoint"
            ),
            "spatial_upsampler": _hf_source_ref(
                repo="Lightricks/LTX-2.3",
                family_hint="ltx",
                role="spatial_upsampler",
            ),
            "text_encoder": _hf_source_ref(
                repo="google/gemma-3-12b-it-qat-q4_0-unquantized",
                family_hint="ltx",
                role="text_encoder",
            ),
        },
    ),
    SupportedModelRecipe(
        model_id="z-image-turbo-local",
        display_name="Z-Image Turbo",
        family="z_image",
        family_variant="z-image-turbo",
        recommendation_tier="recommended",
        support_level="promoted",
        tasks=("image.generate",),
        provider="huggingface",
        source_summary="Tongyi-MAI/Z-Image-Turbo",
        license="other",
        access_state="public",
        source_ref=_hf_source_ref(
            repo="Tongyi-MAI/Z-Image-Turbo", family_hint="z_image"
        ),
    ),
    SupportedModelRecipe(
        model_id="qwen-image-local",
        display_name="Qwen-Image 2512",
        family="qwen_image",
        family_variant="qwen-image-2512",
        recommendation_tier="recommended",
        support_level="promoted",
        tasks=("image.generate",),
        provider="huggingface",
        source_summary="Qwen/Qwen-Image-2512",
        license="apache-2.0",
        access_state="public",
        source_ref=_hf_source_ref(
            repo="Qwen/Qwen-Image-2512", family_hint="qwen_image"
        ),
    ),
    SupportedModelRecipe(
        model_id="qwen-image-edit-local",
        display_name="Qwen-Image Edit 2511",
        family="qwen_image",
        family_variant="qwen-image-edit-2511",
        recommendation_tier="advanced",
        support_level="supported",
        tasks=("image.edit",),
        provider="huggingface",
        source_summary="Qwen/Qwen-Image-Edit-2511",
        license="apache-2.0",
        access_state="public",
        notes="Supported for editing, but not currently promoted as a default row.",
        source_ref=_hf_source_ref(
            repo="Qwen/Qwen-Image-Edit-2511", family_hint="qwen_image"
        ),
    ),
    SupportedModelRecipe(
        model_id="flux2-klein-9b-local",
        display_name="FLUX.2 Klein 9B",
        family="flux2",
        family_variant="flux.2-klein-9b",
        recommendation_tier="recommended",
        support_level="promoted",
        tasks=("image.generate", "image.edit"),
        provider="huggingface",
        source_summary="black-forest-labs/FLUX.2-klein-9B",
        license="other-non-commercial",
        access_state="public",
        source_ref=_hf_source_ref(
            repo="black-forest-labs/FLUX.2-klein-9B", family_hint="flux2"
        ),
    ),
)
