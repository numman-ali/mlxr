from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mlx_runtime_schemas import ProvenanceRecord

from ..generation import (
    AudioConditioningInput,
    ConditioningInput,
    GeneratedVideo,
    VideoGenerator,
)
from ..prompt_encoding import PromptEncoder, PromptEncodingResult


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
    audio_conditioning: AudioConditioningInput | None = None
    generated_video: GeneratedVideo | None = None
