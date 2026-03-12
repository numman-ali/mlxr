from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol, runtime_checkable

try:
    import mlx.core as mx
    import numpy as np
    from mlxr.core.mlx_models import (
        Qwen2TextModel,
        Qwen2VLImageProcessorConfig,
        Qwen2VLVisionModel,
        TextTokenizer,
        expand_image_placeholders,
        load_local_qwen2_text_model,
        load_local_qwen2_vl_vision_model,
        load_local_text_tokenizer,
        preprocess_qwen2_vl_images,
    )
    from PIL import Image
    from PIL.Image import Resampling
    from transformers import AutoTokenizer
except Exception as exc:  # pragma: no cover - exercised via runtime import failure path
    _RUNTIME_IMPORT_ERROR: Exception | None = exc
else:
    _RUNTIME_IMPORT_ERROR = None

from .prompt_encoding import PromptEncoder, PromptEncodingResult

_PROMPT_TEMPLATE = (
    "<|im_start|>system\n"
    "Describe the image by detailing the color, shape, size, texture, quantity, "
    "text, spatial relationships of the objects and background:<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
_PROMPT_TEMPLATE_DROP_TOKENS = 34
_EDIT_PROMPT_TEMPLATE = (
    "<|im_start|>system\n"
    "Describe the key features of the input image (color, shape, size, texture, "
    "objects, background), then explain how the user's text instruction should "
    "alter or modify the image. Generate a new image that meets the user's "
    "requirements while maintaining consistency with the original input where "
    "appropriate.<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
_EDIT_PROMPT_TEMPLATE_DROP_TOKENS = 64
_IMAGE_TOKEN = "<|image_pad|>"
_TOKENIZER_MAX_LENGTH = 1024
_CONDITION_IMAGE_AREA = 384 * 384

if _RUNTIME_IMPORT_ERROR is None:

    @runtime_checkable
    class _RawTokenizer(Protocol):
        def __call__(
            self, text: list[str], **kwargs: object
        ) -> Mapping[str, object]: ...

        def convert_tokens_to_ids(self, token: str) -> int: ...

    @dataclass(slots=True)
    class _RuntimePromptEncoder(PromptEncoder):
        text_encoder_path: Path
        tokenizer_path: Path
        _model: Qwen2TextModel
        _tokenizer: TextTokenizer

        def encode(
            self,
            prompt: str,
            *,
            max_length: int = 512,
            negative_prompt: str | None = None,
            image_paths: tuple[Path, ...] = (),
        ) -> PromptEncodingResult:
            if image_paths:
                raise ValueError(
                    "Qwen-Image generate prompt encoding does not accept image_paths"
                )
            positive = _encode_prompt(
                prompt_text=prompt,
                max_length=max_length,
                model=self._model,
                tokenizer=self._tokenizer,
            )
            negative_embeddings: object | None = None
            negative_attention_mask: object | None = None
            if negative_prompt is not None:
                negative_embeddings, negative_attention_mask, _, _, _ = _encode_prompt(
                    prompt_text=negative_prompt,
                    max_length=max_length,
                    model=self._model,
                    tokenizer=self._tokenizer,
                )
            return PromptEncodingResult(
                prompt_embeddings=positive[0],
                prompt_attention_mask=positive[1],
                prompt_text=prompt,
                token_count=positive[2],
                sequence_length=positive[3],
                hidden_size=positive[4],
                config_source=str(self.text_encoder_path / "config.json"),
                negative_prompt_text=negative_prompt,
                negative_prompt_embeddings=negative_embeddings,
                negative_prompt_attention_mask=negative_attention_mask,
            )

        def close(self) -> None:
            return None

    @dataclass(slots=True)
    class _RuntimeMultimodalPromptEncoder(PromptEncoder):
        text_encoder_path: Path
        tokenizer_path: Path
        processor_path: Path
        _model: Qwen2TextModel
        _tokenizer: _RawTokenizer
        _vision_model: Qwen2VLVisionModel
        _processor_config: Qwen2VLImageProcessorConfig
        _image_token_id: int

        def encode(
            self,
            prompt: str,
            *,
            max_length: int = 512,
            negative_prompt: str | None = None,
            image_paths: tuple[Path, ...] = (),
        ) -> PromptEncodingResult:
            if not image_paths:
                raise ValueError("Qwen-Image image.edit requires at least one image")
            positive = _encode_multimodal_prompt(
                prompt_text=prompt,
                image_paths=image_paths,
                max_length=max_length,
                model=self._model,
                tokenizer=self._tokenizer,
                vision_model=self._vision_model,
                processor_config=self._processor_config,
                image_token_id=self._image_token_id,
            )
            negative_embeddings: object | None = None
            negative_attention_mask: object | None = None
            if negative_prompt is not None:
                negative_embeddings, negative_attention_mask, _, _, _ = (
                    _encode_multimodal_prompt(
                        prompt_text=negative_prompt,
                        image_paths=image_paths,
                        max_length=max_length,
                        model=self._model,
                        tokenizer=self._tokenizer,
                        vision_model=self._vision_model,
                        processor_config=self._processor_config,
                        image_token_id=self._image_token_id,
                    )
                )
            return PromptEncodingResult(
                prompt_embeddings=positive[0],
                prompt_attention_mask=positive[1],
                prompt_text=prompt,
                token_count=positive[2],
                sequence_length=positive[3],
                hidden_size=positive[4],
                config_source=str(self.text_encoder_path / "config.json"),
                negative_prompt_text=negative_prompt,
                negative_prompt_embeddings=negative_embeddings,
                negative_prompt_attention_mask=negative_attention_mask,
            )

        def close(self) -> None:
            return None


def create_prompt_encoder(
    *,
    text_encoder_path: Path,
    tokenizer_path: Path,
    processor_path: Path | None = None,
    task: str = "image.generate",
) -> PromptEncoder:
    if _RUNTIME_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Qwen-Image prompt encoding runtime dependencies failed to import"
        ) from _RUNTIME_IMPORT_ERROR
    if task == "image.edit":
        if processor_path is None:
            raise ValueError("Qwen-Image image.edit requires processor_path")
        raw_tokenizer = AutoTokenizer.from_pretrained(
            str(processor_path),
            local_files_only=True,
            model_max_length=131072,
        )
        if not isinstance(raw_tokenizer, _RawTokenizer):
            raise ValueError(
                "Qwen-Image processor tokenizer does not satisfy the expected callable contract"
            )
        image_token_id = raw_tokenizer.convert_tokens_to_ids(_IMAGE_TOKEN)
        if not isinstance(image_token_id, int) or image_token_id < 0:
            raise ValueError("Qwen-Image processor tokenizer is missing image token id")
        return _RuntimeMultimodalPromptEncoder(
            text_encoder_path=text_encoder_path,
            tokenizer_path=tokenizer_path,
            processor_path=processor_path,
            _model=load_local_qwen2_text_model(text_encoder_path),
            _tokenizer=raw_tokenizer,
            _vision_model=load_local_qwen2_vl_vision_model(text_encoder_path),
            _processor_config=Qwen2VLImageProcessorConfig.from_path(
                processor_path / "preprocessor_config.json"
            ),
            _image_token_id=image_token_id,
        )
    return _RuntimePromptEncoder(
        text_encoder_path=text_encoder_path,
        tokenizer_path=tokenizer_path,
        _model=load_local_qwen2_text_model(text_encoder_path),
        _tokenizer=load_local_text_tokenizer(
            tokenizer_path,
            model_max_length=131072,
            padding_side="right",
        ),
    )


def _encode_prompt(
    *,
    prompt_text: str,
    max_length: int,
    model: Qwen2TextModel,
    tokenizer: TextTokenizer,
) -> tuple[object, object, int, int, int]:
    if max_length <= 0:
        raise ValueError("Qwen-Image max_length must be positive")
    requested_length = min(int(max_length), _TOKENIZER_MAX_LENGTH)
    encoded = tokenizer.encode_text(
        _PROMPT_TEMPLATE.format(prompt_text),
        max_length=_TOKENIZER_MAX_LENGTH + _PROMPT_TEMPLATE_DROP_TOKENS,
        truncation=True,
        # Upstream pads to the batch max and only drops the fixed prompt scaffold
        # after the text encoder pass. Using longest keeps that contract for the
        # first owned single-prompt slice without forcing 1k-token padding.
        padding="longest",
    )
    input_ids = mx.array(encoded.input_ids, dtype=mx.int32)
    attention_mask = mx.array(encoded.attention_mask, dtype=mx.int32)
    _, hidden_states = model(
        input_ids,
        attention_mask=attention_mask,
        return_hidden_states=True,
    )
    final_hidden = hidden_states[-1]
    embeddings, attention_mask = _select_valid_prompt_tokens(
        hidden=final_hidden,
        attention_mask=encoded.attention_mask,
        drop_tokens=_PROMPT_TEMPLATE_DROP_TOKENS,
        max_length=requested_length,
    )
    return (
        embeddings,
        attention_mask,
        int(np.asarray(attention_mask).sum()),
        int(embeddings.shape[1]),
        int(embeddings.shape[2]),
    )


def _select_valid_prompt_tokens(
    *,
    hidden: mx.array,
    attention_mask: np.ndarray,
    drop_tokens: int,
    max_length: int,
) -> tuple[mx.array, mx.array]:
    if attention_mask.ndim != 2 or int(attention_mask.shape[0]) != 1:
        raise ValueError(
            "Qwen-Image prompt encoding currently supports one prompt at a time"
        )
    valid_indices = np.flatnonzero(attention_mask[0])
    if valid_indices.size <= drop_tokens:
        raise ValueError("Qwen-Image prompt encoding produced no valid prompt tokens")
    start = int(valid_indices[0])
    end = int(valid_indices[-1]) + 1
    if not np.all(attention_mask[0, start:end] == 1):
        raise ValueError(
            "Qwen-Image prompt attention_mask must contain one contiguous valid span"
        )
    selected = hidden[0, start:end, :][drop_tokens:]
    if int(selected.shape[0]) <= 0:
        raise ValueError("Qwen-Image prompt encoding produced no post-template tokens")
    selected = selected[:max_length]
    selected = mx.expand_dims(selected, axis=0)
    prompt_attention_mask = mx.ones((1, int(selected.shape[1])), dtype=mx.int32)
    return selected, prompt_attention_mask


def _encode_multimodal_prompt(
    *,
    prompt_text: str,
    image_paths: tuple[Path, ...],
    max_length: int,
    model: Qwen2TextModel,
    tokenizer: _RawTokenizer,
    vision_model: Qwen2VLVisionModel,
    processor_config: Qwen2VLImageProcessorConfig,
    image_token_id: int,
) -> tuple[object, object, int, int, int]:
    if max_length <= 0:
        raise ValueError("Qwen-Image max_length must be positive")
    images = tuple(
        _resize_condition_image(Image.open(path).convert("RGB")) for path in image_paths
    )
    processed = preprocess_qwen2_vl_images(images, config=processor_config)
    base_img_prompt = "".join(
        f"Picture {index + 1}: <|vision_start|>{_IMAGE_TOKEN}<|vision_end|>"
        for index in range(len(images))
    )
    template = _EDIT_PROMPT_TEMPLATE.format(base_img_prompt + prompt_text)
    expanded_text = expand_image_placeholders(
        [template],
        image_grid_thw=processed.image_grid_thw,
        merge_size=processor_config.merge_size,
        image_token=_IMAGE_TOKEN,
    )[0]
    encoded = tokenizer(
        [expanded_text],
        return_tensors="np",
        padding=True,
    )
    input_ids_np = np.asarray(encoded["input_ids"], dtype=np.int32)
    attention_mask_np = np.asarray(encoded["attention_mask"], dtype=np.int32)
    mm_token_type_ids_np = np.zeros_like(input_ids_np, dtype=np.int32)
    mm_token_type_ids_np[input_ids_np == image_token_id] = 1
    image_features = vision_model(
        mx.array(processed.pixel_values, dtype=mx.float32),
        grid_thw=mx.array(processed.image_grid_thw, dtype=mx.int32),
    )
    input_embeddings = np.asarray(
        model.embed_tokens(mx.array(input_ids_np, dtype=mx.int32)).astype(mx.float32)
    )
    mask_flat = input_ids_np.reshape(-1) == image_token_id
    if int(mask_flat.sum()) != int(image_features.shape[0]):
        raise ValueError(
            "Qwen-Image image feature count does not match image token count"
        )
    flat_embeddings = input_embeddings.reshape(-1, input_embeddings.shape[-1])
    flat_embeddings[mask_flat] = np.asarray(image_features.astype(mx.float32))
    position_ids = _compute_image_position_ids(
        input_ids=input_ids_np,
        attention_mask=attention_mask_np,
        image_grid_thw=processed.image_grid_thw,
        mm_token_type_ids=mm_token_type_ids_np,
        spatial_merge_size=processor_config.merge_size,
    )
    _, hidden_states = model(
        mx.array(input_ids_np, dtype=mx.int32),
        input_embeddings=mx.array(flat_embeddings.reshape(input_embeddings.shape)),
        attention_mask=mx.array(attention_mask_np, dtype=mx.int32),
        position_ids=mx.array(position_ids, dtype=mx.int32),
        return_hidden_states=True,
    )
    final_hidden = hidden_states[-1]
    embeddings, attention_mask = _select_valid_prompt_tokens(
        hidden=final_hidden,
        attention_mask=attention_mask_np,
        drop_tokens=_EDIT_PROMPT_TEMPLATE_DROP_TOKENS,
        max_length=min(int(max_length), _TOKENIZER_MAX_LENGTH),
    )
    return (
        embeddings,
        attention_mask,
        int(np.asarray(attention_mask).sum()),
        int(embeddings.shape[1]),
        int(embeddings.shape[2]),
    )


def _resize_condition_image(image: Image.Image) -> Image.Image:
    width, height = image.size
    resized_width, resized_height = _calculate_dimensions(
        _CONDITION_IMAGE_AREA,
        width / height,
    )
    return image.resize((resized_width, resized_height), resample=Resampling.BICUBIC)


def _calculate_dimensions(target_area: int, ratio: float) -> tuple[int, int]:
    width = (target_area * ratio) ** 0.5
    height = width / ratio
    return round(width / 32) * 32, round(height / 32) * 32


def _compute_image_position_ids(
    *,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    image_grid_thw: np.ndarray,
    mm_token_type_ids: np.ndarray,
    spatial_merge_size: int,
) -> np.ndarray:
    position_ids = np.zeros((3, input_ids.shape[0], input_ids.shape[1]), dtype=np.int32)
    image_iter = iter(image_grid_thw.tolist())
    for batch_index in range(input_ids.shape[0]):
        current_input_ids = input_ids[batch_index]
        current_types = mm_token_type_ids[batch_index]
        current_mask = attention_mask[batch_index].astype(bool)
        valid_input_ids = current_input_ids[current_mask]
        valid_types = current_types[current_mask]
        groups: list[tuple[int, int, int]] = []
        start = 0
        while start < len(valid_types):
            end = start + 1
            while end < len(valid_types) and valid_types[end] == valid_types[start]:
                end += 1
            groups.append((int(valid_types[start]), start, end))
            start = end
        current_pos = 0
        llm_pos_ids_list: list[np.ndarray] = []
        for modality_type, start_idx, end_idx in groups:
            if modality_type == 0:
                text_len = end_idx - start_idx
                llm_pos_ids_list.append(
                    np.arange(text_len, dtype=np.int32)[None, :].repeat(3, axis=0)
                    + current_pos
                )
                current_pos += text_len
            else:
                grid_t, grid_h, grid_w = next(image_iter)
                vision_positions = _vision_position_ids(
                    start_position=current_pos,
                    grid_thw=(grid_t, grid_h, grid_w),
                    spatial_merge_size=spatial_merge_size,
                )
                llm_pos_ids_list.append(vision_positions)
                current_pos += max(grid_h, grid_w) // spatial_merge_size
        llm_positions = np.concatenate(llm_pos_ids_list, axis=1)
        position_ids[:, batch_index, current_mask] = llm_positions
        if llm_positions.shape[1] != len(valid_input_ids):
            raise ValueError("Qwen-Image multimodal position ids length mismatch")
    return position_ids


def _vision_position_ids(
    *,
    start_position: int,
    grid_thw: tuple[int, int, int],
    spatial_merge_size: int,
) -> np.ndarray:
    grid_t, grid_h, grid_w = grid_thw
    llm_grid_h = grid_h // spatial_merge_size
    llm_grid_w = grid_w // spatial_merge_size
    image_seq_length = llm_grid_h * llm_grid_w * grid_t
    position_width = np.arange(
        start_position,
        start_position + llm_grid_w,
        dtype=np.int32,
    ).repeat(llm_grid_h * grid_t)
    position_height = np.arange(
        start_position,
        start_position + llm_grid_h,
        dtype=np.int32,
    ).repeat(llm_grid_w * grid_t)
    position_temporal = np.full((image_seq_length,), start_position, dtype=np.int32)
    return np.stack([position_temporal, position_height, position_width], axis=0)
