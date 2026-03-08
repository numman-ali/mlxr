from __future__ import annotations

import mlx.core as mx

from .base import create_attention_mask


class KVCache:
    step = 256

    def __init__(self) -> None:
        self.keys: mx.array | None = None
        self.values: mx.array | None = None
        self.offset = 0

    def update_and_fetch(
        self,
        keys: mx.array,
        values: mx.array,
    ) -> tuple[mx.array, mx.array]:
        previous_offset = self.offset
        if self.keys is None or (previous_offset + int(keys.shape[2])) > int(
            self.keys.shape[2]
        ):
            batch_size, num_heads, _, key_dim = keys.shape
            value_dim = int(values.shape[3])
            num_steps = (self.step + int(keys.shape[2]) - 1) // self.step
            key_shape = (batch_size, num_heads, num_steps * self.step, key_dim)
            value_shape = (batch_size, num_heads, num_steps * self.step, value_dim)
            new_keys = mx.zeros(key_shape, dtype=keys.dtype)
            new_values = mx.zeros(value_shape, dtype=values.dtype)
            if self.keys is not None and self.values is not None:
                if previous_offset % self.step != 0:
                    self.keys = self.keys[..., :previous_offset, :]
                    self.values = self.values[..., :previous_offset, :]
                self.keys = mx.concatenate([self.keys, new_keys], axis=2)
                self.values = mx.concatenate([self.values, new_values], axis=2)
            else:
                self.keys = new_keys
                self.values = new_values
        if self.keys is None or self.values is None:
            raise RuntimeError("KV cache failed to initialize")
        self.offset += int(keys.shape[2])
        self.keys[..., previous_offset : self.offset, :] = keys
        self.values[..., previous_offset : self.offset, :] = values
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]

    def make_mask(
        self,
        sequence_length: int,
        *,
        return_array: bool,
        window_size: int | None,
    ) -> mx.array | str | None:
        if self.offset == 0:
            return create_attention_mask(
                mx.zeros((1, sequence_length, 1), dtype=mx.float32),
                None,
                return_array=return_array,
                window_size=window_size,
            )
        return _create_cache_mask(
            sequence_length,
            offset=self.offset,
            return_array=return_array,
            window_size=window_size,
        )


class RotatingKVCache:
    step = 256

    def __init__(self, max_size: int, keep: int = 0) -> None:
        self.keep = keep
        self.keys: mx.array | None = None
        self.values: mx.array | None = None
        self.offset = 0
        self.max_size = max_size
        self._index = 0

    def _trim(
        self,
        trim_size: int,
        values: mx.array,
        append: mx.array | None = None,
    ) -> mx.array:
        parts: list[mx.array] = []
        if trim_size > 0:
            parts = [
                values[..., : self.keep, :],
                values[..., trim_size + self.keep :, :],
            ]
        else:
            parts = [values]
        if append is not None:
            parts.append(append)
        return mx.concatenate(parts, axis=2)

    def _temporal_order(self, values: mx.array) -> mx.array:
        if self._index == int(values.shape[2]):
            return values
        if self._index < self.offset:
            return mx.concatenate(
                [
                    values[..., : self.keep, :],
                    values[..., self._index :, :],
                    values[..., self.keep : self._index, :],
                ],
                axis=2,
            )
        return values[..., : self._index, :]

    def _update_concat(
        self,
        keys: mx.array,
        values: mx.array,
    ) -> tuple[mx.array, mx.array]:
        if self.keys is None or self.values is None:
            self.keys = keys
            self.values = values
        else:
            self.keys = self._temporal_order(self.keys)
            self.values = self._temporal_order(self.values)
            self._index = int(self.keys.shape[2])
            trim_size = self._index - self.max_size + 1
            self.keys = self._trim(trim_size, self.keys, keys)
            self.values = self._trim(trim_size, self.values, values)
        self.offset += int(keys.shape[2])
        self._index = int(self.keys.shape[2])
        return self.keys, self.values

    def _update_in_place(
        self,
        keys: mx.array,
        values: mx.array,
    ) -> tuple[mx.array, mx.array]:
        batch_size, num_heads, steps, key_dim = keys.shape
        previous_offset = self.offset
        if (
            self.keys is None
            or self.values is None
            or (
                previous_offset >= int(self.keys.shape[2])
                and int(self.keys.shape[2]) < self.max_size
            )
        ):
            value_dim = int(values.shape[3])
            new_size = min(self.step, self.max_size - previous_offset)
            key_shape = (batch_size, num_heads, new_size, key_dim)
            value_shape = (batch_size, num_heads, new_size, value_dim)
            new_keys = mx.zeros(key_shape, dtype=keys.dtype)
            new_values = mx.zeros(value_shape, dtype=values.dtype)
            if self.keys is not None and self.values is not None:
                self.keys = mx.concatenate([self.keys, new_keys], axis=2)
                self.values = mx.concatenate([self.values, new_values], axis=2)
            else:
                self.keys = new_keys
                self.values = new_values
            self._index = previous_offset
        if self.keys is None or self.values is None:
            raise RuntimeError("Rotating KV cache failed to initialize")
        trim_size = int(self.keys.shape[2]) - self.max_size
        if trim_size > 0:
            self.keys = self._trim(trim_size, self.keys)
            self.values = self._trim(trim_size, self.values)
            self._index = self.max_size
        if self._index == self.max_size:
            self._index = self.keep
        self.keys[..., self._index : self._index + steps, :] = keys
        self.values[..., self._index : self._index + steps, :] = values
        self.offset += steps
        self._index += steps
        if self.offset < self.max_size:
            return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]
        return self.keys, self.values

    def update_and_fetch(
        self,
        keys: mx.array,
        values: mx.array,
    ) -> tuple[mx.array, mx.array]:
        if int(keys.shape[2]) == 1:
            return self._update_in_place(keys, values)
        return self._update_concat(keys, values)

    def make_mask(
        self,
        sequence_length: int,
        *,
        return_array: bool,
        window_size: int | None,
    ) -> mx.array | str | None:
        effective_offset = min(self.offset, self.max_size)
        return _create_cache_mask(
            sequence_length,
            offset=effective_offset,
            return_array=return_array,
            window_size=window_size,
        )


def _create_cache_mask(
    sequence_length: int,
    *,
    offset: int,
    return_array: bool,
    window_size: int | None,
) -> mx.array | str | None:
    if window_size is not None:
        from .base import create_causal_mask

        return create_causal_mask(sequence_length, offset, window_size=window_size)
    if sequence_length == 1:
        return None
    if return_array:
        from .base import create_causal_mask

        return create_causal_mask(sequence_length, offset)
    return "causal"
