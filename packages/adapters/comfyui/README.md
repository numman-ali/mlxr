# adapter-comfyui

This package is the planned thin runtime-backed adapter for ComfyUI.

The goal is to expose `MLXR` capabilities inside ComfyUI without duplicating
inference logic or teaching the adapter to own runtime orchestration.

The first node set should focus on:

- fast text-to-video
- fast image-to-video
- runtime model and capability selection

Advanced CUDA-centric flows, direct transformer patching, or broader parity
surfaces are intentionally later work.
