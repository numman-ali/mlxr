# model-family-ltx

This package is the first family-adapter target for MLXR.

First implementation slice:

- local `LTX-2.3 Fast` text-to-video
- local `LTX-2.3 Fast` image-to-video
- local `LTX-2.3 Fast` multi-image keyed image-to-video through the current
  `video.condition.image` row
- local `LTX-2.3 Fast` audio-conditioned video with resolved runtime audio handles
- strict-local MLX-native prompt encoding
- artifactized fast-path assets with validated component payloads
- resolved image-conditioning handles in the runtime worker
- resolved audio-conditioning handles in the runtime worker
- runtime-managed H.264/AAC `mp4` and `wav` outputs with per-stage memory and timing telemetry

Current measured caveat:

- the repo-owned strict-local MLX prompt encoder and distilled two-stage generation backend now complete direct artifact-backed `LTX-2.3` T2V and I2V smoke tests against the real upstream weights
- the bridge now fails closed on prompt/generation contract drift, 22B transformer feature mismatches, missing VAE latent statistics, and unsupported x2 upsampler layouts instead of silently producing low-fidelity output
- the current fixed-seed dog ladder now passes a clear-dog `384x224 / 17f / 24fps` rung and a coherent `768x512 / 33f / 24fps` rung against the real Hugging Face-converted artifact set
- the AV bridge now exports checkpoint-backed audio as muxed `mp4` or standalone `wav`, and the current repo-owned MLX audio path now includes the checkpoint’s full BWE wrapper instead of stopping at the `AMP1` base vocoder
- the current `video.condition.audio` slice is real on the artifact-backed bridge, preserves the resolved reference audio into the output path, and now clears strict Gemini review on both a natural-audio dog row and a non-dog anime row through the real `mlxr` product path; it is now a quality-promoted conditioned-audio row, while broader reference-video input and LoRA-driven controls remain tracked separately in the capability matrix
- the owned runtime now also has a real full-checkpoint `one_stage` lane for the
  `ltx-2.3-22b-dev.safetensors` artifact path, and the artifact contract no
  longer forces that dev row to carry the x2 upsampler just because the fast
  row does; that lane is implemented and workflow-selectable, and it now has a
  first meaningful `6s / 768x448 / 145f` runtime receipt, but the visual
  result is still much weaker than two-stage and not yet promotable
- the owned runtime now also has a real full-checkpoint `two_stage` lane for
  `dev` artifacts that carry both the x2 upsampler and the official distilled
  LoRA; that lane is implemented and workflow-selectable, and it now has a
  first meaningful `6s / 768x448 / 145f` runtime receipt, but Gemini still
  scores that receipt as a mismatch, so the row is not yet promoted
- the owned runtime now also has a real full-checkpoint `two_stage_hq` lane
  for `dev` artifacts that carry both the x2 upsampler and the official
  distilled LoRA; that lane is implemented and workflow-selectable, and it now
  has a first meaningful `6s / 768x448 / 145f` runtime receipt, but Gemini
  still scores that receipt as a mismatch, so the row is not yet promoted
- the owned runtime now also has a real `video.retake` lane on the canonical
  CLI/runtime path for artifacts that support video generation; it validates
  the source-video and retake-window contract through the shared workflow,
  server, and adapter surfaces, and it now has a first meaningful
  `6s / 768x448 / 145f` receipt on the fast path, but the current edit
  semantics are still too weak for promotion
- the owned runtime now also has a real `video.interpolate` lane on the full
  `dev` two-stage artifact path; it follows the official keyframe row more
  closely than the older keyed-image bridge by treating every keyed image,
  including frame zero, as a guiding keyframe instead of replacing the first
  latent, and it now has a first meaningful `6s / 768x448 / 145f` receipt, but
  the current keyframe adherence is still too weak for promotion
- the owned runtime now also has a real `video.condition.video` lane on the
  fast artifact path; it follows the official `ICLoraPipeline` semantics more
  closely by requiring exactly one reference video plus one IC-LoRA input and
  running that control in the owned distilled backend; the current
  `union_ic_lora` slice now has a meaningful `6s / 768x448 / 145f` receipt
  with a Gemini `match`, while `motion_track_control` is also real on the same
  surface but currently weaker and not yet promoted on its own
- the canonical CLI/runtime path now also supports multiple keyed image
  references on the current distilled `video.condition.image` row; the first
  real first/last-frame receipt is `tmp/manual-runs/20260309T2324Z-zimage-ltx-keyframe-samurai-to-fire.mp4`,
  which proves the surface is real without over-claiming official interpolation
  or IC-LoRA-style strong control
- the current promoted runtime path now passes caller-authored prompts through verbatim instead of composing extra text or synthesized negative prompts in the workflow layer; conditioning references remain real model inputs, while prompt-authoring behavior stays out of the promoted inference path until it earns a deliberate client-layer design
- the current first promoted 10-second text-first showcase pack is now the score-friendly set under `tmp/showcase-runs/showcase-pack-20260308.json`: anime neon chase, stop-motion workshop, vintage nostalgic street, cinematic spacewalk, and noir rainy city all clear the current visual/audio review bar at `384x224 / 241f / 24fps`
- the current best-available promoted five-clip pack is `tmp/showcase-runs/showcase-pack-20260308-best-available.json`, which swaps in the recovered natural-audio dog clip for one of the older stylized rows
- the natural-audio showcase story is still separate: the 10-second dog scene is still green, and the newer showcase-runner shaping slice now clearly rescues `heron_marsh_documentary` at the same 10-second `384x224 / 241f` rung; broader no-music wins such as `basketball_court_dusk` and `cafe_sidewalk_human` also exist in the current repo receipts, but passive animal ambience, vintage-natural scenes, and the current quiet human CLI rows still need more work
- the remaining truth gap is broader capability completion plus recommended-resolution and HQ-profile validation once that capability matrix is green

Out of first slice:

- recommended-resolution and HQ profile validation
- reference-video conditioning
- IC-LoRA validation and promotion
- real two-stage receipts and promotion on the full dev checkpoint
- real retake receipts and promotion
- real interpolation receipts and promotion
- real HQ receipts and promotion on the full dev checkpoint with the distilled LoRA
- advanced Comfy node parity

Canonical CLI examples:

```bash
uv run mlxr generate \
  --model-id ltx-2.3-fast-cli-local \
  --prompt "Samurai in a rainy alley, cinematic fire growing over time." \
  --width 768 \
  --height 512 \
  --num-frames 33 \
  --fps 24 \
  --image first_frame.png \
  --keyframe-image last_frame.png 32 0.75 \
  --artifact-format mp4 \
  --wait \
  --export-path keyed_i2v.mp4
```

```bash
uv run mlxr generate \
  --model-id ltx-2.3-fast-cli-local \
  --prompt "Quiet dog in a park, natural ambience and bark timing preserved." \
  --width 768 \
  --height 512 \
  --num-frames 145 \
  --fps 24 \
  --audio bark.wav \
  --artifact-format mp4 \
  --wait \
  --export-path conditioned_audio.mp4
```
