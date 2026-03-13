---
name: media-asset-composer
description: Design static and light-motion media assets such as posters, slides, cards, flyers, invitations, social graphics, thumbnails, hero images, banners, certificates, README art, and short promo reels. Use when Codex needs to turn text, images, icons, or video into polished layout-driven visuals, especially through a reproducible Python, Pillow, SVG, or ffmpeg workflow.
---

# Media Asset Composer

Create polished media assets with a scripted, repeatable workflow.

Use this skill when the task is to make a visual deliverable feel intentional: a poster for a children's play, a government notice, a birthday card, a conference slide, a social graphic, a launch hero, or a short reel poster.

## Default workflow

1. Define the brief.
   Identify the audience, purpose, delivery surface, required copy, deadline, and emotional tone. If the user does not specify all of them, infer the most important missing constraints from context.
2. Lock the frame.
   Choose dimensions, orientation, safe margins, and export format before composing. Distinguish between print, slide, web, social, and video-poster requirements.
3. Choose one visual system.
   Commit to one palette, one type hierarchy, one shape language, and one compositional mood. Avoid mixing multiple unrelated styles in the same asset.
4. Gather the source material.
   Use the strongest available media: illustrations, photos, screenshots, icons, logos, frames, or generated images. Prefer fewer strong sources over many weak ones.
5. Compose with code.
   Prefer a focused Python script or deterministic pipeline over manual one-off editing. Build backgrounds, panels, text blocks, image placements, and export variants from code.
6. Review like a designer.
   Check legibility, balance, contrast, hierarchy, cropping, and emotional fit. If motion is involved, review sequencing and pacing as well.
7. Export the real deliverable.
   Write the final asset files, keep the source script, and produce the variants actually needed by the surface: print, retina web, thumbnail, poster, or presentation size.

## Core rules

- Put the message first. Decoration is there to support the communication, not replace it.
- Match the style to the audience. A government poster, a birthday card, and a children’s play poster should not feel like the same design with different text.
- Keep the hierarchy obvious. The viewer should know what to read first, second, and third.
- Make assets reproducible. If the job may need revision, script it.
- Use high-resolution sources whenever possible. Low-quality input tends to look worse once it is framed and enlarged.
- Validate the actual output files, not just the source code. Open the exported asset and inspect it directly.
- Use a multimodal reviewer when available for motion assets or when semantic visual quality matters.

## What to load

- Load [references/asset-playbook.md](references/asset-playbook.md) for audience-aware design heuristics and layout recipes.
- Load [references/python-media-pipeline.md](references/python-media-pipeline.md) for the reusable scripted workflow using Python, Pillow, SVG, and ffmpeg.

## Deliverable standard

Aim for assets that do four things:

- communicate clearly
- feel visually deliberate
- fit the audience and context
- remain easy to revise
