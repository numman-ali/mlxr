# Python Media Pipeline

Use this reference when the asset should be generated or refined through code rather than hand-built in a design tool.

## Tool choices

Use small, dependable tools with clear roles:

- Use `Pillow` for canvas creation, image placement, masks, gradients, blur, panels, and text layout.
- Use raw `SVG` for simple scalable marks, logos, icons, and lockups.
- Use `ffmpeg` for frame extraction, resizing, crop, concat, poster strips, and short reel assembly.
- Use Python dataclasses or small typed config objects to make the asset recipe explicit.

If the job is mostly layout, Pillow plus ffmpeg is often enough.

## Organize the script

Keep the script readable and deterministic.

Recommended shape:

1. constants for paths, colors, dimensions, and fonts
2. small dataclasses for source assets and clip metadata
3. helper functions for recurring composition tasks
4. one builder function per output asset
5. one `main()` that creates directories and writes outputs

Prefer functions like these:

- `build_poster()`
- `build_slide()`
- `build_card()`
- `build_social_tile()`
- `build_reel_poster()`
- `build_reel()`
- `cover(image, size)`
- `rounded_panel(base, box, fill, radius)`
- `draw_background(image)`
- `font(size)`

## Compose static assets

The common static pipeline is:

1. Create a blank RGB canvas at target size.
2. Draw the background or backdrop glow.
3. Add panels or framing shapes.
4. Place media with cover-crop logic.
5. Add labels, headings, and supporting text.
6. Export the final image.

Core pattern:

```python
from PIL import Image, ImageDraw, ImageFont


def build_asset() -> Image.Image:
    canvas = Image.new("RGB", (1600, 900), "#07111C")
    draw = ImageDraw.Draw(canvas)
    draw_background(canvas)
    paste_media_card(canvas, "hero.png", (820, 100, 1500, 560))
    draw.text((90, 110), "Title", fill="#EAF6FF", font=font(88))
    draw.text((92, 240), "Supporting copy", fill="#A8C5D9", font=font(30))
    return canvas
```

## Use cover-crop for media

When placing images, scale them to cover the target rectangle, then crop from center unless the subject demands a different anchor.

Core pattern:

```python
def cover(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    width, height = size
    source_ratio = image.width / image.height
    target_ratio = width / height
    if source_ratio > target_ratio:
        new_height = height
        new_width = int(height * source_ratio)
    else:
        new_width = width
        new_height = int(width / source_ratio)
    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    left = (new_width - width) // 2
    top = (new_height - height) // 2
    return resized.crop((left, top, left + width, top + height))
```

## Create reusable panel and card helpers

Assets become easier to evolve when cards and panels are their own helpers.

Useful helpers:

- rounded rectangles with optional outline
- shadowed media cards
- blurred glow layers
- tag chips
- frame strips built from several stills

Once these helpers exist, the asset-specific functions become much smaller.

## Render vector marks with SVG

Use SVG when the logo or icon is geometric and should stay sharp across sizes.

Good SVG use cases:

- simple marks
- wordmarks
- lockups
- badges
- icons used in README or slide headers

Write the SVG as a string and save it directly. Keep gradients and shapes simple enough to edit by hand.

## Assemble motion assets with ffmpeg

For a short reel:

1. Render title cards or outro cards as PNGs.
2. Define clips with source path, start time, and duration.
3. Normalize each clip to the same output frame size.
4. Concatenate the pieces into one short export.

Typical flow:

```python
clips = [
    ("loop", "intro.png", 0.0, 1.6),
    ("video", "clip_a.mp4", 1.2, 3.4),
    ("video", "clip_b.mp4", 0.8, 2.8),
    ("loop", "outro.png", 0.0, 1.4),
]
```

Then build an `ffmpeg` command that:

- loops stills for the desired duration
- trims videos at the chosen timestamps
- scales and crops every source to the same size
- concatenates them in order
- exports H.264 with `+faststart`

If the reel needs a poster, extract 2 to 3 good frames and build a composite card around them.

## Export deliberately

Choose formats based on use:

- PNG for crisp layout images and UI-heavy assets
- JPEG when photographic compression is acceptable
- SVG for scalable marks
- MP4 for short motion assets

Export only the variants that matter. Common examples:

- print poster
- presentation slide
- README hero
- social-square variant
- thumbnail or reel poster

## Review the outputs

Do not trust the script because it ran successfully.

Open the exported files and inspect:

- text clipping
- awkward line breaks
- bad image crops
- muddy contrast
- oversized safe margins
- timing or pacing problems in motion

If the environment allows it, run a multimodal review pass on short videos or reels to catch weak clip choices and sequence drift.
