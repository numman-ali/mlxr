"""Regenerate the README brand, poster, gallery, and reel assets."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
BRAND_DIR = REPO_ROOT / "docs" / "assets" / "brand"
README_ASSETS_DIR = REPO_ROOT / "docs" / "assets" / "readme"
TMP_DIR = REPO_ROOT / "tmp" / "readme-assets"
FONT_PATH = Path("/System/Library/Fonts/Supplemental/Futura.ttc")

DEEP_INK = "#07111C"
MIDNIGHT = "#0C1E2C"
OCEAN = "#12324B"
ICE = "#EAF6FF"
MIST = "#A8C5D9"
CYAN = "#5CD6FF"
TEAL = "#1FC6B2"
EMBER = "#FF8A45"
MAGENTA = "#FF4F78"
SOFT_GOLD = "#FFD27A"


@dataclass(frozen=True, slots=True)
class ShowcaseImage:
    label: str
    title: str
    path: Path


@dataclass(frozen=True, slots=True)
class VideoClip:
    label: str
    source: Path
    start_seconds: float
    duration_seconds: float


SHOWCASE_IMAGES = (
    ShowcaseImage(
        label="Qwen-Image",
        title="High-resolution cinematic generation",
        path=REPO_ROOT
        / "tmp"
        / "showcase-runs"
        / "qwen-official-eval-20260311T190800Z"
        / "floating_city_wuli4_1664x928.png",
    ),
    ShowcaseImage(
        label="FLUX.2",
        title="Atmospheric still-image generation",
        path=REPO_ROOT
        / "tmp"
        / "showcase-runs"
        / "flux2-showcase-flat-20260310T100316Z"
        / "04_monorail_rainforest.png",
    ),
    ShowcaseImage(
        label="Z-Image",
        title="Prompt-first surreal composition",
        path=REPO_ROOT
        / "tmp"
        / "showcase-runs"
        / "zimage-showcase-20260309T190247Z"
        / "01_storm_atlas_library.png",
    ),
    ShowcaseImage(
        label="Qwen-Image Edit",
        title="Fast local image editing",
        path=REPO_ROOT
        / "tmp"
        / "showcase-runs"
        / "qwen-official-eval-20260311T221605Z"
        / "kenji_edit_lightning4_1344x768.png",
    ),
    ShowcaseImage(
        label="FLUX.2",
        title="Detailed landscape rendering",
        path=REPO_ROOT
        / "tmp"
        / "showcase-runs"
        / "flux2-showcase-flat-20260310T100316Z"
        / "05_northern_lighthouse.png",
    ),
    ShowcaseImage(
        label="Z-Image",
        title="Portrait and worldbuilding",
        path=REPO_ROOT
        / "tmp"
        / "showcase-runs"
        / "zimage-showcase-20260309T190247Z"
        / "03_martian_greenhouse_portrait.png",
    ),
)

REEL_CLIPS = (
    VideoClip(
        label="LTX",
        source=REPO_ROOT
        / "tmp"
        / "manual-runs"
        / "20260308T065915Z-anime-japanese-dialogue-768"
        / "anime-japanese-dialogue-768_768x512_241f.mp4",
        start_seconds=1.2,
        duration_seconds=3.4,
    ),
    VideoClip(
        label="LTX",
        source=REPO_ROOT
        / "tmp"
        / "manual-runs"
        / "20260308T071120Z-oil-news-live-768"
        / "oil-news-live-768_768x512_241f.mp4",
        start_seconds=1.0,
        duration_seconds=3.4,
    ),
    VideoClip(
        label="LTX",
        source=REPO_ROOT
        / "tmp"
        / "manual-runs"
        / "20260312T1619Z-ltx-fast-union-ic-lora-6s"
        / "ltx_fast_union_ic_lora_768x448_145f.mp4",
        start_seconds=0.6,
        duration_seconds=3.2,
    ),
)


def main() -> None:
    BRAND_DIR.mkdir(parents=True, exist_ok=True)
    README_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    _write_logo_assets()
    _build_showcase_grid()
    hero = _build_hero()
    hero.save(README_ASSETS_DIR / "hero.png", optimize=True)
    poster = _build_reel_poster()
    poster.save(README_ASSETS_DIR / "reel-poster.png", optimize=True)
    _build_reel()


def _write_logo_assets() -> None:
    mark_svg = _mark_svg()
    (BRAND_DIR / "mlxr-mark.svg").write_text(mark_svg, encoding="utf-8")
    (BRAND_DIR / "mlxr-logo-dark.svg").write_text(
        _logo_svg(text_fill=DEEP_INK, subtitle_fill=MIDNIGHT), encoding="utf-8"
    )
    (BRAND_DIR / "mlxr-logo-light.svg").write_text(
        _logo_svg(text_fill=ICE, subtitle_fill=MIST), encoding="utf-8"
    )


def _mark_svg() -> str:
    return f"""<svg width="512" height="512" viewBox="0 0 512 512" fill="none" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="tile" x1="72" y1="56" x2="432" y2="464" gradientUnits="userSpaceOnUse">
      <stop stop-color="{MIDNIGHT}"/>
      <stop offset="1" stop-color="{DEEP_INK}"/>
    </linearGradient>
    <radialGradient id="glow" cx="0" cy="0" r="1" gradientUnits="userSpaceOnUse" gradientTransform="translate(256 256) rotate(90) scale(210)">
      <stop stop-color="#173956"/>
      <stop offset="1" stop-color="#173956" stop-opacity="0"/>
    </radialGradient>
  </defs>
  <rect x="56" y="56" width="400" height="400" rx="112" fill="url(#tile)"/>
  <rect x="56" y="56" width="400" height="400" rx="112" fill="url(#glow)"/>
  <path d="M138 350L216 162" stroke="{CYAN}" stroke-width="54" stroke-linecap="round"/>
  <path d="M256 350V162" stroke="{TEAL}" stroke-width="54" stroke-linecap="round"/>
  <path d="M374 350L296 162" stroke="{EMBER}" stroke-width="54" stroke-linecap="round"/>
  <path d="M156 186L356 328" stroke="{MAGENTA}" stroke-width="42" stroke-linecap="round"/>
  <circle cx="256" cy="256" r="20" fill="{SOFT_GOLD}"/>
</svg>
"""


def _logo_svg(*, text_fill: str, subtitle_fill: str) -> str:
    return f"""<svg width="1180" height="320" viewBox="0 0 1180 320" fill="none" xmlns="http://www.w3.org/2000/svg">
  <g transform="translate(0 0)">
    <defs>
      <linearGradient id="tile" x1="35" y1="28" x2="218" y2="236" gradientUnits="userSpaceOnUse">
        <stop stop-color="{MIDNIGHT}"/>
        <stop offset="1" stop-color="{DEEP_INK}"/>
      </linearGradient>
      <radialGradient id="glow" cx="0" cy="0" r="1" gradientUnits="userSpaceOnUse" gradientTransform="translate(126 126) rotate(90) scale(110)">
        <stop stop-color="#173956"/>
        <stop offset="1" stop-color="#173956" stop-opacity="0"/>
      </radialGradient>
    </defs>
    <rect x="28" y="28" width="196" height="196" rx="56" fill="url(#tile)"/>
    <rect x="28" y="28" width="196" height="196" rx="56" fill="url(#glow)"/>
    <path d="M70 174L108 82" stroke="{CYAN}" stroke-width="28" stroke-linecap="round"/>
    <path d="M126 174V82" stroke="{TEAL}" stroke-width="28" stroke-linecap="round"/>
    <path d="M182 174L144 82" stroke="{EMBER}" stroke-width="28" stroke-linecap="round"/>
    <path d="M78 94L174 162" stroke="{MAGENTA}" stroke-width="22" stroke-linecap="round"/>
    <circle cx="126" cy="126" r="10" fill="{SOFT_GOLD}"/>
  </g>
  <text x="278" y="142" fill="{text_fill}" style="font-family: 'Futura', 'Trebuchet MS', 'Avenir Next', sans-serif; font-size: 108px; font-weight: 700; letter-spacing: 10px;">MLXR</text>
  <text x="282" y="210" fill="{subtitle_fill}" style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; font-size: 34px; font-weight: 500; letter-spacing: 2px;">local-first MLX runtime for Apple Silicon</text>
</svg>
"""


def _build_hero() -> Image.Image:
    image = Image.new("RGB", (2000, 1125), DEEP_INK)
    draw = ImageDraw.Draw(image)
    _draw_background(image, centers=((250, 240), (1500, 270), (1600, 860)))

    display = _font(78)
    display_bold = _font(78)
    body = _font(30)
    chip = _font(22)
    mono = _font(26)
    small = _font(26)

    _draw_mark(image, (94, 78, 238, 222))
    draw.text((272, 82), "MLXR", fill=ICE, font=display)
    draw.text(
        (272, 174),
        "One runtime for\nlocal video,\nimage generation,\nand editing.",
        fill=ICE,
        font=display_bold,
        spacing=8,
    )
    draw.text(
        (96, 548),
        "A local-first generative runtime built for Apple Silicon.\n"
        "Thin clients. Truthful capability reporting. Provenance preserved end to end.",
        fill=MIST,
        font=body,
        spacing=10,
    )

    chips = (
        "Apple Silicon",
        "CLI + daemon",
        "Video + image",
        "Provider-aware",
        "Open-source runway",
    )
    x = 96
    y = 706
    for label in chips:
        width = int(draw.textlength(label, font=chip)) + 44
        _rounded_panel(image, (x, y, x + width, y + 50), MIDNIGHT, 24, outline=OCEAN)
        ImageDraw.Draw(image).text((x + 22, y + 12), label, fill=ICE, font=chip)
        x += width + 14

    _draw_terminal_card(
        image=image,
        box=(96, 750, 790, 1022),
        title="Quickstart",
        lines=(
            "$ uv tool install mlxr",
            "$ mlxr models list",
            "$ mlxr models install ltx-2.3-fast-local",
            "$ mlxr generate --model-id ltx-2.3-fast-local \\",
            '    --prompt "golden retriever in a park" \\',
            "    --wait --export-path out.mp4",
        ),
        title_font=small,
        body_font=mono,
    )

    _paste_media_card(
        image,
        SHOWCASE_IMAGES[0].path,
        (1100, 110, 1880, 590),
        label="Qwen-Image",
        title="wide local generation",
        label_color=CYAN,
    )
    _paste_media_card(
        image,
        SHOWCASE_IMAGES[3].path,
        (1440, 620, 1880, 1020),
        label="Qwen-Image Edit",
        title="fast stylized editing",
        label_color=MAGENTA,
    )
    _paste_media_card(
        image,
        SHOWCASE_IMAGES[1].path,
        (980, 666, 1380, 1010),
        label="FLUX.2",
        title="still-image generation",
        label_color=EMBER,
    )
    return image


def _build_showcase_grid() -> None:
    image = Image.new("RGB", (2200, 1420), DEEP_INK)
    draw = ImageDraw.Draw(image)
    _draw_background(image, centers=((320, 260), (1860, 260), (1100, 1180)))
    title_font = _font(82)
    body_font = _font(28)
    draw.text((90, 70), "MLXR showcase", fill=ICE, font=title_font)
    draw.text(
        (92, 156),
        "Real outputs from the current runtime across video and image families.",
        fill=MIST,
        font=body_font,
    )

    boxes = (
        (90, 250, 760, 660),
        (810, 250, 1480, 660),
        (1530, 250, 2110, 660),
        (90, 720, 760, 1330),
        (810, 720, 1480, 1330),
        (1530, 720, 2110, 1330),
    )
    items = (
        SHOWCASE_IMAGES[0],
        SHOWCASE_IMAGES[1],
        SHOWCASE_IMAGES[2],
        SHOWCASE_IMAGES[3],
        SHOWCASE_IMAGES[4],
        ShowcaseImage(
            label="LTX",
            title="conditioned motion and scene control",
            path=_video_strip(
                REEL_CLIPS[2].source,
                "fox_strip.png",
                frame_times=(0.5, 2.5, 5.0),
            ),
        ),
    )
    colors = (CYAN, MAGENTA, EMBER, MAGENTA, SOFT_GOLD, TEAL)
    for box, item, color in zip(boxes, items, colors, strict=True):
        _paste_media_card(
            image,
            item.path,
            box,
            label=item.label,
            title=item.title,
            label_color=color,
        )
    image.save(README_ASSETS_DIR / "showcase-grid.png", optimize=True)


def _build_reel_poster() -> Image.Image:
    base = Image.new("RGB", (1600, 900), DEEP_INK)
    _draw_background(base, centers=((220, 250), (1260, 220), (1240, 700)))
    strip = _video_strip(
        REEL_CLIPS[0].source,
        "poster_strip.png",
        frame_times=(1.2, 4.2, 7.2),
    )
    _paste_media_card(
        base,
        strip,
        (760, 96, 1520, 528),
        label="LTX",
        title="watch the local runtime reel",
        label_color=CYAN,
    )
    _paste_media_card(
        base,
        SHOWCASE_IMAGES[1].path,
        (900, 548, 1520, 832),
        label="FLUX.2",
        title="still-image generation",
        label_color=MAGENTA,
    )
    _draw_mark(base, (90, 92, 210, 212))
    draw = ImageDraw.Draw(base)
    draw.text((248, 108), "MLXR reel", fill=ICE, font=_font(82))
    draw.text(
        (92, 242),
        "A short cut of the current local-first runtime in motion.\n"
        "Video, stills, edits, and owned surfaces\n"
        "on Apple Silicon.",
        fill=MIST,
        font=_font(30),
        spacing=10,
    )
    play_center = (1260, 335)
    draw.ellipse(
        (
            play_center[0] - 72,
            play_center[1] - 72,
            play_center[0] + 72,
            play_center[1] + 72,
        ),
        fill=(7, 17, 28, 180),
        outline=ICE,
        width=4,
    )
    draw.polygon(
        (
            play_center[0] - 18,
            play_center[1] - 28,
            play_center[0] - 18,
            play_center[1] + 28,
            play_center[0] + 36,
            play_center[1],
        ),
        fill=ICE,
    )
    return base


def _build_reel() -> None:
    intro_path = TMP_DIR / "reel-intro.png"
    outro_path = TMP_DIR / "reel-outro.png"
    _build_reel_title_card(
        "MLXR",
        "Local-first generative runtime for Apple Silicon",
        intro_path,
    )
    _build_reel_title_card(
        "One runtime.\nMany thin surfaces.",
        "CLI, daemon, video, stills, editing, provenance.",
        outro_path,
    )

    inputs: list[str] = []
    filter_parts: list[str] = []
    concat_inputs: list[str] = []

    intro_duration = 1.6
    outro_duration = 1.6
    still_duration = 1.8

    sources = (
        ("loop", intro_path, 0.0, intro_duration),
        (
            "video",
            REEL_CLIPS[0].source,
            REEL_CLIPS[0].start_seconds,
            REEL_CLIPS[0].duration_seconds,
        ),
        (
            "video",
            REEL_CLIPS[1].source,
            REEL_CLIPS[1].start_seconds,
            REEL_CLIPS[1].duration_seconds,
        ),
        ("loop", SHOWCASE_IMAGES[2].path, 0.0, still_duration),
        (
            "video",
            REEL_CLIPS[2].source,
            REEL_CLIPS[2].start_seconds,
            REEL_CLIPS[2].duration_seconds,
        ),
        ("loop", outro_path, 0.0, outro_duration),
    )

    for index, (kind, path, start_seconds, duration) in enumerate(sources):
        if kind == "loop":
            inputs.extend(["-loop", "1", "-t", f"{duration:.2f}", "-i", str(path)])
        else:
            inputs.extend(
                [
                    "-ss",
                    f"{start_seconds:.2f}",
                    "-t",
                    f"{duration:.2f}",
                    "-i",
                    str(path),
                ]
            )
        filter_parts.append(
            f"[{index}:v]scale=1280:720:force_original_aspect_ratio=increase,"
            f"crop=1280:720,setsar=1,format=yuv420p[v{index}]"
        )
        concat_inputs.append(f"[v{index}]")

    filter_parts.append(
        f"{''.join(concat_inputs)}concat=n={len(sources)}:v=1:a=0[vout]"
    )

    output_path = README_ASSETS_DIR / "mlxr-reel.mp4"
    command = [
        "ffmpeg",
        "-y",
        *inputs,
        "-filter_complex",
        ";".join(filter_parts),
        "-map",
        "[vout]",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    subprocess.run(command, check=True)


def _build_reel_title_card(title: str, subtitle: str, destination: Path) -> None:
    image = Image.new("RGB", (1280, 720), DEEP_INK)
    _draw_background(image, centers=((220, 180), (1020, 180), (960, 620)))
    _draw_mark(image, (80, 80, 180, 180))
    draw = ImageDraw.Draw(image)
    draw.text((220, 110), title, fill=ICE, font=_font(84), spacing=8)
    draw.text((86, 274), subtitle, fill=MIST, font=_font(34), spacing=10)
    image.save(destination)


def _video_strip(
    source: Path, filename: str, *, frame_times: tuple[float, float, float]
) -> Path:
    destination = TMP_DIR / filename
    frames: list[Image.Image] = []
    for index, timestamp in enumerate(frame_times):
        frame_path = TMP_DIR / f"{destination.stem}-{index}.jpg"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                f"{timestamp:.2f}",
                "-i",
                str(source),
                "-frames:v",
                "1",
                str(frame_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        frames.append(Image.open(frame_path).convert("RGB"))
    widths = [frame.width for frame in frames]
    heights = [frame.height for frame in frames]
    strip = Image.new("RGB", (sum(widths), max(heights)), DEEP_INK)
    x = 0
    for frame in frames:
        strip.paste(frame, (x, 0))
        x += frame.width
    strip.save(destination, quality=95)
    return destination


def _draw_background(
    image: Image.Image, *, centers: tuple[tuple[int, int], ...]
) -> None:
    background = Image.new("RGBA", image.size, DEEP_INK)
    gradient = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(gradient)
    for center, color in zip(
        centers,
        ((25, 198, 178, 120), (92, 214, 255, 110), (255, 79, 120, 90)),
        strict=True,
    ):
        for radius in range(520, 0, -8):
            alpha = int(color[3] * (radius / 520) ** 2)
            tint = color[:3] + (alpha,)
            draw.ellipse(
                (
                    center[0] - radius,
                    center[1] - radius,
                    center[0] + radius,
                    center[1] + radius,
                ),
                fill=tint,
            )
    gradient = gradient.filter(ImageFilter.GaussianBlur(40))
    composite = Image.alpha_composite(background, gradient)
    image.paste(composite.convert("RGB"))

    grid = Image.new("RGBA", image.size, (0, 0, 0, 0))
    grid_draw = ImageDraw.Draw(grid)
    for x in range(0, image.width, 72):
        grid_draw.line((x, 0, x, image.height), fill=(255, 255, 255, 12), width=1)
    for y in range(0, image.height, 72):
        grid_draw.line((0, y, image.width, y), fill=(255, 255, 255, 12), width=1)
    image.paste(Image.alpha_composite(image.convert("RGBA"), grid).convert("RGB"))


def _draw_mark(image: Image.Image, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    tile = Image.new("RGBA", (x1 - x0, y1 - y0), (0, 0, 0, 0))
    draw = ImageDraw.Draw(tile)
    radius = int((x1 - x0) * 0.28)
    draw.rounded_rectangle(
        (0, 0, tile.width, tile.height), radius=radius, fill=MIDNIGHT
    )

    glow = Image.new("RGBA", tile.size, (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow)
    glow_draw.ellipse(
        (-tile.width * 0.2, -tile.height * 0.1, tile.width * 1.2, tile.height * 1.3),
        fill=(22, 57, 86, 220),
    )
    glow = glow.filter(ImageFilter.GaussianBlur(28))
    tile = Image.alpha_composite(tile, glow)
    draw = ImageDraw.Draw(tile)

    def line(
        start: tuple[float, float], end: tuple[float, float], color: str, width: int
    ) -> None:
        draw.line(
            (
                start[0] * tile.width,
                start[1] * tile.height,
                end[0] * tile.width,
                end[1] * tile.height,
            ),
            fill=color,
            width=width,
            joint="curve",
        )

    stroke = max(8, tile.width // 9)
    cross = max(8, tile.width // 11)
    line((0.24, 0.72), (0.46, 0.24), CYAN, stroke)
    line((0.5, 0.72), (0.5, 0.24), TEAL, stroke)
    line((0.76, 0.72), (0.54, 0.24), EMBER, stroke)
    line((0.28, 0.3), (0.72, 0.62), MAGENTA, cross)
    center = (tile.width // 2, tile.height // 2)
    r = max(5, tile.width // 16)
    draw.ellipse(
        (center[0] - r, center[1] - r, center[0] + r, center[1] + r), fill=SOFT_GOLD
    )

    mask = Image.new("L", tile.size, 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rounded_rectangle(
        (0, 0, tile.width, tile.height), radius=radius, fill=255
    )
    tile.putalpha(mask)
    image.paste(tile, (x0, y0), tile)


def _rounded_panel(
    base: Image.Image,
    box: tuple[int, int, int, int],
    fill: str,
    radius: int,
    *,
    outline: str | None = None,
) -> None:
    panel = Image.new("RGBA", (box[2] - box[0], box[3] - box[1]), (0, 0, 0, 0))
    draw = ImageDraw.Draw(panel)
    draw.rounded_rectangle(
        (0, 0, panel.width - 1, panel.height - 1),
        radius=radius,
        fill=fill,
        outline=outline,
        width=2 if outline else 0,
    )
    base.paste(panel, (box[0], box[1]), panel)


def _paste_media_card(
    base: Image.Image,
    source_path: Path,
    box: tuple[int, int, int, int],
    *,
    label: str,
    title: str,
    label_color: str,
) -> None:
    radius = 34
    shadow = Image.new("RGBA", base.size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    shadow_draw.rounded_rectangle(
        (box[0] + 18, box[1] + 26, box[2] + 18, box[3] + 26),
        radius=radius,
        fill=(0, 0, 0, 120),
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(24))
    base.paste(Image.alpha_composite(base.convert("RGBA"), shadow).convert("RGB"))

    card = Image.new("RGBA", (box[2] - box[0], box[3] - box[1]), (0, 0, 0, 0))
    content = Image.open(source_path).convert("RGB")
    fitted = _cover(content, (card.width, card.height))
    card.paste(fitted, (0, 0))

    overlay = Image.new("RGBA", card.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rounded_rectangle(
        (0, 0, card.width - 1, card.height - 1),
        radius=radius,
        outline=(255, 255, 255, 40),
        width=2,
    )
    overlay_draw.rectangle(
        (0, int(card.height * 0.67), card.width, card.height),
        fill=(6, 12, 20, 162),
    )
    overlay_draw.rounded_rectangle(
        (24, 24, 190, 72),
        radius=20,
        fill=(7, 17, 28, 175),
    )
    label_font = _font(22)
    title_font = _font(28)
    overlay_draw.text((42, 38), label, fill=label_color, font=label_font)
    overlay_draw.text((28, int(card.height * 0.74)), title, fill=ICE, font=title_font)

    card = Image.alpha_composite(card.convert("RGBA"), overlay)
    mask = Image.new("L", card.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, card.width, card.height), radius=radius, fill=255
    )
    card.putalpha(mask)
    base.paste(card, (box[0], box[1]), card)


def _draw_terminal_card(
    *,
    image: Image.Image,
    box: tuple[int, int, int, int],
    title: str,
    lines: tuple[str, ...],
    title_font: ImageFont.FreeTypeFont,
    body_font: ImageFont.FreeTypeFont,
) -> None:
    _rounded_panel(image, box, MIDNIGHT, 36, outline=OCEAN)
    draw = ImageDraw.Draw(image)
    draw.text((box[0] + 34, box[1] + 22), title, fill=ICE, font=title_font)
    dot_y = box[1] + 28
    for index, color in enumerate((MAGENTA, SOFT_GOLD, TEAL)):
        x = box[2] - 112 + index * 26
        draw.ellipse((x, dot_y, x + 16, dot_y + 16), fill=color)
    y = box[1] + 72
    for line in lines:
        fill = CYAN if line.startswith("$") else MIST
        draw.text((box[0] + 34, y), line, fill=fill, font=body_font)
        y += 32


def _cover(image: Image.Image, size: tuple[int, int]) -> Image.Image:
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


def _font(size: int) -> ImageFont.FreeTypeFont:
    if not FONT_PATH.exists():
        raise FileNotFoundError(f"Missing design font at {FONT_PATH}")
    return ImageFont.truetype(str(FONT_PATH), size=size)


if __name__ == "__main__":
    main()
