# Source Catalog

Last refreshed: 2026-03-06

This catalog records the sources that actually shaped the rewrite. “Official” means highest-authority upstream source. It does not mean part of this repository.

## Classification Notes

- `official`: first-party repo, docs, or model source maintained by the original project authors
- `ecosystem`: active community project used as a reference input
- `system`: Apple or security guidance
- `freshness risk`: how likely the source is to drift quickly

## Core Sources

| Source | Type | Last verified | Freshness risk | What it proves | Action relevance |
| --- | --- | --- | --- | --- | --- |
| `ml-explore/mlx` local clone `be872ebd` | official repo | 2026-03-06 | high | current official MLX baseline | core runtime |
| `mlx` compile docs | official docs | 2026-03-06 | medium | shape and purity caveats | scheduler and build cache |
| `mlx.core.exporter` docs | official docs | 2026-03-06 | medium | exporter or importer remains experimental | artifact portability |
| `mlx` extensions docs | official docs | 2026-03-06 | medium | extension seam is real | native hotspot roadmap |
| `mlx` custom Metal kernel docs | official docs | 2026-03-06 | medium | Metal-kernel seam is real | native hotspot roadmap |
| `mlx` streams and Metal memory docs | official docs | 2026-03-06 | medium | live telemetry and stream primitives exist | scheduler |

## Language Surface Sources

| Source | Type | Last verified | Freshness risk | What it proves | Action relevance |
| --- | --- | --- | --- | --- | --- |
| `ml-explore/mlx-c` local clone `1370f59` | official repo | 2026-03-06 | medium | C bridge exists and is first-party | ABI and extension seam |
| `ml-explore/mlx-swift` local clone `a2f0d76` | official repo | 2026-03-06 | medium | Swift surface is active and embedding-aware | host SDK and launcher |
| `ml-explore/mlx-swift-examples` remote `HEAD c6844888` | official repo | 2026-03-06 | high | first-party examples exist outside the core Swift repo | host integration evidence |
| `ml-explore/mlx-swift-lm` remote `HEAD 6bb84aac` | official repo | 2026-03-06 | high | Apple-native LM and VLM path is real | embedded-mode planning |

## LTX And Host Sources

| Source | Type | Last verified | Freshness risk | What it proves | Action relevance |
| --- | --- | --- | --- | --- | --- |
| `Lightricks/LTX-2` local clone `9e8a28e` | official repo | 2026-03-06 | high | current official LTX runtime repo and pipeline surface | product track |
| `Lightricks/LTX-2.3` model card and files | official model source | 2026-03-06 | high | current weights, license, and “Diffusers coming soon” status | provider, policy, artifacts |
| `Lightricks/ltx-desktop` local clone `32589e6` | official repo | 2026-03-06 | high | current desktop macOS gap and direct-path assumptions | desktop adapter |
| `Lightricks/ComfyUI-LTXVideo` local clone `82bd963` | official repo | 2026-03-06 | high | advanced Comfy node pack, not baseline access layer | Comfy strategy |
| ComfyUI LTX docs and current releases | official docs and repo | 2026-03-06 | high | LTX is now in core and current releases carry LTX fixes | refresh stale assumptions |
| `Lightricks/LTX-Video` | official legacy repo | 2026-03-06 | medium | legacy line is still visible and needs explicit separation | documentation clarity |

## Apple And Security Sources

| Source | Type | Last verified | Freshness risk | What it proves | Action relevance |
| --- | --- | --- | --- | --- | --- |
| Apple WWDC unified memory and media guidance | system | 2026-03-06 | low | unified memory and media engines are architecture inputs | scheduler and output path |
| Metal `recommendedMaxWorkingSetSize` and `currentAllocatedSize` docs | system | 2026-03-06 | low | memory telemetry is available at the platform level | admission control |
| OWASP CSRF cheat sheet | system | 2026-03-06 | medium | browser-origin and anti-CSRF controls still matter locally | HTTP security model |

## Ecosystem Reference Sources

| Source | Type | Last verified | Freshness risk | What it proves | Action relevance |
| --- | --- | --- | --- | --- | --- |
| `Blaizzy/mlx-vlm` local clone `1ad8937` | ecosystem repo | 2026-03-06 | high | active VLM and omni stack patterns | VLM research |
| `Blaizzy/mlx-audio` local clone `1da513b` | ecosystem repo | 2026-03-06 | high | active audio packaging and server patterns | audio research |
| `vllm-project/vllm-metal` local clone `e0f97be` | ecosystem repo | 2026-03-06 | high | serving and Metal-performance contrast case | text-serving contrast |

## Research Policy

- Prefer official docs and repos when they exist.
- Use ecosystem packages to understand practical patterns and gaps, not to define architecture truth.
- Re-check high-freshness sources before freezing any decision that depends on them.
