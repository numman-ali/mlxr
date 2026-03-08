# First Text-First Showcase Pack

## Purpose

This note records the current best-available first-party `LTX-2.3` showcase pack for `MLXR`.

It exists so future sessions do not have to rediscover:

- which five clips are currently promotable
- which receipts back those clips
- what this pack proves
- what it does **not** prove yet

## Current pack

The canonical current receipt is:

- `tmp/showcase-runs/showcase-pack-20260308-best-available.json`

That pack contains five `10`-second text-first clips at `384x224 / 241f / 24fps`:

1. `dog_park_natural`
2. `anime_neon_chase`
3. `vintage_nostalgic_street`
4. `cinematic_spacewalk`
5. `stop_motion_workshop`

The dog row comes from the newer negative-guidance/rescaled-CFG path. The other four come from the earlier promoted stylized/score-friendly set.

## What this pack proves

- the real non-preview `LTX-2.3` bridge can now carry five distinct `10`-second text-first clips through the current `MLXR` showcase path
- the current best-available set includes:
  - one natural-audio dog scene
  - one anime scene
  - one vintage / old-school scene
  - one cinematic science-fiction scene
  - one handcrafted / stop-motion-like scene
- every promoted row has:
  - a real runtime receipt
  - first/mid/last stills
  - `ffprobe` output
  - Gemini review evidence

## What this pack does not prove

- that text-first natural-audio realism is solved broadly
- that all natural scenes now avoid soundtrack-like music
- that `video.condition.audio` scene semantics are quality-promoted yet
- that recommended, HQ, or longer-form profiles are ready to promote

At the time of writing, the natural dog row is green, but the nature-documentary and natural-vintage rows still need to clear the newer guidance path on the same Gemini review bar.

Follow-up rerun truth:

- `tmp/showcase-runs/20260308T035528Z-showcase-nature-documentary/` still comes back as a visual match with piano-like music
- `tmp/showcase-runs/20260308T035838Z-showcase-vintage-old-school/` still comes back as a visual match with nostalgic instrumental music

So this remains the best available mixed pack, not evidence that natural-audio realism is broadly solved.

## Receipts by scene

### Dog In Park

- run dir:
  `tmp/showcase-runs/20260308T034004Z-showcase-dog-park-natural/`
- key review result:
  Gemini match with barking plus natural outdoor ambience

### Anime Neon Chase

- run dir:
  `tmp/showcase-runs/20260308T024231Z-showcase-anime-neon-chase/`
- key review result:
  promoted in the earlier score-friendly pack

### Vintage Nostalgic Street

- run dir:
  `tmp/showcase-runs/20260308T030514Z-showcase-vintage-nostalgic-street/`
- key review result:
  promoted in the earlier score-friendly pack

### Cinematic Spacewalk

- run dir:
  `tmp/showcase-runs/20260308T030741Z-showcase-cinematic-spacewalk/`
- key review result:
  promoted in the earlier score-friendly pack

### Stop-Motion Workshop

- run dir:
  `tmp/showcase-runs/20260308T024540Z-showcase-stop-motion-workshop/`
- key review result:
  promoted in the earlier score-friendly pack

## Recommended next step

Broaden the natural-audio win beyond the dog scene.

In practice, that means:

- rerun `nature_documentary` on the newer guidance path
- rerun `vintage_old_school` on the newer guidance path
- promote either or both only if they clear the same Gemini video-plus-audio bar as the dog scene

Until that happens, this pack is the best truthful public story.
