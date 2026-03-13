# mlxr-mac-app

This is the first-party native macOS client for `MLXR`.

Current status:

- real Swift package scaffold
- buildable with `swift build`
- tested with `swift test`
- launchable as a real dev `.app` bundle for Peekaboo and normal macOS app targeting
- still early and not yet a polished released app

The app is intentionally:

- SwiftUI-first
- a thin client over the shared runtime daemon
- separate from the `ltx-desktop` compatibility adapter
- smaller than the longer-term `MLXR Studio` vision

## What It Covers

- `Images`
  - `image.generate`
  - `image.edit`
- `Video`
  - `video.generate`
  - `video.condition.image`
  - `video.condition.audio`
  - `video.condition.video`
  - `video.interpolate`
  - `video.retake`
- `Library`
  - runtime output browsing and preview
- `Jobs`
  - active and completed job visibility
- `Settings`
  - runtime status
  - curated installs
  - advanced import

## Architecture

- native macOS client, not Tauri or Electron
- shared runtime backend, not a second inference engine
- modular Swift package targets for:
  - design system
  - app domain
  - runtime bridge
  - feature surfaces
  - app shell

## Development

From the repo root:

```bash
uv run python scripts/dev.py mac-app
```

This is the canonical development path. It:

- builds the Swift package
- stages a real `MLXRMacApp.app` bundle under `packages/clients/mlxr-mac-app/.build/dev-app/`
- embeds repo-aware runtime defaults into the bundle launcher
- launches the app through `open`, so macOS and Peekaboo see it as a normal app

Package-only checks still work directly when you just want to validate the Swift code:

```bash
cd packages/clients/mlxr-mac-app
swift build
swift test
```

Useful flags:

```bash
uv run python scripts/dev.py mac-app --no-launch --print-path
uv run python scripts/dev.py mac-app --runtime-home ~/.mlx-runtime
uv run python scripts/dev.py mac-app --uds-path /tmp/mlxr-control-plane.sock
```

Peekaboo flow:

```bash
peekaboo app switch --to MLXRMacApp
peekaboo see --app MLXRMacApp --json --annotate
```

The app expects the repo-local Python runtime path by default:

- repo root inferred from the checkout
- Python from `.venv/bin/python`
- runtime home from `MLX_RUNTIME_HOME` or `~/.mlx-runtime`
- UDS socket under the runtime home, with a shorter temp fallback when needed

Useful overrides:

- `MLXR_MAC_APP_REPO_ROOT`
- `MLXR_MAC_APP_PYTHON`
- `MLXR_MAC_APP_RUNTIME_HOME`
- `MLXR_MAC_APP_UDS_PATH`
