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
- already moving toward one unified consumer studio shell instead of separate
  technical forms

## What It Covers

- `Home`
  - startup actions
  - recent assets
  - model and runtime readiness
- `Library`
  - project-first browsing and continuity
  - generated outputs plus imported assets
  - run-group-aware asset sets
  - uniform `4:5` browser tiles with a large in-window modal preview
  - keyboard-driven selection, preview, and continue-in-composer actions
- `Composer`
  - runtime-led planning and submission over the shared workflow seam
  - image generation and editing
  - video generation, animation, audio-conditioned, guided-video, interpolation, and retake flows
  - model-aware presets and advanced controls behind disclosure
  - shared draft behavior instead of a separate top-level Studio destination
- `Models`
  - curated installs
  - first-run starter-model setup
  - install queue
  - installed model details and removal
- `Activity`
  - left-rail utility panel for runtime-accepted running work, installs, and dismissible failures
- `Settings`
  - runtime status
  - advanced import
  - diagnostics and log discovery

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

Reference plan:

- [docs/mac-app-runtime-contract.md](../../../docs/mac-app-runtime-contract.md)
- [docs/working/mac-app-runtime-parity.md](../../../docs/working/mac-app-runtime-parity.md)

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
