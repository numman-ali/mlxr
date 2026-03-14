# Open-Source Release Checklist

Use this before calling a milestone “ready to publish.”

## Legal And Repo Basics

- top-level code license exists
- contributing, security, support, and conduct docs exist
- repo docs explain the difference between repo-code licensing and upstream
  model licensing

## Public Truth

- `README.md` explains what `MLXR` is in plain English
- [current-status.md](./current-status.md) is current
- [roadmap.md](./roadmap.md) is current
- family capability docs match the actual runtime truth
- promoted, supported-but-unpromoted, blocked, and planned rows are not mixed

## Runtime And CLI

- the public CLI story is documented
- the daemon, CLI, and host-surface relationship is clear
- install, runtime-home, and log-discovery expectations are documented
- public docs do not leak raw-path assumptions into the generic runtime API

## Release Health

- the repo validation gate is green, or any remaining failures are explicitly
  documented as known blockers
- any public speed, memory, or capability claim has a local receipt or a
  clearly marked hypothesis label
- temporary receipts that shaped public truth are referenced from docs where
  needed

## Product Framing

- the first Mac app is positioned as a thin client over the runtime
- `ltx-desktop` compatibility is clearly separate from the first-party app
- the first-party Mac app is clearly on the unified-studio path rather than a
  “simple app first, Studio later” split
- prompt enhancement is positioned truthfully as optional host-side UX unless
  and until the runtime contract changes
