# Operations And Packaging

Status: required platform design, not a future deployment note.

## Service Modes

The runtime must support three modes:

### 1. LaunchAgent-managed daemon

Preferred shared-service mode on macOS for local tools that need a persistent runtime.

### 2. On-demand trusted CLI mode

Starts the runtime on demand for local CLI usage without requiring a permanently running daemon.

### 3. Embedded first-party mode

Reserved for first-party Apple apps that need a shared core or SDK path instead of loopback transport.

### 4. App-private helper mode

Reserved for app-bundled helpers where XPC is the right private boundary. XPC is not the right universal cross-tool transport, but it is relevant for first-party app-private helpers.

## Runtime Home

The runtime owns one home directory such as:

```text
$MLX_RUNTIME_HOME/
  config/
  logs/
  jobs/
  temp/
  sources-ref/
  artifacts-portable/
  build-cache-local/
```

The exact path can remain configurable, but the portability split cannot.

## Installation Requirements

- installer must create config and writable runtime-home directories
- service registration must be explicit and reversible
- logs must be discoverable
- upgrade must not silently delete portable artifacts or provenance manifests
- build cache may be invalidated more aggressively than portable artifacts

## Upgrade And Migration Policy

Each runtime release must define migration behavior for:

- capability schema versions
- portable artifact manifests
- worker protocol versions
- build-cache invalidation

A runtime upgrade may invalidate machine-local build cache. It must not silently rewrite source provenance or portable artifact identity.

## Disk Quota And Garbage Collection

The runtime must separately manage:

- source references and provider-cache references
- portable artifacts
- machine-local build cache
- temporary job files

GC must be policy-aware:

- do not delete an artifact still referenced by a pinned model registration
- build cache can be evicted first
- temp files should be short-lived
- source reference manifests are small and should survive longer than bulk data

## Crash Recovery

The control plane should be restartable without corrupting:

- model registrations
- source manifests
- artifact indexes
- completed job records

Workers should be restartable independently. Crash recovery must distinguish between:

- transient worker failure
- repeatable adapter failure
- corrupted source or artifact state

## Sleep And Wake

Sleep and wake are first-class macOS lifecycle events.

The runtime must define:

- whether in-flight jobs are cancelled, paused, or marked failed on sleep
- how build-cache or worker state is handled after wake
- how long stale warm workers remain trusted after wake

## Codesigning And Notarization

If the runtime becomes part of a first-party macOS app distribution, it will need:

- codesigned binaries
- a notarization-compatible packaging path
- a clear story for bundled native extensions or helpers

This does not block local open development, but it should shape how native helpers are packaged now.

## Media Dependencies

The runtime must define its policy for:

- media encode and mux dependencies
- optional codecs
- audio and video export formats
- Apple-native output paths versus generic Python media stacks

The runtime should separate:

- model execution time
- decode time
- encode or mux time

Those are operationally different stages.

## Apple Packaging Direction

For a shared local runtime, the default packaging target should be a per-user LaunchAgent or equivalent user service.

Use XPC only when the runtime is intentionally app-private.

Reasons:

- a universal runtime needs to be reachable by CLI and multiple local tools
- app-private XPC is a good helper model, but not a universal local integration surface
- launchd behavior influences on-demand startup, crash restart, and logout semantics

## Multi-User And Shared-Machine Behavior

The design currently assumes a local single-user developer or creator machine. Shared-machine behavior still needs explicit rules for:

- token storage
- runtime-home location
- permissions on Unix domain sockets
- whether provider auth is per-user or shared

Until those rules exist, multi-user support is provisional.
