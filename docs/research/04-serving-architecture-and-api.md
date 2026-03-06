# Serving Architecture And API Research

Status: provisional serving design. The native API is the right center of gravity, but the exact public route set and SDK shape remain unfrozen.

## Executive Recommendation

Use a control-plane daemon with worker processes behind it.

- default transport on macOS: Unix domain socket
- loopback HTTP: opt-in only
- generic HTTP contract: handle-based inputs and outputs
- raw absolute file paths: trusted CLI or embedded surface only
- event delivery: SSE-style stream for v1

Compatibility APIs stay outside the core contract.

## Threat Model

The runtime is a local service, but local does not mean safe by default.

Threats that matter:

- browser-originated mutation against localhost HTTP endpoints
- path injection into mutating routes
- accidental cross-tool policy drift
- provider auth leakage
- remote-code execution without an explicit approval boundary

Threats currently not solved in v1:

- fully hostile multi-user shared-machine environments
- remote network serving

Those must be documented as exclusions, not ignored.

Important browser reality:

- `localhost` is treated as a potentially trustworthy origin by modern browsers
- that makes browser access more capable, not more trusted
- the daemon must never treat “it came from localhost” as equivalent to “it is safe”

## Transport Modes

### Unix domain socket

Preferred default for macOS shared-service usage.

Why:

- narrower exposure than loopback HTTP
- better fit for local desktop and CLI clients
- easier permission modeling

### Loopback HTTP

Allowed only as an opt-in transport for clients that truly need HTTP.

Requirements:

- mandatory auth
- strict browser-origin checks
- no wildcard CORS
- no ambient browser credential assumptions

### Embedded direct mode

Reserved for trusted first-party Apple apps or test harnesses. This is where direct-path convenience may remain acceptable.

## Auth And Browser-Origin Protections

The old “optional token” posture is gone.

### Mutating route requirements

- mandatory bearer or equivalent explicit session secret
- reject requests that fail policy even if they come from `127.0.0.1`
- if `Origin` or `Referer` is present, it must match an explicit allowlist
- if `Sec-Fetch-Site` is present and indicates cross-site, reject
- disable wildcard CORS

### Client guidance

- browser clients should usually talk to a trusted local wrapper, not the raw daemon
- the CLI should use UDS or the trusted local surface when possible
- if a browser-facing local UI is required, prefer a same-origin local app or wrapper instead of assuming a public website can safely reach into the daemon

## Daemon And Worker Execution Model

The control plane owns:

- auth and transport
- provider resolution
- artifact and provenance indexes
- scheduler
- capability service
- job store and event streaming

Workers own:

- model-family loading
- execution stages
- MLX compile and local build-cache behavior
- optional native extensions

The daemon should not directly execute remote-code-bearing model logic.

## File And Artifact Handle Model

The generic API uses handles, not raw file paths.

### Import flow

1. client imports or references input content
2. runtime returns an `input_handle`
3. jobs reference the handle

### Output flow

Jobs write to runtime-managed artifacts first. Clients then:

- stream results
- export to a destination
- or ask the runtime to reveal a trusted local path where appropriate

### Trusted local direct-path mode

The CLI or embedded callers may use direct local paths. That mode is not the canonical HTTP contract and should not leak into host adapters.

## Provisional Native API Shape

### Capability and model management

- `register` or `resolve` model source
- `inspect` source and artifact
- `list` or `describe` capabilities
- `load` or `prepare` model profile when explicit warmup is desired

### Inputs and artifacts

- `import input`
- `list input handles`
- `export artifact`
- `list artifacts`

### Jobs

- `submit job`
- `stream events`
- `cancel job`
- `query status`

### Example job payload

```json
{
  "model_id": "ltx-2.3-fast-local",
  "task": "video.generate",
  "inputs": {
    "prompt": "cinematic slow dolly shot of a fox in snow",
    "images": [
      {
        "input_handle": "inp_01H...",
        "frame_index": 0,
        "strength": 1.0
      }
    ]
  },
  "params": {
    "width": 768,
    "height": 512,
    "num_frames": 97,
    "fps": 24,
    "seed": 42
  },
  "output": {
    "artifact_format": "mp4",
    "destination": {
      "mode": "runtime_managed"
    }
  },
  "extensions": {
    "ltx": {}
  }
}
```

## Capability And Policy Propagation

Every host surface should see the same:

- task list
- constraints
- profiles
- output formats
- hardware-tier notes
- license and access-state facts
- remote-code requirement and approval state

That information belongs in the capability and provenance surfaces, not in adapter-local guesswork.

## Scheduler Recommendations

The scheduler should be treated as a resource graph, not as a one-line batching policy.

### Resources that matter

- unified memory
- compute streams
- CPU threads
- media encode or decode resources
- disk bandwidth
- worker slots
- build-cache activity

### Operational policy

- use family-declared stage graphs
- reserve per stage
- use live telemetry where available
- keep one heavy media job per worker as the v1 default
- prefer clear rejection reasons over optimistic overcommit

## Compatibility Facades

OpenAI-style or other compatibility APIs may still exist, but they must:

- translate into native jobs and handles
- not become the source of truth
- not erase provenance or policy fields

This is especially important for text families, where compatibility pressure is strongest.

## What This Doc No Longer Claims

- it does not claim the public route set is final
- it does not claim loopback HTTP alone is a sufficient safety boundary
- it does not claim raw file paths belong in the generic HTTP contract
- it does not claim all hosts must always use daemon mode instead of trusted embedded access
