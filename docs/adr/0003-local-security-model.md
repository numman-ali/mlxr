# ADR-0003: Local Security Model

Status: proposed

## Decision

The runtime will not treat loopback HTTP plus optional auth as a sufficient default security posture.

Default security posture:

- macOS shared-service transport defaults to Unix domain socket
- loopback HTTP is opt-in only
- mutating HTTP routes require mandatory auth
- browser-origin protections apply when HTTP is enabled
- the generic HTTP contract uses handles, not raw absolute file paths
- `trust_remote_code` stays off by default and is tracked as policy

## Why

The daemon is a local service, but it is still a security boundary. Browser-reachable localhost endpoints with mutating routes and path-based payloads are not acceptable as a casual default.

Modern browsers also treat `localhost` as a potentially trustworthy origin, which increases what browser clients can do. That is another reason loopback alone is not a trust model.

## Consequences

### Positive

- smaller attack surface by default
- clearer separation between trusted local callers and generic clients
- safer future desktop and browser integrations

### Negative

- slightly more complexity for local HTTP clients
- more auth and wrapper work for developer tooling

## Notes

This ADR does not solve full hostile multi-user local security yet. It establishes the minimum credible baseline for the runtime described in this repository.
