# Security Policy

## Scope

Security issues include things like:

- auth bypass or missing auth on local HTTP mode
- unsafe browser-reachable mutation routes
- raw-path leakage across generic runtime surfaces
- unsafe file import or export behavior
- provider token handling
- provenance or policy bypasses
- unsafe process, socket, or runtime-home permissions

Non-security issues include:

- output quality problems
- unsupported model requests
- benchmark regressions without a trust or isolation impact
- upstream model-license questions that do not involve a repo code flaw

## Supported Versions

For now, only the current default branch should be treated as supported for
security fixes.

## Reporting

If GitHub Security Advisories are enabled for the repo, use them.

If they are not enabled, do not post full exploit details in a public issue.
Open a minimal issue requesting a private reporting path, or contact the repo
owners privately through the repository hosting service if that path is
available.

## What To Include

Please include:

- affected commit or branch
- macOS version and hardware
- whether UDS or HTTP mode was used
- exact reproduction steps
- expected versus actual behavior
- whether credentials, local files, or model artifacts are exposed

## Response Expectations

This is an open-source, agent-native project with best-effort response times.
There is no formal SLA yet.

We will prefer:

- confirming the report
- reproducing it locally
- fixing the issue with minimal public detail before disclosure
- updating docs if the issue changes the public security posture
