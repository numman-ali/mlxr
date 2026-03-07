# Skill Policy

Status: guidance for when repo work should stay in `AGENTS.md` versus move into a skill.

## Principle

`AGENTS.md` is the always-on repo constitution.

Skills are just-in-time overlays for workflows that need extra procedural structure, bundled references, scripts, or assets.

Repo-owned skills should live under `.agents/skills/` so Codex can discover them natively from the repo. Use `.codex/` for project-local Codex config, not as the skill directory.

The core workflow-orchestration split between runtime core, family adapters, and host adapters is repo doctrine, so it belongs in docs and `AGENTS.md`, not only inside a skill.

## Keep In `AGENTS.md`

Keep these in `AGENTS.md`:

- repo identity
- operating doctrine
- platform invariants
- mandatory dev loop
- source-of-truth order
- expectations for claims and doc updates

If a rule should apply to most sessions, it belongs in `AGENTS.md`.

## Good Skill Candidates

Create or expand a skill when a workflow is:

- repeated often
- error-prone enough to need a deterministic recipe
- distinct enough to benefit from its own operating style
- better served by bundled references, scripts, or assets

Examples in this repo:

- runtime debugging and log triage
- benchmark execution and validation
- family bring-up and workflow-boundary planning
- release hygiene
- specialized research workflows

Current repo-owned narrow skill:

- `validation`: runtime validation, log triage, and benchmark-discipline overlay
- `family-bringup`: new-family onboarding, workflow-boundary planning, and truthful first-slice framing
- `ltx-fidelity-debugging`: real-weight LTX fidelity bring-up, safe smoke runs, and stage-debug workflow

## Skill Design Rules

- keep the skill narrow
- keep the skill lean
- point to repo docs instead of duplicating doctrine
- prefer one-level references from `SKILL.md`
- use scripts when deterministic execution is better than freeform instructions

## Default Bias

When unsure:

- keep the rule in `AGENTS.md` if it should shape most sessions
- create a skill only when the workflow is specialized enough to justify it
