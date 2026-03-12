# Family Evaluation Framework

Status: reusable validation contract for deciding whether a family or variant
really works, rather than merely runs once.

## Purpose

This note exists to stop two recurring mistakes:

- confusing a minimal smoke rung with a product-quality conclusion
- promoting fast adapters before the base model has been evaluated on the
  official recipe

Use it whenever a new family, new row, or new adapter variant starts producing
real outputs and we need to decide what is actually true.

## Core Rule

Evaluate base first.

If the upstream base model has not been run on the official recommended recipe,
we do not yet know whether a failure belongs to:

- our runtime
- the recipe we chose
- the checkpoint row itself
- or an adapter we layered on top

Adapters such as Lightning LoRAs, turbo LoRAs, distilled rows, or quantized
rows are comparison lanes, not the truth source for whether the base family is
healthy.

## Minimum Evaluation Ladder

Run these in order.

### 1. Probe rung

Goal: prove the backend path returns at all.

This rung may use tiny sizes, very low step counts, or reduced settings, but it
must never be used to make a product-quality claim.

What it proves:

- the runtime path is alive
- prompt encoding runs
- denoise or decode path returns
- output export works

What it does not prove:

- the base model is good
- the official recipe works
- the adapter is better

### 2. Official base rung

Goal: run the exact official base recipe before judging the base row.

Match upstream as closely as possible:

- same model row
- same task
- same prompt style
- same negative-prompt policy
- same recommended step count
- same CFG or true-CFG setting
- same aspect-ratio bucket or published example size
- same image-count assumptions

If upstream says prompt enhancement is recommended but optional, evaluate both:

- raw prompt
- recommended prompt-enhanced prompt

Do not collapse those into one result.

### 3. Practical base rung

Goal: find the strongest base recipe that is still realistic on local Apple
Silicon.

This is where we establish the first promoted local base claim.

If the official recipe is too slow for normal iteration, that does not mean the
base model is broken. It means we need two truths:

- official-quality truth
- practical-local truth

### 4. Adapter comparison rung

Goal: compare Lightning, turbo, LoRA, quantized, or distilled variants against
the same base scenario.

Hold these constant:

- prompt
- negative prompt
- seed
- width and height
- task
- output count

Only then vary:

- adapter or profile
- intended step count
- scheduler preset when the adapter explicitly requires it

### 5. Promotion rung

Goal: decide what becomes the documented default.

Promote a row only after:

- real runtime receipt
- visual review
- timing and memory receipt
- truthful docs update

## Required Comparison Discipline

For any serious model comparison, keep these fixed:

- same prompt
- same negative prompt policy
- same seed
- same width and height
- same task
- same reference images
- same export format

Do not compare a `512x288` turbo run against a `704x384` base run and call that
a quality verdict.

## Required Receipt Bundle

Every promoted evaluation should leave behind:

- exported artifact
- run manifest
- runtime `events.jsonl`
- model id
- family variant
- adapter identity if any
- step count
- guidance or true-CFG value
- width and height
- seed
- a short human verdict

Use Gemini or another reviewer as a sidecar, not as the only truth source.
Human-visible review still matters.

## Serial Execution Rule

Heavy image and video evaluations should run one heavyweight job at a time by
default on Apple Silicon.

That means:

- one model stack loaded at a time unless there is benchmark evidence to widen
  concurrency
- one evaluation run active at a time for large image and video rows
- queue comparison batches sequentially rather than co-resident

This avoids turning evaluation noise into false model conclusions when the real
problem is memory pressure or Metal instability.

## Qwen-Specific Reminder

For `Qwen-Image` specifically:

- base `Qwen-Image-2512` must be judged on the official recipe before Lightning
  becomes the default story
- base `Qwen-Image-Edit-2511` must be judged on the official edit recipe before
  a low-step black-frame smoke is allowed to imply base failure
- Lightning and turbo LoRAs are valid fast lanes, but they are not the ground
  truth for whether base Qwen works

## Output Labels

Use these labels consistently:

- `probe only`
- `official base recipe passed`
- `official base recipe failed`
- `practical local base rung passed`
- `adapter faster but weaker`
- `adapter faster and still strong`
- `not yet enough evidence`

If the result is still ambiguous, say that directly instead of collapsing it
into a binary “works” or “broken.”
