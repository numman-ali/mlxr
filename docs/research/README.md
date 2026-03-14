# Research Docs

Use this folder for exploratory investigation, reference-gathering, and open
technical questions.

These docs are not canonical product or capability truth. They exist to help
the repo think clearly, validate hypotheses, and preserve technical context
that has not yet graduated into stable docs.

## Platform and architecture research

- [01-upstream-baseline.md](./01-upstream-baseline.md) — original upstream and
  repo-baseline scan
- [02-model-research-matrix.md](./02-model-research-matrix.md) — family and
  workload comparison matrix
- [03-language-and-runtime-choice.md](./03-language-and-runtime-choice.md) —
  early language and runtime tradeoffs
- [04-serving-architecture-and-api.md](./04-serving-architecture-and-api.md) —
  daemon and API-shape research
- [05-optimization-playbook.md](./05-optimization-playbook.md) — performance
  ideas and likely optimization seams
- [06-source-catalog.md](./06-source-catalog.md) — upstream source inventory
- [08-acceleration-techniques-survey.md](./08-acceleration-techniques-survey.md)
  — Apple-Silicon acceleration survey
- [09-open-questions-and-validation-plan.md](./09-open-questions-and-validation-plan.md)
  — unresolved platform questions and validation backlog

## LTX and family bring-up research

- [07-ltx-integration-seams.md](./07-ltx-integration-seams.md) — where LTX
  pressures the platform and what should stay family-local versus shared
- [10-ltx-fidelity-debug-playbook.md](./10-ltx-fidelity-debug-playbook.md) —
  fidelity-debug workflow for real LTX outputs
- [12-ltx-prompting-guide.md](./12-ltx-prompting-guide.md) — prompt-shape
  heuristics and cautionary notes
- [14-first-showcase-pack.md](./14-first-showcase-pack.md) — historical first
  showcase-pack rationale and receipts
- [17-multimodal-inference-engine.md](./17-multimodal-inference-engine.md) —
  supporting research for the shared owned-substrate direction

## Notes on scope

Family capability truth tables and active execution plans now live outside this
folder:

- canonical family truth lives in [docs/families/](../families/README.md)
- active working plans live in [docs/working/](../working/)
- the active LTX showcase plan lives in [ltx-showcase-execution-plan.md](../working/ltx-showcase-execution-plan.md)
- the active owned-substrate migration plan lives in [ltx-owned-substrate-migration-plan.md](../working/ltx-owned-substrate-migration-plan.md)
