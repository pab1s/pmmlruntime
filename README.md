# pmmlruntime

> **Rust port of [jpmml/jpmml-evaluator](https://github.com/jpmml/jpmml-evaluator) — PMML 4.4 reference evaluator**

Mechanical transpilation from Java → Rust following the strategy of [bun.com/blog/bun-in-rust](https://bun.com/blog/bun-in-rust):
* same architecture, same performance, same test suite — just Rust
* `Drop` instead of GC, borrow-checker instead of style guide
* 0 tests skipped — parity gated on `pmml-evaluator-testing` fixtures

**Status:** `chore/migration-plan` — plan only, no code yet. See `docs/PLAN.md`.

**License:** TBD (upstream is AGPL-3.0 / commercial BSD). Rust port will decide before first code commit.

## Repo layout (gitflow)

- `main` — releasable (protected)
- `development` — integration (protected)
- `chore/*` `feat/*` `fix/*` — work branches, PR → `development` → `main`

## Links

- Upstream Java: https://github.com/jpmml/jpmml-evaluator (`pmml-evaluator:37,925 LOC`, `jpmml-model:22,405`)
- Upstream Model: https://github.com/jpmml/jpmml-model
- PMML 4.4 spec: https://dmg.org/pmml/v4-4/GeneralStructure.html (`pmml.xsd:4,490 lines`)
- Original plan (this repo): `docs/PLAN.md`
