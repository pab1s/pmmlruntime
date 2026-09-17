# Full-repo documentation plan - pmmlruntime + all bindings

Follows installed skill `generating-documentation` (5 layers). Adapts Layer 3 to the
repo's existing **mdBook** instead of migrating to Docusaurus/MkDocs (reuse wins).
The published tree has already been rewritten once against these rules, so this plan records how to extend it. Page shape lives in `docs/src/style-guide.md` and is worked out in `docs/src/getting-started/quickstart.md`: H1 title, audience callout, "you will learn" bullets, `Step N:` sections, prose and code and visual rhythm, output tables, `Next Steps` with 4 links. A published page never names another product (style-guide section "No other products"). Every worker and suborchestrator follows the style guide verbatim.
Two commands gate the tree: `python3 docs/check-docs.py` and `mdbook build docs`. Both pass today; keep them passing.

## 0. Skill → repo mapping

| Skill layer | Repo application | Tool (per skill Tool Matrix) |
| --- | --- | --- |
| L2 code docs, Rust core | `crates/pmmlruntime/src/**/*.rs` | **rustdoc** doc-comments (`///`, `//!`), `cargo doc --no-deps` |
| L2 code docs, Python | `python/pmmlruntime/__init__.py` (+ bridge `_native`) | **Sphinx, Google-style** docstrings (skill `references/code-documentation.md`); rustdoc on `python/_native/src/lib.rs` for the PyO3 surface |
| L2 code docs, JS node/web | `javascript/node/index.{js,d.ts}`, `test.js`, `node/src/lib.rs`, `web/src/lib.rs` | **TypeDoc + TSDoc** on `index.d.ts`/`index.js`; rustdoc on the two napi/wasm `lib.rs` shims |
| L2 code docs, Java | `java/src/main/java/com/pmmlruntime/*.java` | **Javadoc** (skill pattern = TypeDoc equivalent for Java); rustdoc on `java/native/src/lib.rs` (JNI surface) |
| L2 code docs, C ABI | `include/pmml_runtime.h`, `.hpp`, `crates/.../ffi/mod.rs` | rustdoc `Safety` contracts + Doxygen-style header comments, kept in sync by worker M7 (single source: `ffi/mod.rs`) |
| L3 site | `docs/src/**`, `SUMMARY.md`, `book.toml` | **mdBook** (already built, mermaid wired). Apply skill `references/documentation-sites.md` principles (structure, search, versioning, deploy) onto mdBook; do NOT migrate frameworks |
| L4 ADRs | `docs/src/internals/*.md` + new `docs/src/decisions/*.md` | **MADR** template from skill `templates/adr-template.md` |
| L5 diagrams | every concept-heavy page | **Mermaid** (`graph TD/LR`, `sequenceDiagram`), palette per `style-guide.md`: base `#0b7285`, infra `#36404a`, session `#e8590c` |
| L1 API docs | n/a (no REST). Binding APIs are L2 code-first per skill "code-first for existing" | Embed OpenAPI-style tables only where a binding has an HTTP-adjacent surface (none today - skip) |
| CI | `.github/workflows/docs.yml` (new) | skill `references/ci-cd-integration.md`: `cargo doc`, `mdbook build`, link + spec validation |

Existing assets to reuse verbatim: `docs/src/style-guide.md` (voice, scaffold, lengths, cross-links),
`docs/ARCHITECTURE.md` + `docs/src/internals/*`, `docs/book.toml` (mermaid), canonical fixtures
`DecisionTreeIris.pmml` / `GradientBoosterTest.pmml`, `bench/pmml/` (52 fixtures).

## 1. Orchestrator (1 agent, runs first, blocks all workers)

Owns `docs/src/style-guide.md` + this plan. Outputs before any worker starts:

1. **Topology + relationships** (Mermaid, lands in `docs/src/internals/architecture.md`):
   `bytes → xml (cold) → ir/lower+verify → engine (pure &[Value]) → session::run (Batch row/col) → ffi/python/java/node/web`.
   Ownership: `Arc<Ir>` shared, `Session: Send+Sync`, `BumpArena` hot vs `Rodeo` cold interning,
   `Cpu` provider + `rayon` sharding, `wide f64x4` behind `simd` feature.
2. **Binding matrix** (lands in `docs/src/deployment/*` + `docs/src/concepts/session.md`):
   Rust core → C ABI (`ffi`, opaque `PmmlEnv`/`PmmlSession`) → `include/*.h*` → pyo3 (`python/_native`) →
   JNI (`java/native`) → napi (`javascript/node`) / wasm (`javascript/web`). Rule: **behavior is
   documented once in core; bindings document only mapping + ownership + errors.**
3. **Per-language doc skeletons** (copy-paste templates for workers, from skill references):
   - Rust: `///` + `# Arguments / # Returns / # Errors / # Examples` (runnable doctest, preamble
     `let env = PmmlEnv::new();`), `//!` on every `mod.rs`/`lib.rs`.
   - Python: Google-style (`Args/Returns/Raises/Example` with `>>>`), `__init__.py` re-export docs.
   - JS: TSDoc (`@param/@returns/@throws/@example/@public`) on `index.d.ts`; examples in `test.js` style.
   - Java: Javadoc (`@param/@return/@throws {@link}`, `<pre>` example) mirroring `PmmlSession`/`PmmlEnv`.
   - C headers: `/** @brief/@param/@return/@safety */` mirroring `ffi` Safety contracts.
4. **ADR backlog** (MADR, new `docs/src/decisions/` + SUMMARY section): session-based API (vs static
   eval); hardened XML (`MAX_DEPTH 512`, 100 MB cap, XXE/DTD block); IR (`Vec<NodeIr>` flat + `Vec<Op>`
   bytecode + `Rodeo`); dual Batch layouts + Arrow; `Cpu`/rayon provider model; C ABI opacity +
   `Safety` contracts; one-binding-per-runtime strategy (pyo3/JNI/napi/wasm).
5. **Definition of Done + CI gates** (writes `.github/workflows/docs.yml`):
   `cargo doc --no-deps -p pmmlruntime` with `#![warn(missing_docs)]` clean; `cargo test --doc`;
   `mdbook build docs` clean; `wc -w` lengths per style-guide ±15%; every page keeps
   audience callout + `you will learn` bullets (tutorials) + `## Next Steps` (congrats + 4 links)
   - ≥1 admonition + code block every ≤4 paragraphs + captioned visual on concept pages;
   H2s short/stable (right-TOC anchors); Mermaid renders (no broken fences); SUMMARY nesting ≤2;
   no new framework deps. Two commands gate every change: `python3 docs/check-docs.py` and
   `mdbook build docs` (mdbook plus mdbook-mermaid). CI should run both.

## 2. Suborchestrators (11, one per module, fan out after §1)

Each owns its files, its mdBook pages, one module diagram, and cross-link repair.
Each spawns **1 worker per code file** (workers only edit their file + report drift).
Each suborchestrator enforces the style-guide check rules on its workers (the report must name
the reference page read and the humanizer pass) and owns the module's pages:
M1-M4 concepts, M5 models, M6 session and batch, M7-M10 bindings (working from `java/`,
`javascript/`, `python/`, and `include/` sources), M11 the book scaffolding, `api.md`, and the
evaluation pages.

| # | Module / owner | Files (→ 1 worker each) | Module outputs |
| --- | --- | --- | --- |
| M1 | `base` | `arena.rs`, `error.rs`, `field.rs`, `value.rs`, `mod.rs` (5) | `Value/FieldId/DataType/PmmlError/BumpArena` rustdoc; values table in `concepts/values.md` |
| M2 | `xml` (cold) | `reader.rs`, `unmarshal.rs`, `mod.rs` (3) | Hardening contract (limits, XXE/DTD, fuzz) → `production/security.md`; cold-path sequence diagram |
| M3 | `ir` | `ir.rs`, `lower.rs`, `intern.rs`, `verify.rs`, `mod.rs` (5) | `Ir/FieldMeta/Op/DerivedFieldIr` DAG docs → `internals/ir.md`; `bytes→RawPmml→Ir` diagram |
| M4 | `engine` core | `mining_schema.rs`, `output.rs`, `predicate.rs`, `targets.rs`, `simd.rs`, `transform/{mod,vm,builtin,discretize,mapvalues}.rs`, `mod.rs` (11) | Transform/predicate semantics → `transforms/*`, `concepts/schema.md`; VM `Op` table |
| M5 | `engine/models` | 17 model files (`tree, regression, general_regression, mining, scorecard, clustering, naive_bayes, nearest_neighbor, neural_network, support_vector_machine, association, rule_set, sequence, bayesian_network, gaussian_process, time_series, anomaly_detection, baseline, text` + `mod.rs`) (~19) | Per-model rustdoc + group pages `models/{trees,regression,classification,neural}.md` (Overview→export→score→knobs→Fixture) |
| M6 | `session` | `env.rs`, `options.rs`, `session.rs`, `batch.rs`, `input.rs`, `arrow.rs`, `providers/{mod,cpu}.rs`, `mod.rs` (9) | `PmmlEnv/Session/Batch/Cpu` lifecycle + row/col layouts → `concepts/session.md`, `batch/*`; hub diagram `bytes→session→run` |
| M7 | `ffi` + C headers | `ffi/mod.rs`, `include/pmml_runtime.h`, `include/pmml_runtime.hpp` (3) | C ABI reference (opaque handles, ownership, `Safety`, error codes) → `deployment/c.md`; header/rustdoc sync check |
| M8 | Python binding | `python/_native/src/lib.rs`, `python/pmmlruntime/__init__.py`, `tests/test_inference.py`, `test_arrow.py` (4) | Sphinx Google docstrings + `deployment/python.md`; Arrow/CSV example from real tests |
| M9 | Java binding | `java/native/src/lib.rs`, `PmmlEnv.java`, `PmmlSession.java`, `PmmlException.java`, `NativeLoader.java`, `NodeInfo.java`, `InferenceTest.java` (7) | Javadoc + `deployment/` Java section (or extend `rust.md` hub); JNI ownership/error mapping table |
| M10 | JS bindings | `javascript/node/src/lib.rs`, `index.js`, `index.d.ts`, `test.js`, `javascript/web/src/lib.rs` (5) | TSDoc on `index.d.ts` + node/web usage → `deployment/` JS section; wasm vs napi differences |
| M11 | book + examples/tests | `docs/src/**` pages, `SUMMARY.md`, `examples/score_file.rs`, `bench_real.rs`, `tests/*.rs`, `bench/` fixtures (workers per example/test file) | SUMMARY repair, `api.md` (docs.rs links), `evaluation/correctness.md` fixture index, runnable examples verified by `cargo test` |

Total workers ≈ 70 (one per file above). No worker touches another worker's file or SUMMARY
(M11 owns SUMMARY; M7 owns headers).

## 3. Worker contract ( identical for every file-worker)

0. **Read the reference pages first (hard rule).** Open `docs/src/getting-started/quickstart.md`
   for layout and voice, plus the page you are changing and the two pages it links to. Name what
   you read in your report: no report without it is accepted.
0b. **Pass the humanizer skill over your prose (hard rule).** Run the `humanizer` skill
   (`/home/pab1s/.agents/skills/humanizer/SKILL.md`) on every page you write or revise: mark the
   AI patterns, draft, read aloud, then verify no claim was added or dropped. Style-guide
   §Humanizer rule lists the bans that apply here (em/en dashes, spaced dash substitutes,
   overused words, sales rhythm, bold mini-heading lists). Report what you rewrote.

1. Read orchestrator §1 skeletons + module brief from your suborchestrator; read the file + its
   existing mdBook page(s).
2. Document **why + contracts + errors**, not what-is-obvious (skill Pitfalls: drift,
   over-documentation, no-examples). Public items get full header (params/returns/errors +
   runnable example with canonical fixtures); internals get `//` intent comments only.
3. Binding workers (M7-M10): document **mapping table only** (native type ↔ core type, ownership,
   error translation) + link to core docs; never duplicate core semantics.
4. Report back to suborchestrator: (a) facts you were unsure of (→ ADR/manual flags, never invented),
   (b) cross-links touched, (c) doctest/example command you ran.
5. DoD per file: `cargo doc --no-deps` warning-free for touched crate; doctests/examples pass
   (`cargo test --doc -p pmmlruntime`, `pytest python/tests/`, `npm test` in `javascript/node`,
   `mvn test` in `java/` where touched); style-guide voice/scaffold/length respected.

## 4. Waves + validation

- **Wave 0** orchestrator (§1 artifacts + `docs.yml` skeleton). Gate: skeletons + ADR backlog merged.
- **Wave 1** M1-M6 core (rustdoc + internals/concepts pages). Gate: `cargo doc --no-deps`, `cargo test --doc`, `mdbook build docs`.
- **Wave 2** M7-M10 bindings (headers/Javadoc/TSDoc/Sphinx). Gate: binding test suites green + header/rustdoc sync diff empty.
- **Wave 3** M11 book repair (SUMMARY, api.md, correctness index, scaffold sweep:
  audience callouts, learn bullets, Step rhythm, captions, Next Steps).
  Gate: full `mdbook build`, link check, `wc -w` lengths, Mermaid fence check.
- **Never**: migrate mdBook → Docusaurus, invent behavior, duplicate core semantics into bindings,
  break the style-guide scaffold (audience callout, Step rhythm, captions, Next Steps),
  name another product in a published page,
  ship prose that has not passed the humanizer skill,
  or write a new page when an existing one fits (prefer extending, per Omnigent/Reddit AS-IS rule).
