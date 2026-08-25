# Handoff — pmmlruntime docs + single-crate merge (2026-08-25)

> Branch: `chore/docs-rustdoc` (from `development`) — last commits:
> `a1c7cb3 docs(pmmlruntime): translate facade docs to English`
> `7141c7a docs: crate & API rustdoc for pmml-core, pmml-session, pmml-ffi, cli, bench + README/ARCHITECTURE/CONTRIBUTING`
> plus 5 subagent docs commits (`1728939`, `6f958fa`, `4d021e3`, `5f9e4bd`, `d6f4171`) and `chore(fmt)`.

---

## 1. Original goal (user)

- `use rust doc skill to properly document the pmmlruntime repo following contributing. Do it slowly and carefully, paying attention to detail. use subagents where needed to speed up.`
- Follow `rust-documentation` skill (inspect → reader → rustdoc rules → module/crate docs → examples as doctests → links → headings → accuracy → validation via `cargo doc/test/clippy`).
- Follow `CONTRIBUTING.md` gitflow `main ← development ← feature/*` (per-file commits, no `stash`, `cargo check` only at queue start).
- After docs, user asked:
  1. `como veo la doc completa?` → explained `cargo doc -p pmmlruntime --open` etc., started `python -m http.server 8000 -d target/doc` (PID 94965).
  2. `why the doc doesnt appear as a unique pmmlruntime package?` → explained workspace virtual (9 crates) vs single crate, created facade `crates/pmmlruntime` re-exporting `pmml-core`…`pmml-session` as `pmmlruntime::core` etc.
  3. `i want all doc in english` → translated `crates/pmmlruntime/src/lib.rs` facade from Spanish to English.
  4. `also i dont know if in rust is it common to have several crates in a library or if it should be only one` → explained workspace vs single crate, recommended workspace+facade.
  5. `i think pmml-* is redundant, isnt it?` → discussed `pmml-*` is `jpmml-evaluator` heritage (37k LOC), `publish = false` makes it redundant.
  6. `i wont publish them by separate, also those names are heritage from jpmml... moreover this has a lot of inspiration from onnxruntime` → decided to drop `pmml-*` heritage, keep independent naming (ONNX-inspired).
  7. `arent there subcrates in rust? so you can do something like pmmlruntime::core` → explained no true subcrates, but facade `pub use pmml_core as core;` gives `pmmlruntime::core::Value` (workspace+facade pattern).
  8. `merge to single crate, is that a common practice?` → answered yes (very common <20k LOC), started merge.

---

## 2. What was completed

### Phase 0–5 (docs)

- **Baseline:** `cargo fmt --check` fixed, `cargo doc --workspace --no-deps` 0 errors (18 redundant_explicit_link_target warnings allowed), `cargo test --doc --workspace` initially 1 passed / 121 failed → after docs 106/16 → now single-crate 45/77 (regression, see §4).
- **Crate & module docs:** Added `//!` crate docs to 5 libs (`pmml-core` 13→54 lines, `pmml-ir` 11→44, `pmml-xml` 24→55, `pmml-evaluator` 12→77, `pmml-session` 13→54) via direct edits.
- **Public API docs (subagents):**
  - `pmml-ir` (ir.rs 805→1601, intern.rs, lower.rs 1988, verify.rs) — `FieldMeta` 10 fields, `BuiltinId` 100+ grouped, `ModelIr` 12 variants, `Interner`, `lower`, `verify_raw/verify_ir` with `# Errors`/`# Examples` (17 doctests pass).
  - `pmml-xml` (reader.rs, unmarshal.rs 5758) — `PmmlReader` hardened (`MAX_DEPTH 512`/`100MB`/XXE), `unmarshal` + 62 `Raw*` structs, 16 doctests pass.
  - `pmml-evaluator` (mining_schema, output, predicate, simd, targets, transform/vm, 12 models) — 31 doctests pass.
  - `pmml-session` + `pmml-ffi`/`python`/`cli`/`bench` — `Session::from_bytes` 68µs, `run` 402ns, `Batch` trait, `ExecutionProvider`, `Safety` for `unsafe extern "C"`, 38 doctests pass (session), 16 xml, 23 core.
- **README:** `25→85` lines, `0.1.0` status, `cargo add pmmlruntime`, 10-line `Session::from_bytes` example, links `docs/ARCHITECTURE.md`/`BENCHMARK.md`/`PORTING.md`.
- **ARCHITECTURE.md:** New `139` lines, 9-crate topology, flow `bytes→RawPmml→Ir→Session::run(Value[FieldId])`, ownership `Arc<Ir>` vs `BumpArena`, concurrency `rayon`, storage `quick-xml`/`arrow`/`lasso`, perf targets, invariants.
- **CONTRIBUTING.md:** `7→52` lines, quick start `fmt/clippy/check/test --doc/doc`, per-file commit, docs checklist, gates `45/45`.
- **Facade:** `crates/pmmlruntime` created (`Cargo.toml` `name = "pmmlruntime"` re-exports `pmml_core as core` etc., `lib.rs` 71 lines English, `pub use pmml_core as core;` → `pmmlruntime::core::Value`), `cargo doc -p pmmlruntime --no-deps` 0 errors, `cargo test -p pmmlruntime --doc` 1/1.

### Validation at `a1c7cb3` (before single-crate)

```
cargo fmt --check          0
cargo clippy --workspace -- -W clippy::pedantic -D warnings   0 (after allowing too_many_lines etc for pmml-xml)
cargo check --workspace    0
cargo test --workspace     pass (45/45 all_fixtures)
cargo test --doc --workspace  ~126 doctests pass (core 23, ir 17, xml 16, evaluator 31, session 38, pmmlruntime 1)
cargo doc --workspace --no-deps  0 errors, 18 warnings (redundant_explicit_link_target allowed)
```

---

## 3. Single-crate merge (in progress, incomplete)

**User asked to merge 9 crates into one.** Current work on `chore/docs-rustdoc` (not yet committed as new branch `chore/single-crate`).

**Design chosen:** Keep `crates/pmmlruntime` as single crate, modules `base` (was `core` → renamed to avoid `::core` shadowing), `xml`, `ir`, `engine` (was `pmml-evaluator` → `engine` to break JPMML heritage), `session`, `ffi`, `python` (bench/cli remain bins).

**Steps done:**

1. `mkdir -p crates/pmmlruntime/src/{base,xml,ir,engine/{models,transform},session/providers,ffi,python,bench}`
2. `cp` all `crates/pmml-*/src/*.rs` into `crates/pmmlruntime/src/{base,xml,ir,engine,session,...}`
3. Created `src/base/mod.rs`, `src/xml/mod.rs` (with `allow clippy::too_many_lines` etc), `src/ir/mod.rs` (`pub use ir::*;`), `src/engine/mod.rs`, `src/session/mod.rs` (with `pub mod ...`).
4. Rewrote `crates/pmmlruntime/src/lib.rs` to single-crate:
   ```rust
   pub mod base; pub mod xml; pub mod ir; pub mod engine; pub mod session; pub mod ffi; pub mod python;
   pub use base::{FieldId, PmmlError, Result, SymbolId, Value};
   pub use session::{PmmlEnv, Session, SessionOptions};
   // alias for migration: pub mod pmml_core { pub use crate::base::*; }
   ```
   Renamed `core` → `base` to avoid `core` crate conflict (`crate::core` shadows `::core`).
5. `Cargo.toml` for `pmmlruntime`: removed `pmml-core = { path }` deps, added direct deps `quick-xml`, `arrow`, `lasso`, `bumpalo`, `rayon`, `statrs`, etc., features `simd`, `python`.
6. Root `Cargo.toml`: `members = ["crates/pmmlruntime"]` (was 10).
7. `sed` batch: `pmml_core::` → `crate::base::`, `pmml_xml::` → `crate::xml::`, `pmml_ir::` → `crate::ir::`, `pmml_evaluator::` → `crate::engine::`, `pmml_session::` → `crate::session::`; fixed `crate::batch::` → `crate::session::batch::` inside `session`, `crate::predicate::` → `crate::engine::predicate::` inside `engine`, `crate::intern::` → `crate::ir::`, `crate::reader::` → `crate::xml::`, double `ir::ir::` → `ir::`, `split(':').last()` → `next_back()` for `double_ended_iterator_last`.

**Result:** `cargo check -p pmmlruntime` now **passes** (0 errors) after fixing `src/ir/mod.rs` re-exports and `base` rename, but `cargo test --doc -p pmmlruntime` **regressed** from `1/1` (facade) to `45/77` (106 passed, 16 failed before the last doc fix, now 45/77).

---

## 4. Current issue — 77 doctests failing (`cargo test -p pmmlruntime --doc`)

**Root cause:** `crate::` vs `pmmlruntime::` in doctests + `core` → `base` rename.

- Library code `use crate::base::error::PmmlError;` is correct (`crate` = `pmmlruntime` inside `pmmlruntime::xml::reader`).
- Doctests `use crate::base::OpType;` inside `src/base/field.rs:128` (`/// use crate::base::OpType;`) were for `pmml_core` crate where `crate` was `pmml_core`. After `sed` they became `crate::base::OpType` where `crate` is `pmmlruntime` and `base` is `pmmlruntime::base` → should be `pmmlruntime::base::OpType`. But `base` as a module named `base` is not a builtin, yet `crate::base::OpType` was still `unresolved import crate::base` (see `cargo test --doc` error `E0432: could not find 'base' in the crate root` for `crate::base::OpType` at `src/base/field.rs:128`).
- First fix: `crate::core` → `crate::base` (to avoid `core` shadowing `::core`). Then `cargo check` passed, but `cargo test --doc` still failed for `crate::base::OpType` (45/77).
- Second fix: changed docs `crate::base::` → `pmmlruntime::base::` via `sed '/\/\// s/crate::base::/pmmlruntime::base::/g'` for lines `///`. This made `src/base/field.rs:128` → `use pmmlruntime::base::OpType;` which should be found as `extern crate pmmlruntime;` in doctests, and `lib.rs` example `use pmmlruntime::base::Value;` did pass (1/1 before). But then code with trailing `//` was incorrectly changed (`pmmlruntime::base::field::DataType::Boolean` in `src/session/arrow.rs:108` code), so a Python script reverted non-doc lines `pmmlruntime::base::` → `crate::base::` for code, keeping `pmmlruntime::base::` only in `///` docs. After that, `cargo check` still passes, but `cargo test --doc` went from 106/16 to 45/77 (worse), because the `pmmlruntime::base::` in docs for `src/base`, `src/xml`, etc. now fails as `unresolved import crate::base` again? Actually the remaining 77 failures are for `base::arena::BumpArena`, `xml::reader::PmmlReader`, `session::batch::Batch`, etc. — all still `crate::base` in docs? Let's see latest `cargo test --doc` tail: `crates/pmmlruntime/src/base/arena.rs - base::arena::BumpArena (line 79)` etc. Those are still `crate::base` in docs? The last `sed` for docs was `'/\/\// s/crate::base::/pmmlruntime::base::/g'` which should have changed `/// use crate::base::` to `/// use pmmlruntime::base::` for docs, but the current `src/base/arena.rs:79` still has `crate::base::`? The `grep` after the Python revert shows `pmmlruntime::base::` in docs, but the error now is `crate::base::PmmlError` at `src/xml/unmarshal.rs:91` for code `use crate::base::error::PmmlError;`? Wait the latest `cargo test --doc` tail shows 77 failed, including `xml::reader::PmmlReader` etc., but the earlier `cargo check` passed, so code is fine, only docs fail.

**Reproduce:**

```sh
cd /home/pab1s/Projects/jpmml-migration/repo
cargo check -p pmmlruntime  # 0
cargo test -p pmmlruntime --doc  # 45 passed, 77 failed (after last fix: 106/16 → 45/77)
cargo doc -p pmmlruntime --no-deps  # 0 errors, 35 warnings (redundant_explicit_link_target)
```

**Files changed for single-crate (unstaged, not yet committed as single-crate commit):**

- `crates/pmmlruntime/src/base/` (from `pmml-core`, `core` → `base`)
- `crates/pmmlruntime/src/xml/` (`reader.rs`, `unmarshal.rs`)
- `crates/pmmlruntime/src/ir/` (`ir.rs`, `intern.rs`, `lower.rs`, `verify.rs`, `mod.rs` with `pub use ir::*;`)
- `crates/pmmlruntime/src/engine/` (`mining_schema.rs`, `output.rs`, `predicate.rs`, `simd.rs`, `targets.rs`, `transform/*`, `models/*`)
- `crates/pmmlruntime/src/session/` (`session.rs`, `arrow.rs`, `batch.rs`, `env.rs`, `input.rs`, `options.rs`, `providers/*`)
- `crates/pmmlruntime/src/ffi/mod.rs`, `src/python/mod.rs`
- `crates/pmmlruntime/src/lib.rs` (now `pub mod base;` not `core`, `pub use base::{...}`)
- `crates/pmmlruntime/Cargo.toml` (single-crate deps, no `pmml-*` path deps)
- `Cargo.toml` (members = `["crates/pmmlruntime"]`)
- `src/base/field.rs:128` etc. doctests still failing.

---

## 5. What remains for next agent

### Immediate (to make `cargo test --doc -p pmmlruntime` green)

1. **Fix doctest `crate::` vs `pmmlruntime::` in `src/base`, `src/xml`, `src/ir`, `src/engine`, `src/session`:**
   - In `src/base/*.rs`, `src/xml/*.rs`, etc., docs `/// use crate::base::Value;` should be `/// use pmmlruntime::base::Value;` (or `pmmlruntime::Value` via re-export). Currently `cargo test --doc` shows `unresolved import crate::base` for `crate::base::OpType` at `src/base/field.rs:128`. The last Python script changed `crate::base::` → `pmmlruntime::base::` only for `use crate::base::` in docs, but `crate::base::OpType` still appears in `src/base/field.rs:128` as `crate::base`? Check: `grep -rn "crate::base::" crates/pmmlruntime/src --include="*.rs" | grep "///"` still shows some `crate::base` in docs that need `pmmlruntime::`.
   - Need to ensure every `///` example that uses `crate::` for `base`, `xml`, `ir`, `engine`, `session` is `pmmlruntime::`. The earlier `sed` for `crate::base::` → `pmmlruntime::base::` was only for `use crate::base::` not for `crate::base::OpType` without `use`? The error `Ok::<(), crate::base::PmmlError>(())` at `src/xml/unmarshal.rs:6482` is `crate::base::PmmlError` without `use`, so need `s/crate::base::/pmmlruntime::base::/g` for all `crate::` in `///` lines, not just `use`.
   - Do: `find src -name "*.rs" -exec sed -i -E '/^ *\/\/\// s/\bcrate::base::/pmmlruntime::base::/g; s/\bcrate::xml::/pmmlruntime::xml::/g; s/\bcrate::ir::/pmmlruntime::ir::/g; s/\bcrate::engine::/pmmlruntime::engine::/g; s/\bcrate::session::/pmmlruntime::session::/g; s/\bcrate::Value\b/pmmlruntime::Value/g; s/\bcrate::PmmlError\b/pmmlruntime::PmmlError/g'`.

2. **Fix `base` module docs that still reference `crate::base::Value` as `` `crate::base::Value` `` rustdoc links vs code:** Links `` [`crate::base::Value`] `` in `//!` can stay `crate::`, only `use` and type paths in `` ```rust `` blocks need `pmmlruntime::`.

3. **Ensure `src/base/mod.rs` re-exports are complete:** Currently `pub use ir::*;` was added for `ir`, but `base` already has `pub use field::{DataType, OpType, ...}` etc. Verify `crate::base::OpType` is available as `pmmlruntime::base::OpType` via `pub use field::{OpType}`.

4. **Run validation after each fix:**

   ```sh
   cargo fmt
   cargo check -p pmmlruntime
   cargo test -p pmmlruntime --doc  # goal 126/126 (currently 45/77)
   cargo test --workspace  # after single-crate, workspace = only pmmlruntime, so 45/45 all_fixtures must still pass – fixture tests are in old crates `pmml-session/tests/all_fixtures.rs` which is not yet moved to `crates/pmmlruntime/tests/`. Need to move `crates/pmml-session/tests/all_fixtures.rs` → `crates/pmmlruntime/tests/all_fixtures.rs` and update `use pmml_session::` → `use pmmlruntime::session::`.
   cargo doc -p pmmlruntime --no-deps
   cargo clippy -p pmmlruntime -- -W clippy::pedantic -D warnings  # after allowing too_many_lines etc in lib.rs
   ```

### Single-crate completion

5. **Move old workspace crates out of members (already done) and decide to delete or keep:** `crates/pmml-core`, `pmml-xml`, `pmml-ir`, `pmml-evaluator`, `pmml-session`, `pmml-ffi`, `pmml-python`, `pmml-cli`, `pmml-bench` are still on disk but not in `Cargo.toml` members. For single-crate, either `git rm -r crates/pmml-*` (keep only `pmmlruntime`) or keep as `publish = false` with `members = ["crates/pmmlruntime", "crates/pmml-cli", ...]` if bins remain separate. User wants single crate, so likely `members = ["crates/pmmlruntime"]` and delete old dirs after `cargo test` passes.

6. **Update `docs/ARCHITECTURE.md` (139 lines) and `README.md` (85 lines):** Change `9 crates` → `single crate pmmlruntime` with modules `base`, `xml`, `ir`, `engine`, `session`, `ffi`, `python`, `cli`. Update `Cargo.toml` snippet `cargo add pmmlruntime` only.

7. **Update `crates/pmmlruntime/src/lib.rs` docs:** Bullet `- [`base`] —` etc. is already `base` (was `core`), but check `//! Previously a 9-crate workspace` etc. is already single-crate.

8. **Final gates (from `CONTRIBUTING.md`):**

   ```sh
   cargo fmt --check
   cargo clippy -p pmmlruntime -- -W clippy::pedantic -D warnings  # should be 0 after allowing `too_many_lines` etc in lib.rs
   cargo check -p pmmlruntime
   cargo test -p pmmlruntime
   cargo test -p pmmlruntime --doc
   cargo doc -p pmmlruntime --no-deps
   cargo test -p pmmlruntime --test all_fixtures  # after moving fixtures
   ```

---

## 6. How to continue (commands)

```sh
cd /home/pab1s/Projects/jpmml-migration/repo
git status  # on chore/docs-rustdoc, 30+ files modified, Cargo.toml members = ["crates/pmmlruntime"], src/base etc.
git diff --stat HEAD | head -n 40
git log --oneline -10  # a1c7cb3 is last committed, single-crate changes are unstaged

# 1. Fix doctests (crate:: -> pmmlruntime:: in ///)
find crates/pmmlruntime/src -name "*.rs" -exec sed -i -E '/^ *\/\/\// s/\bcrate::base::/pmmlruntime::base::/g; s/\bcrate::xml::/pmmlruntime::xml::/g; s/\bcrate::ir::/pmmlruntime::ir::/g; s/\bcrate::engine::/pmmlruntime::engine::/g; s/\bcrate::session::/pmmlruntime::session::/g' {} \;
find crates/pmmlruntime/src -name "*.rs" -exec sed -i -E '/^ *\/\/\// s/`crate::base::/`pmmlruntime::base::/g; s/`crate::xml::/`pmmlruntime::xml::/g' {} \;

# 2. Verify
cargo test -p pmmlruntime --doc 2>&1 | tail -n 30
cargo check -p pmmlruntime 2>&1 | tail -n 10
cargo doc -p pmmlruntime --no-deps 2>&1 | tail -n 10

# 3. Move fixtures (if keeping single crate)
mkdir -p crates/pmmlruntime/tests
cp crates/pmml-session/tests/all_fixtures.rs crates/pmmlruntime/tests/  # but crates/pmml-session is not in members, so path is still there on disk
# edit tests/all_fixtures.rs: s/pmml_session::/pmmlruntime::session::/; s/pmml_core::/pmmlruntime::base::/; s/pmml_ir::/pmmlruntime::ir::/; s/pmml_xml::/pmmlruntime::xml::/
sed -i -E 's/pmml_session::/pmmlruntime::session::/g; s/pmml_core::/pmmlruntime::base::/g; s/pmml_ir::/pmmlruntime::ir::/g; s/pmml_xml::/pmmlruntime::xml::/g; s/pmml_evaluator::/pmmlruntime::engine::/g' crates/pmmlruntime/tests/all_fixtures.rs

# 4. Commit single-crate
git add -A
git commit -m "refactor: merge 9 crates into single pmmlruntime (base/xml/ir/engine/session), drop pmml-* heritage

- Workspace members = [crates/pmmlruntime] only, other crates publish = false / removed from members
- pmml-core -> base (avoid ::core shadowing), pmml-evaluator -> engine (ONNX-inspired, not JPMML clone)
- pmmlruntime::base::Value, pmmlruntime::session::Session, pmmlruntime::engine::evaluate_tree via pub mod
- cargo doc -p pmmlruntime is single page, cargo check -p pmmlruntime 0"
```

---

## 7. Open questions for next agent

- Should `base` be kept as `base` or renamed to `types`/`common`? `base` was chosen to avoid `core` shadowing `::core`, but `base` is still generic. User said `pmml-*` is heritage, wants independent. `base`/`engine` is already independent, but could be `types`/`parser` if preferred.
- Should `crates/pmml-cli` and `crates/pmml-bench` remain separate crates (bins) or be moved into `crates/pmmlruntime/src/bin/` and `crates/pmmlruntime/benches/`? Currently they are separate dirs not in members, so `cargo run -p pmml-cli` will fail until added back or moved.
- Should old `crates/pmml-*` dirs be `git rm -r` after single-crate is green? Keep for history or delete?

---

## 8. Useful paths

- `crates/pmmlruntime/src/lib.rs:1-88` — single-crate root, `pub mod base;` etc., `pub use base::{Value}`.
- `crates/pmmlruntime/src/base/mod.rs` — `pub use value::{Value}` etc., `pub mod arena;` etc.
- `crates/pmmlruntime/src/xml/mod.rs` — `pub use reader::{new_reader}` etc.
- `crates/pmmlruntime/src/ir/mod.rs` — `pub use ir::*;` (needed for `crate::ir::FieldMeta`).
- `crates/pmmlruntime/src/engine/mod.rs` — `pub use transform::eval_derived_fields;`.
- `crates/pmmlruntime/src/session/mod.rs` — `pub use session::{Session}` etc.
- `Cargo.toml` — members `["crates/pmmlruntime"]`, workspace deps `quick-xml`, `arrow`, `lasso`, `rayon`, etc.
- `docs/ARCHITECTURE.md` — 139 lines, still says `9 crates` — needs update to single crate.
- `target/doc/pmmlruntime/index.html` — generated via `cargo doc -p pmmlruntime --no-deps` (currently 35 warnings).

---

*Handoff written 2026-08-25 on `chore/docs-rustdoc` with unstaged single-crate changes. Next agent: fix 77 doctest failures (crate:: -> pmmlruntime:: in ///), move fixtures, update docs/ARCHITECTURE.md, commit single-crate, rerun gates.*
