# Implementation Plan — What Is Left (post-PMML44 Full Coverage)

> **For: new agent starting from `development`**
> **Date:** 2026-08-26 (updated from 2026-08-24)
> **Repo:** `pab1s/pmmlruntime` (private, gitflow `main ← development ← feat/*`)
> **Current:** `feat/pmml44-full-coverage@44e95a5` — ~26k LOC Rust, single crate `pmmlruntime`, **52/52 bench fixtures load+run pass (51 OK + 1 SKIP weightedConfidence)**, `402 ns` single Tree, `16.5M rows/s` Arrow batched
> **Vault:** `~/Projects/jpmml-migration/` (spec, upstream, bench)
> **Prior plans:** `docs/PLAN.md` (Bun strategy), `.agents/plans/2026-08-23-pmml-runtime-v1-tree-plan.md` (v1 Tree shard), `feat/pmml44-full-coverage` (this branch)

---

## 0. TL;DR for new agent

```sh
cd ~/Projects/jpmml-migration/repo
git fetch --all && git checkout feat/pmml44-full-coverage && git pull
cat docs/IMPLEMENTATION_PLAN.md          # you are here (now 52/52)
cat docs/BENCHMARK.md                    # 52 fixtures, 19 models
cat docs/ARCHITECTURE.md                 # single crate, 19 models
cargo test --manifest-path Cargo.toml -p pmmlruntime --test all_fixtures  # => ok 52/52 (51 OK + 1 SKIP)
cargo bench -p pmml-bench --bench scoring    # => 402 ns single, 61 ns/row batched
```

**Scope of this doc:** everything *not yet done* for **full JPMML parity + ONNX-grade runtime**. No need to redo Tree. Focus on gaps below, in priority order. Each task has branch name, files, gate.

**Gitflow you must follow** (already in `CONTRIBUTING.md`):
- `main` protected, `development` integration
- `feat/<slug>` per task, PR → `development` (draft until gate green)
- Commit per file, `git commit <file>` not `stash`
- Branch naming in §4

---

## 1. Gap Analysis — Done vs Left

### 1.1 Done (verified on `feat/pmml44-full-coverage` 2026-08-26)

| Area | Evidence | Gate |
|---|---|---|
| **IR + Lower** | `crates/pmmlruntime/src/ir/{ir.rs:2414, lower.rs: ~4k, intern.rs:110, verify.rs:156}` — all 304 PMML 4.4 elements mapped, `Lower` handles **19/19** models (Tree/Regression/Mining/Scorecard/Clustering/NaiveBayes/NN/GeneralReg/SVM/Association/RuleSet/**AnomalyDetection/Baseline/TimeSeries/GaussianProcess/Text/Sequence/BayesianNetwork**) | `cargo test` 124 doc + 90 lib pass |
| **Session** | `crates/pmmlruntime/src/session/{session.rs:1319, env.rs, options.rs, providers/{cpu_serial.rs, cpu_batched.rs: rayon par_chunks(256)}}` — `PmmlEnv` + `Session::from_bytes/from_file/run/run_batch/run_batch_arrow` with `Value[FieldId]` array | `all_fixtures_load: 52/52` |
| **Evaluators** | `crates/pmmlruntime/src/engine/models/{tree:275, regression:139, mining:369, general_regression:289, svm:250, nn:146, naive_bayes:123, clustering:114, scorecard:229, association:120, rule_set:100, nearest_neighbor:356, anomaly_detection:355+7, baseline:740, gaussian_process:14k, text:520, time_series:198, sequence:8.5k, bayesian_network:33k}` + `transform/{builtin,vm,discretize,mapvalues}` + `simd` | `cargo test` 52/52 fixtures + 7 sequence/bayesian pass |
| **Bench** | `crates/pmml-bench/{benches/scoring.rs, src/bin/large_trial.rs}` with `DecisionTreeIris` 402 ns, `61 ns/row` batched Arrow | `criterion` html in `target/criterion/` + `BENCHMARK.md` 52 rows |
| **XML** | `crates/pmmlruntime/src/xml/{unmarshal.rs:15475, reader.rs:498,lib.rs}` — `quick-xml 0.37`, depth 512, XXE blocked, 19 models, `Extension` graceful | `cargo test xml` parse_iris + XXE + depth |
| **Core** | `crates/pmmlruntime/src/base/{field.rs, value.rs, arena.rs:125, error.rs}` — `Value` enum, `FieldId`, `SymbolId`, `BumpArena` | — |
| **CLI / FFI / Python** | `pmml-cli` inspect/run/verify, `pmml-ffi` PmmlEnv/Session (stub, P0 deferred), `pmml-python` pyo3 hello (stub, P0 deferred) | `cargo run -p pmml-cli -- inspect` + `cargo check` |

Loc total: `~39121` Rust raw (`26k` non-blank, single crate) + `15475` xml + `~15k` models.

**Update 2026-08-26:** `feat/pmml44-full-coverage` closes L4 (JPMML full verification) — now 19/19 models, 304 elements, 52 fixtures. L1 (Batched+Arrow) + L5 (VM) + L6 (Perf Level2) already green per `BENCHMARK.md` 402ns/61ns. Remaining for 1.0: L2/L3 (Python/FFI real) deferred to 0.2.0, L7/L8/L9 packaging/spec audit.

### 1.2 Left — prioritized backlog (updated 2026-08-26)

| # | Category | Left | Priority | Effort | Status |
|---|---|---|---|---|---|
| **L1** | **Provider Batched + Arrow** | `CpuBatchedProvider` rayon `par_chunks(256)`, `arrow::RecordBatch` bridge, `run_batch`/`run_batch_arrow` | **P0** | 3d | **DONE** (ARCHITECTURE §4, BENCHMARK §3: 61 ns/row) |
| **L2** | **Python bindings (PyO3)** | `pmml-python` only `hello()`, no `#[pyclass] InferenceSession`, no `pyo3` `Session::run` exposing, no `maturin` wheel | **P0→P2** | 3d | **DEFERRED to 0.2.0** (stub retained, `cargo check --features python` green) |
| **L3** | **FFI real + ONNX C API parity** | `pmml-ffi` returns empty `PmmlSession`, no `PmmlRun` with `OrtValue`, no `cbindgen` header | **P1→P2** | 2d | **DEFERRED to 0.2.0** (PmmlEnv/Session create/release green, `PmmlRun` stub) |
| **L4** | **JPMML full verification** | **DONE** `feat/pmml44-full-coverage` — now 19/19 models, 304 elements, 52 fixtures (AnomalyDetection/Baseline/Bayesian/Gaussian/Sequence/Text/TimeSeries all scoring), `Extension` graceful, `ModelComposition`/`CenterFields` still `UnsupportedMarkup` | **P1** | 5d | **DONE 2026-08-26** `cargo test all_fixtures 52/52` |
| **L5** | **Transforms VM full** | `Apply` 100 builtins, `Discretize`, `MapValues` full, `TextIndex`, `Aggregate`, `Lag`, `NormContinuous` | **P1** | 4d | **DONE** (vm.rs + builtin.rs 100 funcs, `sequence_bayesian_quick` 7/7) |
| **L6** | **Perf Level 2 (SIMD+pool)** | `smallvec` pooling, `bumpalo` arena per batch, `AHashMap` ahash, `memchr` fast path | **P1** | 3d | **DONE** (BENCHMARK 402 ns single, 61 ns batched, `with_value_buffer` stack 64) |
| **L7** | **Verification + Fuzz + Safety** | `fuzz/` + `miri` + `cargo fuzz` 1M execs + `hardening_l7` | **P2** | 2d | **DONE 2026-08-26** — `fuzz/fuzz_targets/fuzz_unmarshal.rs` covers unmarshal+lower+Session cold path (`cargo fuzz` 60s ~1M execs), `crates/pmmlruntime/tests/hardening_l7.rs` 14 tests: XML depth 512/100MB/XXE, tree 5k flat Vec, DerivedField cycle, Session leak/thread (Arc/BumpArena/LAG_BUFFER), `proptest` random tree + unmarshal nevers-panic + builtin, `miri`/`clippy` pedantic green. |
| **L8** | **Packaging/Release** | `Cargo.toml` `publish`, `Dockerfile`, `pyproject.toml`/`maturin`, `cbindgen` header, `CHANGELOG.md` | **P2** | 2d | **TODO** — `cargo publish --dry-run` pending, `Dockerfile` missing |
| **L9** | **Spec audit final** | `BENCHMARK.md` full 52 fixtures table + `pmml.xsd` 4490 lines coverage report | **P2** | 1d | **PARTIAL** — BENCHMARK now 52 rows, Java side-by-side for Tree only (§1), full 52 Java compare TODO |

**Total remaining for 1.0:** **L7 DONE** + L8 (packaging) + L9 (full Java 52 compare). **L1/L4/L5/L6/L7 DONE.** L2/L3 deferred to 0.2.0 per full-coverage plan.

Previous total **~25d solo / 14-16d** now **~2d solo** remaining for 1.0 (packaging + spec audit).

---

## 2. Phased Backlog — Branches & Gates

Use **8 agents** per prior plan's 4 worktrees: `.worktrees/{batched,python,ffi,transform}`. Each phase = branch off `development`, PR back, gate = `cargo test` + `criterion` + `all_fixtures`.

### Phase A — Batched Provider (3d, WT `batched`, agent A0)

**Branch:** `feat/batched-arrow` off `development`

| Task | Files | Do | Gate |
|---|---|---|---|
| A1 | `crates/pmml-session/src/providers/cpu_batched.rs` | Replace stub: `rayon::par_iter` chunk 1k, `sessions: Arc<Ir>` `Send+Sync`, `values: &mut [Value]` per thread Alloc. Copy `cpu_serial.rs` logic, shard by `batch.len()/num_cpus`. | `cargo test -p pmml-session -- cpu_batched` |
| A2 | `crates/pmml-core/src/arena.rs` | Add `BumpArena` wrapper over `bumpalo::Bump` for per-batch `Value` vec; reset per `par_iter` chunk, not per row. | `cargo test arena` |
| A3 | `crates/pmml-session/src/session.rs:run_batch` | Add `fn run_batch(&self, batch: Vec<HashMap<String,Value>>) -> Vec<HashMap<String,Value>>` that dispatches to provider, clones `name_to_id` once, not per row. Current `bench/scoring.rs` loops `run(clone)` — fix bench to use `run_batch`. | `criterion` `tree_iris_batch_1k_parallel` < 400 µs (×2 vs 815 µs) |
| A4 | `crates/pmml-xml/src/unmarshal.rs` + `crates/pmml-ir/src/lower.rs` | Add `Arrow` feature: `cargo add arrow --features csv` to convert `InlineTable`/`DataDictionary` to `RecordBatch`; handle `TableLocator` placeholder gracefully. | `cargo test arrow` |
| A5 | `crates/pmml-cli/src/main.rs` | Add `--batch input.csv --output output.csv` using `arrow::csv::Reader` → `RecordBatch` → `run_batch` → `arrow::csv::Writer`. | `cargo run -p pmml-cli -- run --model bench/... --batch input.csv` |

**Exit:** `criterion` shows `1.48M rows/s → 3M rows/s` batched.

### Phase B — Python (3d, WT `python`, agent A1/A2)

**Branch:** `feat/python-bindings` (depends on A)

| Task | Files | Do | Gate |
|---|---|---|---|
| B1 | `crates/pmml-python/Cargo.toml` | Enable `pyo3` feature `extension-module`, add `maturin` metadata, set `crate-type = ["cdylib"]` | `maturin develop` builds |
| B2 | `crates/pmml-python/src/lib.rs` | Implement `#[pyclass] struct PySession { inner: pmml_session::Session }` + `#[pymethods] fn new(path) + fn run(&self, dict) -> PyResult<HashMap>`. Map `pyo3::types::PyDict` ↔ `Value`. Reuse `pmml-session/src/session.rs` no clone per key. | `pytest bench/python/test_parity.py` 6/6 Tree |
| B3 | `bench/python/` | Copy `all_fixtures_rs` logic to `test_parity.py` using `pmmlruntime` wheel, compare to `jpmml_evaluator` (Java) if installed | `45/45` load |
| B4 | `pyproject.toml` | At repo root: `maturin generate-ci github` etc., `pip install -e .` doc | `maturin build --release` |

### Phase C — FFI Real (2d, WT `ffi`, agent A3)

**Branch:** `feat/ffi-onnx` (parallel to B)

| Task | Files | Do | Gate |
|---|---|---|---|
| C1 | `crates/pmml-ffi/src/lib.rs` | Replace stubs: real `PmmlEnv` holds `Arc<EnvInner>`, `PmmlSession` holds `Box<pmml_session::Session>`, implement `PmmlCreateSession` (calls `Session::from_file`), `PmmlRun(session, inputs: *const OrtValue, outputs)` with `OrtValue` = C struct `{ name, type, data }`. Follow `onnxruntime_c_api.h` naming. | `cargo test pmml-ffi` |
| C2 | `crates/pmml-ffi/cbindgen.toml` + `include/pmml_runtime.h` | `cbindgen --config cbindgen.toml --crate pmml-ffi --output include/pmml_runtime.h`, add `PmmlGetInputName/OutputName/Type`. | `cbindgen` passes |
| C3 | `examples/c/main.c` | Example C caller: `PmmlCreateEnv, PmmlCreateSession, PmmlRun, PmmlRelease`. | `gcc main.c -lpmml_runtime && ./a.out` |

### Phase D — JPMML Parity Full (5d, WT `transform`, agent A4-A5)

**Branch:** `feat/jpmml-parity-full` (depends on lower.rs)

| Task | Files | Do | Gate |
|---|---|---|---|
| D1 | `crates/pmml-ir/src/lower.rs` | Add missing lower arms: `Extension`, `AnomalyDetection/Baseline` → verify `verify_raw` rejects gracefully (keep `UnsupportedMarkup`), `Sequence/Text/TimeSeries` → `verify` stub with message. Cover 304 elements audit. | `cargo test pmml-ir` + `all_fixtures` still 45/45 |
| D2 | `crates/pmml-evaluator/src/transform/vm.rs` | Implement full `Op` dispatch for 100 `BuiltinId`: `TextIndex`, `Aggregate(count/sum/avg/min/max)`, `Lag` (need `Session` ring buffer), `NormContinuous` (linear). For KNN done; add `Discretize`, `MapValues` full. | `cargo test transform` |
| D3 | `crates/pmml-evaluator/src/models/*.rs` | Fill stubs: `clustering.rs` already, `association.rs/rule_set.rs` need `Item/InputTable` eval, `general_regression.rs` contrast `Categorical` full (now `Factor contrast`), `support_vector_machine.rs` kernel `rbf/poly` already, verify `NaiveBayes` threshold handling | `cargo test -p pmml-session` 15/15 |
| D4 | `crates/pmml-xml/src/unmarshal.rs` | Fix remaining 7 `unused_mut` warnings, depth limit 512 already, add `XXE` test | `cargo clippy` clean |

### Phase E — Perf Level 2 + Safety (3d, all WTs)

**Branch:** `feat/perf-level2`

| Task | Files | Do | Gate |
|---|---|---|---|
| E1 | `crates/pmml-core/src/arena.rs` + `crates/pmml-session/src/session.rs` | Use `lasso::Rodeo` + `ahash` for `name_to_id`, cache `FieldId` array, avoid `HashMap<String,Value>` clone per `run`. Benchmark `value.rs`. | `criterion` single `712ns→400ns` |
| E2 | `crates/pmml-core/src/value.rs` + `crates/pmml-evaluator/src/transform/builtin.rs` | `smallvec` for `predicates: SmallVec<[PredicateIr;4]>`, `memchr` for `inlineTable` fast path | `cargo bench` |
| E3 | `fuzz/` | `cargo fuzz init`, `fuzz_target_1` = `pmml_xml::unmarshal`, run `cargo fuzz run fuzz_unmarshal -- -max_total_time=60` | 1M execs |
| E4 | `.github/workflows/ci.yml` | Add `miri`, `clippy -- -W clippy::pedantic`, `cargo fuzz` 60s, `criterion` compare vs `main` artifact | CI green |

### Phase F — Release (2d)

**Branch:** `chore/release-0.1.0`

| Task | Files | Do | Gate |
|---|---|---|---|
| F1 | `Cargo.toml` + `CHANGELOG.md` | Set `version = "0.1.0"`, `publish = ["crates/*"]`, fill `BENCHMARK.md` tables for all 45 fixtures (not just Tree) | `cargo publish --dry-run` |
| F2 | `Dockerfile` + `pyproject.toml` | Multi-stage `rust:1.78` → `scratch` for `pmml-cli`, `maturin` docs | `docker build` |
| F3 | `docs/PORTING.md` + `docs/OWNERSHIP.tsv` | Update with real `OP→BuiltinId` mapping, `FieldId` ordinal table | `dead code` warnings zero |

---

## 3. How to Start New Agent — Step-by-Step

```sh
# 1. Pull
git -C ~/Projects/jpmml-migration/repo fetch --all
git -C ~/Projects/jpmml-migration/repo checkout development
git -C ~/Projects/jpmml-migration/repo pull

# 2. Verify gates before touching code
cargo test --manifest-path ~/Projects/jpmml-migration/repo/Cargo.toml --all
cargo test --manifest-path ~/Projects/jpmml-migration/repo/Cargo.toml -p pmml-session --test all_fixtures -- --nocapture  # 45/45

# 3. Create branch per task table
git checkout -b feat/batched-arrow   # or feat/python-bindings etc.

# 4. Worktrees for parallel (8 agents max, same as v1 Tree plan)
git worktree add ../worktrees/batched development
cargo --manifest-path ../worktrees/batched/Cargo.toml test  # isolated target

# 5. Commit & PR
git add -A && git commit -m "feat(batched): <task>"
git push -u origin feat/batched-arrow
gh pr create --base development --head feat/batched-arrow --draft --title "feat(batched): Arrow batch" --body "Gate: criterion 3M rows/s"
```

**Rules to copy from `PORTING.md`:**
- No `git stash` in loops, `cargo check` only at queue start
- 1 impl + 2 adversarial reviews per crate (use `skill` tool `requesting-code-review`)
- `cargo asm` for hot path before merge

---

## 4. Definition of Done — `1.0` Gates (updated 2026-08-26: 52/52, L7 green)

| Gate | Command | Threshold | Current |
|---|---|---|---|
| `all_fixtures` | `cargo test -p pmmlruntime --test all_fixtures` | `52/52` (51 OK + 1 SKIP weightedConfidence) | **52/52** |
| `quick tests` | `cargo test --workspace` | `124 doc + 90 lib + 14 hardening_l7 + 7 seq/bayes` | **pass** |
| `bench single` | `cargo bench -p pmml-bench` | `≤ 800 ns` Tree Iris | **402 ns** |
| `bench batch` | `criterion` `batch_1k` | `≤ 500 µs` batched (parallel) | **336 µs serial / 61 ns batched** |
| `fuzz` | `cargo fuzz run fuzz_unmarshal -- -max_total_time=60` | 1M execs, 0 crashes, 60s | **~1M/60s, see `fuzz/`** |
| `clippy` | `cargo clippy --workspace -- -W clippy::pedantic -D warnings` | 0 warnings | **0** |
| `miri` | `cargo miri test -p pmmlruntime --test hardening_l7` | 0 leaks | **0 (Session/Arc/BumpArena/LAG_BUFFER)** |
| `hardening_l7` | `cargo test -p pmmlruntime --test hardening_l7` | 14/14 | **14/14** (depth 5k, cycle, XXE, 100MB, leak, proptest) |
| `BENCHMARK.md` | `hyperfine` | Table for all 52 fixtures, Java vs Rust | **52 rows** |

---

## 5. Vault Reference — Where Things Are

```
~/Projects/jpmml-migration/
├─ repo/                   # gitflow, current feat/batched-arrow etc.
│  ├─ Cargo.toml           # 9 members
│  ├─ bench/pmml/          # 45 fixtures (copy of upstream test/resources)
│  ├─ spec/pmml.xsd        # 4,490 lines, 168 KB — source of truth
│  ├─ docs/{PLAN,BENCHMARK,PORTING,IMPLEMENTATION_PLAN}.md
│  └─ target/criterion/    # html reports
├─ upstream/               # jpmml-evaluator + jpmml-model (never commit)
├─ spec/                   # GeneralStructure.html, bun-in-rust.html, features.md
└─ worktrees/              # for 8-agent swarm
```

---

## 6. Risks — Copied from `PLAN.md:R6-R7`

- **IPC 1100µs Python** is still there until B done — don't benchmark Python before B3
- **WeakKeys cache** was already replaced with `lasso` — don't regress to `HashMap<String,*>`
- **DerivedField cycle** needs `BitSet` not `HashSet` — watch `vm.rs` stack overflow for depth > 5k (test `depth_limit`)

---

*Generated 2026-08-24 for `development@3a6ffe3`. Next agent: start with Phase A `feat/batched-arrow`.*
