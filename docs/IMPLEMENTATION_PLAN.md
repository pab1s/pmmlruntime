# Implementation Plan — What Is Left (post-v1 Tree)

> **For: new agent starting from `development`**
> **Date:** 2026-08-24
> **Repo:** `pab1s/pmmlruntime` (private, gitflow `main ← development ← feat/*`)
> **Current:** `development@3a6ffe3` — 11.3k LOC Rust, 9 crates, **45/45 bench fixtures load+run pass**, `712 ns` single Tree, `1.48M rows/s` batch
> **Vault:** `~/Projects/jpmml-migration/` (spec, upstream, bench)
> **Prior plans:** `docs/PLAN.md` (Bun strategy), `.agents/plans/2026-08-23-pmml-runtime-v1-tree-plan.md` (v1 Tree shard)

---

## 0. TL;DR for new agent

```sh
cd ~/Projects/jpmml-migration/repo
git fetch --all && git checkout development && git pull
cat docs/IMPLEMENTATION_PLAN.md          # you are here
cat docs/BENCHMARK.md                    # 56× vs Java baseline
cat .agents/plans/2026-08-23-pmml-runtime-v1-tree-plan.md  # prior shard
cargo test --manifest-path Cargo.toml -p pmml-session --test all_fixtures  # => ok 45/45
cargo bench --manifest-path Cargo.toml -p pmml-bench --bench scoring    # => 712 ns
```

**Scope of this doc:** everything *not yet done* for **full JPMML parity + ONNX-grade runtime**. No need to redo Tree. Focus on gaps below, in priority order. Each task has branch name, files, gate.

**Gitflow you must follow** (already in `CONTRIBUTING.md`):
- `main` protected, `development` integration
- `feat/<slug>` per task, PR → `development` (draft until gate green)
- Commit per file, `git commit <file>` not `stash`
- Branch naming in §4

---

## 1. Gap Analysis — Done vs Left

### 1.1 Done (verified on `development`)

| Area | Evidence | Gate |
|---|---|---|
| **IR + Lower** | `crates/pmml-ir/src/{ir.rs:512, lower.rs:1228, intern.rs:110, verify.rs:1.8k}` — all 304 PMML 4.4 elements mapped, `Lower` handles Tree/Regression/Mining/Scorecard/Clustering/NaiveBayes/NN/GeneralReg/SVM/Association/RuleSet | `cargo test pmml-ir` pass |
| **Session** | `crates/pmml-session/src/{session.rs:350, env.rs, options.rs, providers/{cpu_serial.rs:80, cpu_batched.rs: stub}}` — `PmmlEnv` + `Session::from_bytes/from_file/run` with `Value[FieldId]` array | `all_fixtures_load: 45/45` |
| **Evaluators** | `crates/pmml-evaluator/src/models/{tree:275, regression:139, mining:369, general_regression:289, support_vector_machine: ~250, neural_network:146, naive_bayes:123, clustering:114, scorecard:229, association: ~120, rule_set: ~100, nearest_neighbor:356}` | `cargo test -p pmml-session` 15/15 quick tests pass |
| **Bench** | `crates/pmml-bench/{benches/scoring.rs, tests/tree_parity.rs, lib.rs}` with `DecisionTreeIris` 712 ns | `criterion` html in `target/criterion/` |
| **XML** | `crates/pmml-xml/src/{unmarshal.rs:4747, reader.rs:120, lib.rs}` — `quick-xml 0.37`, depth limit test | `pmml_xml::tests::parse_iris` |
| **Core** | `crates/pmml-core/src/{field.rs:241, value.rs, arena.rs, error.rs}` — `Value` enum, `FieldId`, `SymbolId` | — |
| **CLI / FFI / Python stubs** | `pmml-cli/src/main.rs:192` (inspect/run/verify), `pmml-ffi/src/lib.rs: FFI stubs`, `pmml-python/src/lib.rs: stub hello` | `cargo run -p pmml-cli -- inspect --model bench/...` |

Loc total: `11321` Rust (excl. `target`).

### 1.2 Left — prioritized backlog

| # | Category | Left | Priority | Effort |
|---|---|---|---|---|
| **L1** | **Provider Batched + Arrow** | `CpuBatchedProvider` is stub, no `rayon` `par_iter`, no `arrow::RecordBatch` / CSV `RecordBatch` bridge, no `run_batch` API used by bench's `tree_iris_batch_1k_sequential` (currently loops `run` 1k times) | **P0** | 3d |
| **L2** | **Python bindings (PyO3)** | `pmml-python/src/lib.rs` only `hello()`, no `#[pyclass] InferenceSession`, no `pyo3` `Session::run` exposing, no `maturin` wheel, no `bench/python` parity | **P0** | 3d |
| **L3** | **FFI real + ONNX C API parity** | `pmml-ffi/src/lib.rs` returns empty `PmmlSession`, no `PmmlRun` with `OrtValue` style, no `PmmlGetInputName/OutputName`, no `cbindgen` header | **P1** | 2d |
| **L4** | **JPMML full verification** | Spec doc `docs/PLAN.md` says `304 elements, 19 models, 12/19 parity`; but only `45 bench fixtures`. Remaining: `AnomalyDetection/Baseline/Bayesian/Gaussian/Sequence/Text/TimeSeries` stubs throw `UnsupportedAttribute`, plus `Extension` vendor handling, `ModelComposition` deprecated, `InvalidValueTreatment` full mapping | **P1** | 5d |
| **L5** | **Transforms VM full** | `crates/pmml-evaluator/src/transform/{builtin.rs, discretize.rs: 80, mapvalues.rs: 70, mod.rs, vm.rs: 180}` — only `if/greaterThan`, `NormDiscrete` done (KNN fix). Missing: `Apply` 100 builtins, `Discretize`, `MapValues` full, `TextIndex`, `Aggregate`, `Lag`, `NormContinuous` | **P1** | 4d |
| **L6** | **Perf Level 2 (SIMD+pool)** | No `smallvec` pooling, no `bumpalo` arena per batch, `HashMap<String,FieldId>` clones per `run`, no `memchr` fast path for `inlineTable` | **P1** | 3d |
| **L7** | **Verification + Fuzz + Safety** | `fuzz/` missing, `miri` not in CI, `cargo fuzz` XML 1M execs claimed but not in `BENCHMARK.md`, no `proptest` for `Tree` depth > 1000, no `leak-sanitizer` for `Session` | **P2** | 2d |
| **L8** | **Packaging/Release** | No `Cargo.toml` `publish`, no `Dockerfile`, no `pyproject.toml` / `maturin`, no `cbindgen` `pmml_runtime.h`, no `CHANGELOG.md` update, no `BENCHMARK.md` tables for all 45 fixtures (only Tree) | **P2** | 2d |
| **L9** | **Spec audit final** | `docs/BENCHMARK.md` only Tree; needs full 45 fixtures table + JPMML Java side-by-side, plus `spec/pmml.xsd` 4,490 lines coverage report | **P2** | 1d |

**Total remaining:** **~25d solo / 14-16d with 8 agents** (matches prior plan wall). None blocks `cargo test` but all block `1.0` release.

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

## 4. Definition of Done — `1.0` Gates

| Gate | Command | Threshold |
|---|---|---|
| `all_fixtures` | `cargo test -p pmml-session --test all_fixtures` | `45/45` (or `46` if `30-Days` added) |
| `quick tests` | `cargo test --all` | `15/15 + 2/2` per module |
| `bench single` | `cargo bench -p pmml-bench` | `≤ 800 ns` Tree Iris (now `712 ns`) |
| `bench batch` | `criterion` `batch_1k` | `≤ 500 µs` batched (parallel) |
| `fuzz` | `cargo fuzz run fuzz_unmarshal` | 1M execs, 0 crashes |
| `clippy` | `cargo clippy -- -W clippy::pedantic` | 0 warnings (now 7) |
| `miri` | `cargo miri test` | 0 leaks |
| `BENCHMARK.md` | `hyperfine` | Table for all 45 fixtures, Java vs Rust |

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
