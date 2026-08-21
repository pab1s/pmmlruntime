# JPMML → Rust Migration Plan

> **Private repo:** `pab1s/jpmml-evaluator-rs` (gitflow: `main` ← `develop` ← `chore/migration-plan`)
> **Upstream:** `jpmml/jpmml-evaluator@v1.7.7` (2026-02-01) + `jpmml/jpmml-model@v1.7.7`
> **Spec:** PMML 4.4 (Nov 2019, `pmml.xsd:4,490` lines) — last *released*; 4.4.1 announced-not-released
> **Strategy:** [Bun-in-Rust](https://bun.com/blog/bun-in-rust) (2026-07-08) — mechanical transpilation, 0 tests skipped
> **Status:** `chore/migration-plan` — plan only, NOT merged to `develop`

---

## 0. TL;DR Decision Matrix

| Question | Answer (this plan) | Why |
|---|---|---|
| **Feasibility?** | **YES** — 7.5/10 difficulty | 7% size of Bun, no GC-unsafe FFIs, all deps have Rust crates |
| **LOC** | **37,925** core + **12,441** model = **~50k hand-written** (+ ~15-20k XJC-generated) | `pmml-evaluator:278 files`, `pmml-model:245 files`, `46 PMML fixtures` |
| **Time (solo human)** | **4–5 months** | Scaled from Bun's 3 eng × 1 yr for 535k |
| **Time (1 eng + 8-16 LLM swarm, Bun method)** | **4–6 weeks** wall | 2-3 wk mechanical + 2-3 wk test burndown; ~$15-30k LLM |
| **Time (MVP: Regression+Tree+Mining+XML)** | **2–3 weeks swarm** / **6–8 weeks solo** | Covers 70% of real `sklearn2pmml`/`r2pmml` output |
| **Big-bang vs strangler?** | **Hybrid**: swarm *per-crate* mechanical files, **deployed via strangler-fig** with JNI fallback | Bun's big-bang needed 1.3M asserts; JPMML has 46 fixtures → too risky for pure big-bang |
| **jpmml-model?** | **Option A (preferred): port**; **Option B (fast): JNI bridge** via `jni` crate | JNI keeps XML correctness for free, port removes JVM forever |
| **License** | **Decide before first code commit** — upstream AGPL-3.0 dual BSD | Transpilation ≠ relicense; green-field port can be MIT/Apache-2.0 |

---

## 1. Ground Truth — What We Actually Measured

All numbers from live shallow clones at `/tmp/jpmml-clone/*` on 2026-08-21, not estimates.

### 1.1 jpmml-evaluator repo (`jpmml/jpmml-evaluator@23d0761`)

| Module | `src/main/java` LOC | Java files | Role |
|---|---|---|---|
| `pmml-evaluator` | **37,925** | **278** | **Core evaluator** — 95% of complexity |
| `pmml-evaluator-testing` | 1,208 | — | Test utilities (`ModelEvaluatorTest` harness) |
| `pmml-evaluator-reporting` | 321 | — | Trait + reporting impl |
| `pmml-evaluator-reporting-processor` | 820 | — | Annotation processor (`OperationProcessor.java`) |
| `pmml-evaluator-example` | 1,053 | 4 | `EvaluationExample.java:532`, `TestingExample.java:196` CLI |
| `pmml-evaluator-{jackson,kryo,metro,moxy}` | 0–236 | — | Thin SerDe adapters |
| `src/test/java` | ~12,000 | ~130 | JUnit suite |
| `src/test/resources/pmml` | 46 fixtures | — | `Iris.pmml`, `Shopping.pmml`, `MingChain*.pmml`, … |
| **Total repo** | **53,740** (`54,624` incl. POMs) | **434** | `pom.xml` parent `jpmml-parent:1.0.13`, Java 11, Guava 19-33.5, Commons-Math 3.1-3.6, fastcsv 3.6 |

**Top 20 heaviest files** (mechanical port bottleneck):
`GeneralRegressionModelEvaluator:1201` > `MiningModelEvaluator:1063` > `TypeUtil:899` > `NearestNeighbor:898` > `OutputUtil:897` > `Functions:878` > `Table:716` > `ModelManager:660` > `TextUtil:613` > `ExpressionUtil:613` > `ModelEvaluator:611` > …

### 1.2 jpmml-model repo (`jpmml/jpmml-model@1.7.7`)

| Component | LOC | Notes |
|---|---|---|
| `pmml-model/src/main/java` (hand) | 12,441 | 245 files: `PMMLObject`, `Visitor`, `PMMLUtil:76`, `ReflectionUtil:677` |
| `pmml-model/src/main/schema/pmml.xsd` | **4,490** | Full PMML 4.4 XSD |
| `org.dmg.pmml.*` generated (XJC) | ~15-20k equiv | 100+ classes `extends PMMLObject` (e.g. `Cell.java`, `ComparisonField.java`) — not in repo, produced at `target/generated-sources/xjc` |
| `org.jpmml.model.*` | 2,416 | `PMMLException`, `SAXUtil`, `XPathUtil:107` |
| **Total** | **22,405** | `jpmml-model` repo total |

**Combined hand-written**: **~50–55k** (evaluator 37,925 + model 12,441 + supporting) + ~20k generated = **70–76k** effective (`818 files` in combined shallow clone).

### 1.3 Upstream complexity signals

* **Visitors**: 13 files in `visitors/` + 304 `Visitor/visit/applyTo` hits — batteries that mutate `PMMLObject` tree (`AttributeFinalizerBattery`, `ElementInternerBattery`, `ModelEvaluatorVisitorBattery`, `ValueParser:401`, `UnsupportedMarkupInspector:350`).
* **Guava**: `ImmutableBiMap/List/Map`, `BiMap:12`, `HashBasedTable`, `RangeSet/Map`, `LoadingCache:7` + `CacheBuilder`, `Interner:5` — maps to `im`, `bimap`, `rangemap`, `moka`.
* **Commons-Math**: `NormalDistribution`, `Erf`, `Mean/Min/Max/Product/Sum` — maps to `statrs`, `libm`.
* **JAXB**: 267 hits (`jakarta.xml.bind`, `XmlTransient`, `XmlJavaTypeAdapter`) — the hardest part; no Rust equivalent.
* **Reflection**: 9 hits — low.
* **Thread safety**: 6 hits (`synchronized`, `Concurrent`) — `ModelEvaluator` is thread-safe, builder is not for loading variants.

### 1.4 Bun anchor for comparison

| Metric | Bun (Zig→Rust) | JPMML (Java→Rust) | Ratio |
|---|---|---|---|
| LOC | 535,496 Zig (1,448 files) | 37,925 core / 76k with model | **0.07× / 0.14×** |
| Original build | 1 yr solo pre-LLM | — | — |
| LLM big-bang | **11 days, 64 Claudes (4×16), 6,778 commits, 1M+ lines, $165k** (5.9B in / 690M out / 72B cached tokens) | — | Scale ref |
| Test suite | **60k tests / 1.38M expects / 4,174 files** (TS, language-independent) | **~130 tests / 46 PMML fixtures** (Java/JUnit, Java-specific APIs) | **Coverage gap is #1 risk** |
| Motivation | Memory safety (108/150 PRs) | No-JVM, WASM, binary size, AGPL escape, perf (1M scores/sec) | Different driver |
| Workflows | 50 dynamic loops: `PORTING.md` → `LIFETIMES.tsv` → file shards → `cargo check` → smoke → `bun test` → CI burndown 972→0 | Replicate per §5–6 | — |
| Result | 4% `unsafe` (78% single-line FFI), 19 regressions (all *syntactically identical, semantically different*) | Expect similar class | — |

**Lesson**: Size is not the blocker. **Test coverage is.** Bun's 1.3M asserts let big-bang fuse; JPMML's 46 fixtures cannot.

### 1.5 Feature coverage (from `features.md`, verified against `pmml.xsd`)

**Supported (plan covers)**: Pre-processing (`DataDictionary`+`MiningSchema`: strict `DataType`/`OpType`, outlier/missing/invalid handling), all 11+ model types (Association, Clustering, GeneralRegression, KNN, NaiveBayes, NeuralNetwork, Regression, RuleSet, Scorecard, SVM, Tree, **MiningModel ensemble**), `Targets` post-processing, `Output` (20+ `ResultFeature`s except `confidenceIntervalLower/Upper`, `standardError/Deviation`), `ModelVerification`, vendor extensions (sandboxing, Java-backed `Expression`/`Predicate`, MathML reports).

**Explicitly unsupported upstream** (skip in port, keep `UnsupportedMarkupInspector` gate): `AnomalyDetectionModel`, `BaselineModel`, `BayesianNetwork`, `GaussianProcess`, `Sequence`, `Text`, `TimeSeries`, `MiningModel/Segmentation/LocalTransformations` (deprecated 4.1), `ClusteringModel/CenterFields` (removed 3.2), `TableLocator`.

---

## 2. Risk Register — What Bun Taught Us to Fear

| # | Risk | Bun hit | JPMML variant | Severity | Mitigation (enforced in review) |
|---|---|---|---|---|---|
| R1 | `debug_assert!(side_effect)` erases expr | Yes → HMR broke | `assert(verify)` holding `insert` | HIGH | Clippy `debug_assert_with_mut_call` + ban in `PORTING.md` |
| R2 | `bytemuck::cast_slice` panics on odd len | Yes → `Blob.text` panic | `fastcsv`/`ValueUtil` casts | MED | Wrap `&buf[..buf.len() & !1]` guard |
| R3 | `ReleaseFast` no bounds vs Rust bounds | Yes → `ptrs[4095]` spill | `ArrayListMultimap` indexing | MED | Keep checks, proptest overflow blocks |
| R4 | `comptime` fmt vs `format_args!` | Yes → hyperlink `r` leak | `EvaluatorUtil` string templates | MED | Macro `pretty!()` only |
| R5 | Skipped tests | 0 skipped (enforced) | 46→500+ amplification needed | **CRITICAL** | CI gate `0 tests skipped or deleted` |
| R6 | `git stash`/`reset` collisions, disk OOM | Yes | Same with Maven `~/.m2` | MED | 4 worktrees, no `stash`, `cargo --frozen`, `systemd-run` for fuzz |
| R7 | Memory leak (`SSL_SESSION` 6.5KB/call) | Yes | `SimpleCache`/`LoadingCache` leak | MED | `Drop`, `miri`, `LeakSanitizer` |

---

## 3. Architecture — Target Crate Topology (mirrors Maven, prevents Bun's 16k cycle errors)

```
pmml-rs/
├─ crates/pmml-model-rs          # PMML object model (or JNI bridge if Option B)
│  └─ src/{pmml.rs, visitors.rs, types.rs}   # quick-xml + serde, HasFieldReference<E> → trait
├─ crates/pmml-evaluator-rs      # Public API: Evaluator, ModelEvaluator, FieldValue, Value
│  └─ src/{evaluator.rs, field.rs, value.rs, builder.rs}
├─ crates/pmml-evaluator-core    # shared: ExpressionUtil(613), TypeUtil(899), OutputUtil(897), Functions(878), Table(716)
│  └─ src/{expression/, functions/, types/, table.rs, visitors/}
├─ crates/pmml-model-evaluators  # per-model: Mining(1063), GeneralRegression(1201), Tree, Regression, SVM, NN, NB, KNN, Clustering, Association, RuleSet, Scorecard
├─ crates/pmml-evaluator-xml     # PMMLUtil.unmarshal/marshal via quick-xml (Metro replacement), 267 JAXB hits
├─ crates/pmml-evaluator-serde   # serde_json/yaml/toml (Jackson), rmp-serde (Kryo)
├─ crates/pmml-evaluator-testing # 46 fixtures + 500+ amplified (sklearn2pmml/r2pmml generated)
└─ crates/pmml-jni-bridge        # optional: jni crate if jpmml-model kept Java
```

**Rust deps** (in `Cargo.toml` workspace):
`quick-xml 0.36` (+`serde`), `serde{,_json,_yaml}`, `rmp-serde`, `bimap`, `rangemap`, `moka` (Guava `LoadingCache`), `im`/`indexmap`/`dashmap`, `statrs`, `libm`, `ndarray`/`nalgebra`, `chrono`, `regex`, `thiserror`/`anyhow`, `clap` (CLI), `criterion` (bench), `cargo-fuzz`.

**FFI seam** (strangler-fig): each `ModelEvaluator` subclass `#[repr(C)]`, feature-flag `evaluator = "rust-tree"` vs `jni-tree`. Same `extern "C"` idea as Bun's plan doc (but Bun ultimately fused — we stay split until perf gates pass ≤2%).

---

## 4. Bun Strategy Adapted — The Loop We Will Run

**Bun did**: `PORTING.md` (3h) + `LIFETIMES.tsv` (per-field ownership, 2 adversarial reviewers) → trial 3 files → swarm 1,448 files (4 worktrees×16 agents, 1 impl / 2 adversarial / 1 fixer, no `git stash`/`cargo`) → `cargo check` crate-by-crate → smoke per subcommand → `bun test` sharded (100 random) → CI 972→0 → dedup/`unsafe` reduction (4% left, 78% single-line FFI).

**We do the same, per-crate:**

```js
// pseudocode, not real:
while (task = queue.pop()) {
  result = implement(task)          // 1 impl agent, reads PORTING.md + OWNERSHIP.tsv
  feedback = await Promise.all([    // 2 adversarial reviewers, separate context
    review(result, "PORTING.md"),
    review(result, "LIFETIMES:OWNERSHIP.tsv + semantic diffs R1-R4"),
  ])
  apply(fix(feedback), result)      // 1 fixer, then commit per-file only
}
```

**Rules copied verbatim from Bun**: never `git stash`/`reset --hard`/`cargo` in loop; `cargo check` only at work-queue start; reject workaround with paragraph-long comment; `cargo asm` perf gate for hot path; `systemd-run` isolation for fuzz.

---

## 5. Phase Plan — With Gates (do not merge until gate)

| Phase | Wall (swarm 8-16) | Wall (solo) | Work | Exit gate |
|---|---|---|---|---|
| **0 — De-risk & Measure** | 1 wk | 1 wk | `cloc` already done; `mvn test` baseline on `Iris`/`Shopping`; capture `1M scores/sec` on your desktop; lock `PORTING.md` v0 skeleton; decide model port vs JNI | `docs/BENCHMARK.md` with numbers |
| **1 — PORTING.md + OWNERSHIP.tsv** | 1 wk (2p) | 2 wk | Map Java→Rust: `class→struct+trait`, `null→Option`, `instanceof→enum+match`, `BiMap→bimap`, `LoadingCache→moka`, `HasXxx→trait`; trace `PMMLObject` tree ownership, `EvaluatorBuilder` copy-on-build, `FieldValue` lifetimes. 2 adversarial reviews per entry | `docs/PORTING.md` + `docs/OWNERSHIP.tsv` 100% reviewed |
| **2 — Skeleton & CI** | 1 wk | 1 wk | `cargo new --workspace` done; wire `quick-xml` round-trip for `pmml.xsd:4,490`; `moka`/`bimap`/`rangemap` shims; `pmml-model-rs` empty compiles | `cargo check --workspace` green |
| **3 — Mechanical Port (swarm)** | **2-3 wk** | **6-8 wk** | File-sharded: `pmml-model` (if A) → `pmml-evaluator-core` → each `*ModelEvaluator`. 1 impl / 2 adversarial / 1 fixer per file | Every `.java` has `.rs` counterpart, compiles with `unsafe=allow` |
| **4 — `cargo check` as Queue** | 1 wk (parallel) | 1-2 wk | Per-crate `cargo check`, group by file; fix cycles (~2-4k errors expected); ban stubs | `cargo check --workspace` green, no stubs |
| **5 — Smoke → Unit → Fixture Burndown** | **2-3 wk** | **3-4 wk** | Smoke: `EvaluationExample`/`TestingExample` CLI (`--model model.pmml --input input.csv --output output.csv`); Unit: 130 JUnit→`#[test]`; Fixture: 46 PMML + **amplify to 500+** via `sklearn2pmml`/`r2pmml` generated; shard 10 fixtures/agent; `systemd-run` | Smoke green, unit 95%+, fixture 46/46 native + 500+ amplified |
| **6 — Perf & Parity Gates** | 1 wk | 1 wk | `cargo bench` vs `mvn -Dbenchmark` (Openscoring method); gate ≤2% regression, memory ≤ Java; `cargo asm` for `TypeUtil`/`ExpressionUtil`; `cargo fuzz` 24/7 for XML parsers | `1M scores/sec` reproduced; fuzz 1B execs |
| **7 — Idiomatize & Publish** | 2 wk | 3 wk | Reduce `unsafe` (target <5%, mostly FFI), `&'a` where `OWNERSHIP.tsv` allows; `cargo clippy pedantic`, `miri` | `cargo publish` dry-run, `miri` CI green |

**Totals**: **4–6 weeks swarm** / **4–5 months solo**. MVP (Regression+Tree+Mining+XML) **2–3 weeks swarm**.

---

## 6. Work Queues — How Parallelization Avoids Disk Death

* **File shards**: 278 `pmml-evaluator` + 245 `pmml-model` files → 4 worktrees by `crates/*` (as Bun went from 1 to 4 worktrees when `grep` froze EC2 IOPS).
* **Agents**: 8–16 total = 4 worktrees × 2–4 agents (Bun did 4×16=64; we scale down 7×).
* **Commits**: per-file only, `git commit <file>` + `git push`, never `git stash`/`reset --hard`/`cargo` in loop.
* **CI burndown chart**: track `failing PMML fixtures` (46→0) and `cargo check errors` (X→0), same as Bun's `972→23→0` curve.

---

## 7. Alternatives Considered

| Option | Effort | Pros | Cons | When to pick |
|---|---|---|---|---|
| **A — Full port (this plan, hybrid)** | 4-6 wk swarm | No JVM, WASM-ready, single binary, can BSD/MIT | Hardest XML | Default |
| **B — JNI bridge** (`pmml-model` stays Java, Rust evaluators via `jni`) | 1-2 wk | XML correctness free, strangler trivial | Needs JVM at runtime | Need demo in days |
| **C — Fork `hemeda3/rs-pmml` / `AbdealiLoKo/rpmml`** | 2 wk spike | Least work | Both abandoned (1/0 stars, 2023, 71/40 KB), incomplete | Spike only |
| **D — Stay Java + GraalVM `native-image`** | 1 wk | `1M scores/sec` kept, no port | Still JVM semantics, AGPL stays | Risk-averse |

---

## 8. Immediate Next Steps

1. Reviewer confirms **spec scope** (4.4 only vs 4.4.1 deltas, 3.x backwards-compat) and **model strategy** (port vs JNI) + **license target**.
2. Merge this branch after review → `develop` (not `main` yet) — plan becomes living doc.
3. Next branch `chore/benchmark-baseline` — Phase 0.
4. Then `docs/porting-and-ownership` — Phase 1.
5. Then `feat/skeleton` — Phase 2 and swarm kickoff.

---

## 9. References

* Upstream evaluator: https://github.com/jpmml/jpmml-evaluator
* Upstream model: https://github.com/jpmml/jpmml-model
* PMML 4.4 spec: https://dmg.org/pmml/v4-4/GeneralStructure.html
* Bun-in-Rust post: https://bun.com/blog/bun-in-rust
* Bun strangler plan: https://github.com/oven-sh/bun/blob/eeb4d9fdf6e9a7bdd45388d7f3a03dcf570839ad/docs/rust-rewrite-plan.md
* Openscoring bench: https://openscoring.io/blog/2021/08/04/benchmarking_sklearn_jpmml_evaluator/
* This repo: https://github.com/pab1s/jpmml-evaluator-rs (`chore/migration-plan`)

---

## Annex A — Measured Facts (do not edit without re-cloning)

* `pmml-evaluator/src/main/java`: **37,925 LOC, 278 files**
* `pmml-model/src/main/java`: **12,441 LOC, 245 files**
* `pmml.xsd`: **4,490 lines, 165 KB**
* `src/test/resources/pmml`: **46 fixtures**
* `src/test/java`: **~130 files, ~12k LOC**
* `functions/*`: **26 files**, `visitors/*`: **13 files**, `*ModelEvaluator`: **14 types**
* Guava `LoadingCache`: **7**, `BiMap`: **12**, `Immutable*`: **~30**, JAXB: **267**, Reflection: **9**
* Combined clone size: **818 Java files, 76,145 LOC**

## Annex B — What `features.md` Says We Can Skip (keep gate)

Not supported upstream — keep `UnsupportedMarkupInspector` failing loudly:
`AnomalyDetectionModel`, `BaselineModel`, `BayesianNetwork`, `GaussianProcess`, `Sequence`, `Text`, `TimeSeries`, `MiningModel/Segmentation/LocalTransformations` (deprecated 4.1), `ClusteringModel/CenterFields` (removed 3.2), `TableLocator` (placeholder), `distributionBased` clustering, `aggregateNodes`/`weightedConfidence` tree, `Coefficients` SVM, `VariableWeight` mining, `confidenceIntervalLower/Upper` output (4.4.1).

*Plan written 2026-08-21 by `chore/migration-plan` — amend via PR to `develop`.*
