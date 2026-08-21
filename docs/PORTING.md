# PORTING.md — Java → Rust mapping (draft v0, to be completed in Phase 1)

> Mirrors Bun's `PORTING.md` (Zig→Rust) + `LIFETIMES.tsv`. Every rule must have 2 adversarial reviews.

| Java pattern | Rust pattern | Notes / pitfalls |
|---|---|---|
| `class Foo extends PMMLObject implements HasXxx<E>` | `struct Foo { … }` + `trait HasXxx` + `impl Visitable for Foo` | `PMMLObject` tree owns children via `Arc<Mutex<>>` or `Box<dyn Visitable>`; decide per `OWNERSHIP.tsv` |
| `null` | `Option<T>` | Never `unwrap()` in transpiled phase; map `MissingValueTreatment` explicitly |
| `instanceof` / casts | `enum Value { Continuous(f64), Discrete(String) }` + `match` | 593 hits in evaluator — ban `Any` downcast in port |
| `Guava ImmutableBiMap` | `bimap::BiMap` | `HasEntityRegistry.getEntityRegistry(): BiMap<String,Entity>` |
| `Guava RangeSet/RangeMap` | `rangemap::RangeMap` | `InputField.getContinuousDomain(): RangeSet<Double>` |
| `Guava LoadingCache` | `moka::sync::Cache` | `SimpleLoadingCache`, `ModelEvaluator` caches lazily per `PMML` instance |
| `Guava Interner` | `internment::Intern` or `lasso` | 5 hits |
| `Commons Math NormalDistribution/Erf/Mean/Min/Max` | `statrs`, `libm` | Gate on `VerificationUtil` fixtures |
| `JAXB @XmlTransient/@XmlAdapter` | `quick-xml` + `serde` derive | Hardest: manual `PMMLUtil.unmarshal` SAX security |
| `synchronized` / `Concurrent*` | `Arc<RwLock>` / `once_cell::sync::Lazy` | `ModelEvaluator` is thread-safe — preserve `Send+Sync` bounds |
| `Factory` / `Builder` | `Builder` with `build() -> Self` | `ModelEvaluatorBuilder.build()` copies config — must clone, not alias |
| `defer` / `try-finally` | `Drop` impl | Fix Bun-class leaks: `Evaluator.verify()` warm-up must not leak |
| `int` overflow (`IntMath.checkedAdd`) | `checked_add` + `thiserror` | `Functions.ADD` — preserve `ArithmeticException` semantics, not wrap |
| `assert` with side effect | `debug_assert!` erases — ban | `NodeResolver`, `TargetCategoryParser` |
| `Visitor` battery mutating tree | `trait Visitor { fn visit(&mut self, node: &mut dyn Visitable) }` | `AttributeFinalizerBattery` etc mutate in place |

See `OWNERSHIP.tsv` for per-field ownership.
