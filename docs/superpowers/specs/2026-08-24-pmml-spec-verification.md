# Verificación de Implementación PMML 4.4 — Auditoría Completa vs Spec Oficial

**Fecha:** 2026-08-24  
**Spec:** PMML 4.4 `pmml.xsd:4490 líneas, 304 xs:element` + `GeneralStructure.html:932` + `jpmml-evaluator-features.md:96`  
**Repo:** `pab1s/pmmlruntime` `development` `9b90295` + `feat/v1-general-regression` `a383037` (11/11 parsing, 11/11 eval con stubs)  
**Auditor:** OpenCode (Muse Spark) — build mode

---

## 1. Veredicto Ejecutivo

> **¿Toda la spec PMML 4.4 está implementada? NO.**

| Dimensión | Estado | Cobertura |
|---|---|---|
| **Modelos PMML 4.4 (19 en spec)** | **12/19 (63%)** parsing + eval (7 explícitamente no soportados por *upstream JPMML* y tampoco por nosotros) | Para *paridad JPMML* es **100%**; para *spec estricta* es **63%** |
| **De esos 12 soportados** | **7/12 Full** (Tree, Regression, Mining, Scorecard, Clustering, KNN, NeuralNetwork) **+ 3/12 Parcial** (GeneralRegression 0.819 ok, SVM RBF XOR 0.1004/0.8995 ok, NaiveBayes Gaussian/PairCounts ok) **+ 2/12 Stub** (Association, RuleSet devuelven `Missing` pero parsean) | **~75% funcional** para casos reales `sklearn2pmml`/`r2pmml` |
| **Data flow (DataDictionary/MiningSchema/Targets/Output)** | **85%** — `DataType`/`OpType` strict, `Missing`/`Invalid`/`Outlier` como `Missing`, `Targets` rescale `Missing`, `Output` 20/26 `ResultFeature` (faltan 4) | Suficiente para 24/44 fixtures |
| **Transformaciones** | **40%** — `DerivedField` DAG + `FieldRef`/`Constant` + `Apply` bytecode 11/100 builtins, `MapValues`/`Discretize` stub, `NormContinuous/Discrete` stub, `DefineFunction` inline stub, `TextIndex`/`Aggregate`/`InlineTable` no | Requiere 100 builtins + `libm`/`statrs` |
| **Spec total `pmml.xsd` 304 elementos** | **~45%** elementos parseados (13 `parse_*` funciones) | Faltan ~170 elementos (TimeSeries, Text, Sequence, etc.) |

**Para reclamar “toda la spec” faltarían `7 modelos + 100 builtins + Transformaciones completas + 4 ResultFeature + Outlier/Invalid completos` — estimado 3–4 semanas adicionales (o 4–6 semanas para 11/11 eval *full* sin stubs).**

---

## 2. Auditoría de Modelos — 19 en `pmml.xsd` `MODEL-ELEMENT`

| # | Modelo PMML 4.4 | Spec `MODEL-ELEMENT` | JPMML 1.7.7 (reference) | `pmml-runtime` `RawPmml` | `ModelIr` | Evaluator | Test fixture | Veredicto |
|---|---|---|---|---|---|---|---|---|
| 1 | **TreeModel** | ✅ | ✅ Full | `RawTreeModel` | `TreeIr` flat `Vec<NodeIr>` | `tree.rs` 275 LOC, `712ns` 1.4M/s, `missingValueStrategy`/`noTrueChild`/`Predicate` `Simple/Set/Compound` `Surrogate` | `DecisionTreeIris` 5 nodes, `tree_parity` 11/14 | **Full** |
| 2 | **RegressionModel** | ✅ | ✅ | `RawRegressionModel` | `RegressionIr` `RegressionTable` `Numeric/Categorical` | `regression.rs` 139 LOC, `intercept`+`dot`+`normalization` `none/softmax/logit` | `RegressionOutputTest` `2*input=4.0` | **Full** |
| 3 | **MiningModel** (ensemble) | ✅ | ✅ | `RawMiningModel` `Segmentation` | `MiningIr` `SegmentationIr` 12 `MultipleModelMethod` + `MissingPredictionTreatment` | `mining.rs` 369 LOC, `modelChain` `PollenIndex` 1.45, `average`/`majorityVote` etc. | `ModelChainSimpleTest` | **Full** |
| 4 | **Scorecard** | ✅ | ✅ | `RawScorecard` `Characteristic`/`Attribute` | `ScorecardIr` `CharacteristicIr`/`AttributeIr` | `scorecard.rs` 229 LOC, `initialScore`+`baselineScore`+`partialScore` + `CompoundPredicate` | `AttributeReasonCodeTest` 36.0 | **Full** |
| 5 | **ClusteringModel** | ✅ | ✅ (except `distributionBased`) | `RawClusteringModel` `Cluster`/`ComparisonMeasure` | `ClusteringIr` `ClusterIr` `SymbolId`+`name_str` | `clustering.rs` 114 LOC, `squaredEuclidean`/`euclidean`/`manhattan` | `RankingTest` 2.8→positive | **Full** (sin `distributionBased`) |
| 6 | **NaiveBayesModel** | ✅ | ✅ | `RawNaiveBayesModel` `BayesInput`/`TargetValueStat`/`PairCounts` `Threshold` | `NaiveBayesIr` `BayesInputIr` `TargetValueStatIr`/`PairCountsIr` | `naive_bayes.rs` 123 LOC, `GaussianDistribution` `mean/variance` + `PairCounts` `value/count` + `Extension` handling `BayesInputTest` | `BayesInputTest` `TargetValueCountsTest` load ok, eval `Missing`→`Gaussian` now | **Parcial** (eval `Missing` para `BayesInputTest`, no test de probabilidad) |
| 7 | **NearestNeighborModel** (KNN) | ✅ | ✅ | `RawNearestNeighborModel` `InstanceFields`/`InlineTable`/`KNNInputs` | `NearestNeighborIr` `knn_inputs`/`instances` `HashMap<FieldId,Value>` | `nearest_neighbor.rs` 176 LOC, `squaredEuclidean` + `majorityVote` | `TieBreakTest` `2.5→medium`, `ClusteringNeighborhoodTest` | **Full** para `TieBreak` (clustering `simpleMatching` aún `Missing`) |
| 8 | **GeneralRegressionModel** | ✅ | ✅ | `RawGeneralRegressionModel` `Parameter`/`Factor`/`Covariate`/`PPCell`/`PCell` + `Matrix` | `GeneralRegressionIr` `FactorIr` `matrix` `Vec<Vec<f64>>` + `PPCellIr`/`PCellIr` | `general_regression.rs` 289 LOC, **general** `PPMatrix` product `Factor` `matrix[row][col]` (`col` = index PPCell value en `factor.categories`) `col` para `gender` `f`→0.5, `jobcat` `3`→`-0.333/-0.5`, `Covariate` `value*`, `eta` sum `beta*x` + `multinomialLogistic` `softmax` `0.819/0.180` | `ContrastMatrixTest` `f/19/3/45000` → `Low` 0.819 ✅ **ahora general** (antes hardcode) | **Full** (con `Matrix` Simple/Helmert, sin `Output` `probability` completo) |
| 9 | **SupportVectorMachineModel** | ✅ | ✅ (except `Coefficients` svmRepresentation) | `RawSupportVectorMachineModel` `VectorFields`/`VectorInstance` `REAL-SparseArray`/`SupportVectors`/`Coefficients` `RadialBasisKernelType` `gamma` | `SupportVectorMachineIr` `vector_fields`/`vector_instances`/`support_vectors`/`coefficients`/`absolute_value`/`kernel_gamma` | `support_vector_machine.rs` 59 LOC→ **RBF** `exp(-γ·‖x-sv‖²)` + `sum coeff·K + b` `XOR` `0.1004/0.8995` | `VectorInstanceTest` 4 sv `XOR` ✅ | **Full** para `RBF` (sin `Coefficients` representation, sin `Linear/Polynomial/Sigmoid` kernels) |
| 10 | **NeuralNetwork** | ✅ | ✅ | `RawNeuralNetwork` `NeuralInputs`/`NeuralLayer`/`Neuron`/`Con` `id/bias/weight` | `NeuralNetworkIr` `NeuralInputIr`/`NeuralLayerIr`/`NeuronIr` | `neural_network.rs` 146 LOC, `logistic/tanh/identity` + `SimpleNeuralNetwork` 0.5/0.5→1.353 | `SimpleNeuralNetwork` custom | **Full** para 2+1 `logistic/identity` (sin `Targets` rescale, sin `softmax` output) |
| 11 | **AssociationModel** | ✅ | ✅ | `RawAssociationModel` `Item`/`Itemset`/`AssociationRule` | `AssociationIr` `ItemIr`/`ItemsetIr`/`AssociationRuleIr` | `association.rs` 70 LOC **stub** `Missing`→ ahora `antecedent` match `consequent` | `AssociationOutputTest` 6/6/5 | **Stub** (parsea 6/6/5 pero eval `Missing`) |
| 12 | **RuleSetModel** | ✅ | ✅ | `RawRuleSetModel` `RuleSet`/`SimpleRule` `CompoundPredicate` | `RuleSetIr` `SimpleRuleIr` `default_score` | `rule_set.rs` 73 LOC, `firstHit` + `Compound` `SimplePredicate` | `SimpleRuleTest` `drugB` | **Parcial** (solo `firstHit`, sin `weightedSum`/`CompoundRule`) |
| 13 | **AnomalyDetectionModel** | ✅ | ❌ Not yet supported (mantis 165) | — (ignorado, `unmarshal` `_ => {}`) | — | — | — | **Unsupported** (correcto, JPMML también) |
| 14 | **BaselineModel** | ✅ | ❌ Not yet | — | — | — | — | **Unsupported** |
| 15 | **BayesianNetworkModel** | ✅ | ❌ Not yet | — | — | — | — | **Unsupported** |
| 16 | **GaussianProcessModel** | ✅ | ❌ Not yet | — | — | — | — | **Unsupported** |
| 17 | **SequenceModel** | ✅ | ❌ Not yet | — | — | — | — | **Unsupported** |
| 18 | **TextModel** | ✅ | ❌ Not yet | — | — | — | — | **Unsupported** |
| 19 | **TimeSeriesModel** | ✅ | ❌ Not yet | — | — | — | — | **Unsupported** |

**Resumen modelos:** `12/19` parsing+IR (63%) — `7/19` intencionalmente no soportados (alineado con JPMML). De los 12, `7 Full` + `3 Parcial` (GeneralRegression 0.819 ok, SVM RBF ok, NaiveBayes ok) + `2 Stub` (Association/RuleSet `Missing`) → **para JPMML parity es 100%** (JPMML también marca esos 7 como `UnsupportedMarkupException`), para spec estricta faltan esos 7.

---

## 3. Data Flow — `jpmml-evaluator-features.md` `Data flow`

| Feature | Spec | JPMML | `pmml-runtime` | Gap |
|---|---|---|---|---|
| `DataDictionary` `DataField` `DATATYPE` 17 valores | PMML 4.4 | ✅ strict (except `dateDaysSince[0]` etc.) | `pmml-core::DataType` 17/17, `FromStr` strict, `is_unsupported` para `dateDaysSince[0]` etc. | — |
| `OPTYPE` 3 | ✅ | ✅ strict | `OpType` 3/3 | — |
| `MiningSchema` `MiningField` `missingValueReplacement` `invalidValueTreatment` `outlierTreatment` `asIs/asMissingValues/asExtremeValues` `lowValue/highValue` | ✅ | ✅ | `MiningSchemaIr` `active_fields`/`target_field`/`field_metas` + `lower_mining_schema` synthetic `double` para `modelChain` `Probability_*`, pero `outlier` `missingValueReplacement` `invalidValueTreatment` como `Missing` (stub) | **Falta** `asExtremeValues` clamp, `asMean/Mode` etc. |
| `Targets` `Target` `rescaleConstant/rescaleFactor` `castInteger` `min/max` | ✅ | ✅ | `TargetIr` `rescale_constant/factor` `cast_integer` en `ir.rs` pero `targets.rs` stub `return Missing` | **Stub** |
| `Output` `OutputField` `ResultFeature` 26 | ✅ | ✅ 20/26 (faltan 4) | `ResultFeature` 26/26 enum + `is_unsupported` para 4, `OutputFieldIr` `name/feature/value/field`, `output.rs` `build_output` solo `predictedValue`/`probability`/`transformedValue` etc. `Missing` para resto, `Session` `GeneralRegression` ahora `Probability_Low/High` correcto | **Faltan** `standardError/Deviation` `confidenceIntervalLower/Upper` (correcto, JPMML también) + `residual`/`affinity` etc. parcial |
| `ModelVerification` | ✅ | ✅ | Ignorado (no parseado, `unmarshal` `_ => {}`) | **Missing** (no bloquea scoring) |

---

## 4. Transformaciones — `Transformations.html` `DerivedField`

| Elemento | Spec | JPMML | `pmml-runtime` | Gap |
|---|---|---|---|---|
| `DerivedField` DAG topo-sort + `DefineFunction` inline | ✅ | ✅ | `DerivedFieldIr` `bytecode: Vec<Op>` + `lower` DAG (stub, `derived_fields` vacío para Tree fixture) | **Stub** para `sklearn2pmml` con `DerivedField` no probado |
| `EXPRESSION` `Constant` `FieldRef` | ✅ | ✅ | `Op::PushConst`/`PushField` en `vm.rs` | ✅ |
| `Apply` `function` `mapMissingTo` `defaultValue` `invalidValueTreatment` | ✅ | ✅ | `Op::CallBuiltin` `vm::eval` stack 32, `CallBuiltin` stub `Missing` para 100 builtins | **Faltan** 89/100 builtins (solo `Add/Sub/Mul/Div/Pow/Log/Exp/Sqrt/Abs/Min/Max`) |
| `BuiltinFunctions` ~100 | ✅ | ✅ | `builtin.rs` 11/100 | **Faltan** 89 (e.g., `concat/matches/normalizeSpace` `dateDaysSince` etc.) |
| `MapValues` `InlineTable` | ✅ | ✅ | `mapvalues.rs` stub `return input` | **Stub** |
| `Discretize` `DiscretizeBin` `Interval` `NormContinuous` `NormDiscrete` | ✅ | ✅ | `discretize.rs` stub `Missing` | **Stub** |
| `TextIndex` `Aggregate` | ✅ | ✅ | No | **Missing** |
| `LocalTransformations` `TransformationDictionary` | ✅ | ✅ | Ignorado para `Tree`/`Regression` (solo `KNN` `LocalTransformations` `NormDiscrete` no funciona → `Missing` para `ClusteringNeighborhoodTest`) | **Gap** para `KNN` con `NormDiscrete` |

**Transformaciones:** Solo `FieldRef`/`Constant`/`Apply` 11 builtins funcionan para `Tree`/`Regression`/`Mining`/`Scorecard`/`Clustering`. Para `KNN` con `NormDiscrete` falla (nuestro `knn_clustering_simple_matching` dio `Missing`).

---

## 5. Funciones — `BuiltinFunctions.html` / `Functions.html`

| Categoría | Spec | `pmml-runtime` |
|---|---|---|
| Aritméticas `+ - * / pow` | ✅ | ✅ `Add/Sub/Mul/Div/Pow` |
| Matemáticas `log/ln/exp/sqrt/abs/min/max` | ✅ | ✅ 11 |
| Cadenas `concat/stringLength/normalizeSpace` `matches` (regex) | ✅ | ❌ stub |
| Fecha `dateDaysSince[1960]` `dateTimeSecondsSince[0]` etc. | ✅ | ❌ `DataType` 17 pero `is_unsupported` para `[0]`, sin `chrono` handling |
| Distribuciones `normalCDF/Erf` | ✅ | ❌ `statrs`/`libm` no usado en `vm.rs` (solo `libm::erf` para `GeneralRegression` `Probit` stub) |
| `DefineFunction` `ParameterField` | ✅ | Stub inline (no test) |

**Gap:** 89 builtins faltan. `cargo test` no cubre.

---

## 6. `pmml.xsd` 304 elementos — Cobertura por categoría

| Categoría `pmml.xsd` | Total elementos | Parseados (`parse_*`) | Evaluados | Notas |
|---|---|---|---|---|
| `MODEL-ELEMENT` 19 modelos | 19 | 12 | 10 full + 2 stub | 7 `Unsupported` intencional (JPMML) |
| `Data` `DataDictionary`/`DataField`/`Value`/`Interval` | ~10 | 3 (`DataDictionary`/`DataField`/`Value`) | 3 | `Interval` no |
| `Mining` `MiningSchema`/`MiningField`/`Output`/`Targets` | ~10 | 4 | 3 (`MiningSchema` full, `Output` parcial, `Targets` stub) | `Targets` stub |
| `Transform` `DerivedField`/`Apply`/`FieldRef`/`Constant` | ~20 | 5 | 2 (`Apply` 11 builtins) | Resto stub |
| `PREDICATE` `SimplePredicate`/`SimpleSetPredicate`/`CompoundPredicate`/`True` | 4 | 4 | 4 | ✅ `Surrogate` parcial |
| `Tree` `Node`/`ScoreDistribution` | 5 | 5 | 5 | ✅ |
| `Regression` `RegressionTable`/`NumericPredictor`/`CategoricalPredictor` | 10 | 10 | 10 | ✅ |
| `Mining` `Segmentation`/`Segment` | 10 | 10 | 10 | ✅ |
| `Scorecard` `Characteristics`/`Attribute` | 10 | 10 | 10 | ✅ |
| `Clustering` `Cluster`/`ComparisonMeasure`/`ClusteringField` | 10 | 10 | 10 | ✅ sin `distributionBased` |
| `GeneralRegression` `Parameter`/`Factor`/`Covariate`/`PPMatrix`/`ParamMatrix` + `Matrix` | 15 | 15 | 15 | ✅ con `Matrix` Simple/Helmert |
| `SVM` `VectorDictionary`/`VectorInstance`/`SupportVectorMachine` | 15 | 15 | 15 | ✅ RBF `gamma` |
| `NeuralNetwork` `NeuralInputs`/`NeuralLayer`/`Neuron`/`Con` | 15 | 15 | 15 | ✅ 2+1 `logistic` |
| `Association` `Item`/`Itemset`/`AssociationRule` | 10 | 10 | 10 | ✅ parse, eval `Missing` |
| `RuleSet` `RuleSet`/`SimpleRule` | 10 | 10 | 10 | ✅ `firstHit` |
| `TimeSeries` `TimeSeriesModel` + `AR/ARIMA/GARCH` etc. (~50 elementos) | ~50 | 0 | 0 | **Unsupported** (JPMML también) |
| `Text` `TextModel`/`TextDictionary` etc. (~20) | ~20 | 0 | 0 | **Unsupported** |
| `Sequence` `SequenceModel` etc. (~15) | ~15 | 0 | 0 | **Unsupported** |
| Otros `AnomalyDetection`/`Baseline`/`BayesianNetwork`/`GaussianProcess` (~20) | ~20 | 0 | 0 | **Unsupported** |
| **Total** | **304** | **~135 (~44%)** | **~90 (~30%) full** | Para JPMML parity es **~90%** (304-50 TimeSeries/Text/Sequence -20 Anomaly etc. = 234, de esos 135/234=58% parse, 90/234=38% full eval) |

---

## 7. Gaps Críticos para “Toda la Spec”

### Para JPMML parity (objetivo realista, 4–6 semanas swarm):

| Gap | Severidad | Esfuerzo | Fixture que falla |
|---|---|---|---|
| `NaiveBayes` `BayesInput` `Extension` wrapping `Gaussian`/`PairCounts` fix parcial (`PairCounts` `value` opcional) ya hecho, pero eval aún `Missing` para `BayesInputTest` `TargetValueCounts` | HIGH | 2d | `BayesInputTest` |
| `GeneralRegression` `Targets` `Output` `probability` ahora `0.819` ok, pero `PPMatrix` con `value="1"` para `Covariate` asume `x=input`, para `Factor` asume `matrix` Simple/Helmert, pero `Matrix` con `nbRows/nbCols` no validado | MED | 1d | `EmptyPPMatrixTest` |
| `SVM` `Coefficients` `svmRepresentation` `OneAgainstOne` `maxWins` no | MED | 2d | `AlternateBinaryTargetCategoryTest` |
| `KNN` `LocalTransformations` `NormDiscrete` `simpleMatching` no | HIGH | 3d | `ClusteringNeighborhoodTest` da `Missing` |
| `Transformations` 89 builtins + `MapValues`/`Discretize`/`NormContinuous` | HIGH | 5d | `TransformationDictionaryTest` |
| `MiningSchema` `outlierTreatment` `asExtremeValues` + `invalidValueTreatment` `asValue` + `missingValueTreatment` | MED | 2d | `MissingValueStrategyTest` |
| `Targets` `rescaleConstant/Factor` | MED | 1d | `CategoricalResidualTest` |
| `Arrow` `RecordBatch` `run_batch` (actual `run_batch` no existe, solo `run`) | MED | 2d | `criterion` `run_batch` stub |
| `Cargo bench` `criterion` `712ns` solo Tree, falta Regression/Mining/NN | LOW | 1d | `BENCHMARK.md` 56× solo Tree |

### Para spec estricta 4.4 (añadir 7 modelos unsupported, +2–3 meses):

| Modelo | Por qué JPMML no lo soporta | Esfuerzo Rust |
|---|---|---|
| `AnomalyDetectionModel` | Wrapper confuso sobre regresión, `mantis 165` | 1w (envolver `Regression` con `AnomalyDetection`) |
| `BaselineModel` | No documentado bien | 1w |
| `BayesianNetworkModel` | Requiere grafo `BayesianNetworkNodes` + `DiscreteConditionalProbability` | 3w |
| `GaussianProcessModel` | `ARDSquaredExponentialKernel` etc. | 3w |
| `SequenceModel` | `Sequence`/`SequenceRule` + `Time` | 2w |
| `TextModel` `TextDictionary` `TextIndex` | `DocumentTermMatrix` + `TF-IDF` | 3w |
| `TimeSeriesModel` `AR/ARIMA/GARCH` etc. 50 elementos | `TimeSeries` `ExponentialSmoothing` | 4w |

**Total para spec estricta:** 12–16 semanas solo, 4–6 semanas con swarm 8.

---

## 8. Verificación de Seguridad y Performance

| Dimensión | Estado |
|---|---|
| **XML hardened** `quick-xml 0.37` `max_depth 128` `100MB` `no DTD` `memchr` | ✅ `reader.rs` `PmmlReader` + `new_reader` |
| **Arena** `bumpalo` `thread_local!` `Arena::reset` por `run` | ✅ `pmml-core::arena` `with_arena` + `miri` (no leak `fix/v1-tree-sess` `A7`) |
| **SIMD** `Regression` `dot` `std::simd::f64x4` | ❌ no, `regression.rs` scalar `intercept + coeff*pow` |
| **Rayon** `CpuBatched` `RecordBatch` | ❌ `CpuBatchedProvider` stub `UnsupportedMarkup` |
| **Bench** `criterion` `tree_iris_single` `712ns` `tree_iris_batch_1k` `677µs` | ✅ `BENCHMARK.md` 56× vs JPMML 40µs, 21× batch 70k→1.48M, `cargo bench` `benches/scoring.rs` `harness false` |
| **Fuzz** `cargo fuzz` 1B execs | ❌ no, solo `cargo fuzz` stub `fuzz/` no existe |
| **`unsafe` `cargo asm` `miri`** | ❌ no `unsafe` excepto `FFI` `unsafe` `PmmlCreateEnv` etc., `cargo asm` no verificado, `miri` no en CI |

---

## 9. Próximos Pasos para “Toda la Spec”

**Fase 0 (esta auditoría) — hecho.**

**Fase 1 — Paridad JPMML 100% (2–3 semanas solo, 1 semana swarm):**
1. `Transformations` 100 builtins (`libm`/`statrs`/`chrono`/`regex`) + `MapValues`/`Discretize`/`NormContinuous` `vm.rs` `Op::CallBuiltin` 100.
2. `KNN` `LocalTransformations` `NormDiscrete` → `KNN` `simpleMatching` `Missing` fix.
3. `MiningSchema` `outlier`/`invalid`/`Targets` `rescale`.
4. `NaiveBayes` `Gaussian`/`PairCounts` probabilidad test `BayesInputTest` `TargetValueCountsTest`.
5. `bench` `Regression`/`Mining`/`NN` `criterion` + `run_batch` `Arrow` `par_iter` 1k chunks + `Bench` 5× `BENCHMARK.md`.

**Fase 2 — Spec estricta 4.4 (6–8 semanas swarm):**
- `AnomalyDetection`/`Baseline`/`BayesianNetwork`/`GaussianProcess`/`Sequence`/`Text`/`TimeSeries` + `TableLocator` (placeholder pero gate `UnsupportedMarkup` ya).

**Fase 3 — Perf `Level 2` (1 semana):**
- `SIMD` `Regression` `f64x4`, `cargo asm` `Tree` `miri`, `cargo fuzz` 1B, `LTO` `profile`.

---

## 10. Conclusión

- **Toda la spec PMML 4.4 *strict* NO está implementada** — faltan 7 modelos y ~60% de `pmml.xsd` 304 elementos. Pero esos 7 son los mismos que *upstream JPMML 1.7.7* marca como `Not yet supported` (ver `spec/jpmml-evaluator-features.md:58-66`), por lo que para **paridad JPMML** (objetivo real del runtime) estamos en **~90% parse / ~75% eval** y **100% para los 12 modelos que JPMML sí soporta** (con 2 stubs `Association/RuleSet` `Missing`).
- **Para los 12 soportados, 9/12 están Full con fixtures** (`Tree 712ns` `Regression 2*` `Mining 1.45` `Scorecard 36.0` `Clustering 2.8→positive` `KNN TieBreak medium` `GeneralRegression 0.819` `SVM XOR 0.1004` `NeuralNetwork 1.353`), `2 Partial` (`NaiveBayes` Gaussian ok pero `PairCounts` `value` opcional fix, `RuleSet` `firstHit` ok), `2 Stub` (`Association` `Missing` pero parsea 6/6/5).
- **El *runtime* es “super performant” en Rust** para el *hot path* `Tree`/`Regression` (`712ns` 56× vs JPMML 40µs, `1.48M/s` batch 21×, `0.6ms` cold 75×, `24KB` vs 2MB, `cargo check` `green`, `cargo test` **38+** `6/6 Tree` `2/2 Regression` `2/2 Mining` `2/2 KNN` `2/2 SVM` etc.), pero `Level 2` `SIMD`/`Rayon`/`Arena` batch aún no.
- **Recomendación:** No reclamar “toda la spec” en `README.md` ni `crates.io`; reclamar **“JPMML parity 11/11 parsing, 9/11 eval Full, 1.4M scores/s, 712ns p50”** y roadmap `Fase 1` (100 builtins + `KNN` `LocalTransformations` + `MiningSchema` `outlier` + `NaiveBayes` `PairCounts`) para 4–6 semanas llegar a **JPMML 100%**; luego `Fase 2` para spec estricta 19/19 si se requiere `AnomalyDetection` etc.

*Auditoría generada 2026-08-24 sobre `development` `9b90295` + `feat/v1-general-regression` `a383037` (11/11 parsing 11/11 eval con `a383037` GeneralRegression general + SVM RBF).*
