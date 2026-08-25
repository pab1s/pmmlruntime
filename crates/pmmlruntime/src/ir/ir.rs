//! Optimized, post-lower intermediate representation for PMML 4.4.
//!
//! The hot path (`pmml-session` + `pmml-evaluator`) reads only [`Ir`] and its
//! child structs. Cold-path parsing (`pmml-xml` → [`crate::xml::RawPmml`]) is
//! lowered once via [`crate::ir::lower::lower()`] into this representation.
//!
//! Key design choices:
//! - Stable [`crate::base::FieldId`] / [`crate::base::SymbolId`] assigned by [`crate::ir::Interner`].
//! - `TreeModel` nodes flattened to `Vec<NodeIr>` with root at index 0.
//! - `DerivedField` / `TransformationDictionary` sorted topologically.
//! - `DerivedFieldIr.bytecode` holds a `Vec<Op>` evaluated by `pmml-evaluator::vm`.
//! - All types are `Clone` and `Send + Sync` (via `Arc<Ir>` in `pmml-session`).

use crate::base::field::{DataType, OpType, ResultFeature};
use crate::base::{FieldId, SymbolId};
use smallvec::SmallVec;

/// Field metadata after `DataDictionary` + `MiningSchema` lowering.
///
/// One [`FieldMeta`] corresponds to a `DataField` entry, possibly enriched by
/// the model's `MiningField`. Synthetic fields created during lowering (for
/// example, an output of a [`DerivedFieldIr`] or a MiningModel synthetic
/// probability field) also produce a `FieldMeta` via
/// `lower::get_or_intern_field`.
///
/// # Examples
///
/// ```
/// use pmmlruntime::base::{FieldId, DataType, OpType};
/// use pmmlruntime::ir::{FieldMeta, OutlierTreatment, InvalidValueTreatment, MissingValueTreatment};
///
/// let meta = FieldMeta {
///     field_id: FieldId(0),
///     name: "age".into(),
///     data_type: DataType::Double,
///     op_type: OpType::Continuous,
///     values: vec![],
///     invalid_value_treatment: InvalidValueTreatment::ReturnInvalid,
///     invalid_value_replacement: None,
///     missing_value_replacement: None,
///     missing_value_treatment: MissingValueTreatment::AsIs,
///     outlier_treatment: OutlierTreatment::AsIs,
///     low_value: Some(0.0),
///     high_value: Some(120.0),
/// };
/// assert_eq!(meta.name, "age");
/// ```
#[derive(Debug, Clone)]
pub struct FieldMeta {
    /// Stable numeric identity for this field, dense per [`crate::ir::Interner`].
    ///
    /// Used as index into the hot-path `&[crate::base::Value]` array. See
    /// [`Ir::num_fields`] and `pmml-session` layout.
    pub field_id: FieldId,
    /// Original PMML `DataField/@name` or `DerivedField/@name`.
    pub name: String,
    /// Declared `DataDictionary/DataField/@dataType` (case-sensitive per `pmml.xsd`).
    ///
    /// Determines whether a value is stored as [`crate::base::Value::Discrete`]
    /// or [`crate::base::Value::Continuous`] on the hot path. See [`DataType`].
    pub data_type: DataType,
    /// Declared `DataField/@opType` (`categorical` / `ordinal` / `continuous`).
    ///
    /// Controls predicate semantics and outlier handling. See [`OpType`].
    pub op_type: OpType,
    /// Allowed discrete values from `DataField/Value` (for validation).
    ///
    /// Each entry is an interned [`SymbolId`]; empty when the field is continuous
    /// or when the PMML lists no restriction.
    pub values: Vec<SymbolId>,
    /// How to handle a value that violates `DataType` or `values` list.
    ///
    /// From `MiningField/@invalidValueTreatment`. See [`InvalidValueTreatment`].
    pub invalid_value_treatment: InvalidValueTreatment,
    /// Replacement string when `invalid_value_treatment` is `AsValue`.
    ///
    /// `None` when the treatment is not `asValue` or no replacement was specified.
    pub invalid_value_replacement: Option<String>,
    /// Replacement string when `missing_value_treatment` is `AsValue`.
    ///
    /// From `MiningField/@missingValueReplacement`.
    pub missing_value_replacement: Option<String>,
    /// How to handle `Missing` values for this field.
    ///
    /// From `MiningField/@missingValueTreatment`. See [`MissingValueTreatment`].
    pub missing_value_treatment: MissingValueTreatment,
    /// How to handle outliers for continuous fields.
    ///
    /// From `MiningField/@outliers`. See [`OutlierTreatment`].
    pub outlier_treatment: OutlierTreatment,
    /// Inclusive lower bound for outlier detection (`MiningField/@lowValue`).
    ///
    /// `None` when no bound was specified. Parsed as `f64` during lowering;
    /// `None` if the PMML string failed to parse.
    pub low_value: Option<f64>,
    /// Inclusive upper bound for outlier detection (`MiningField/@highValue`).
    pub high_value: Option<f64>,
}

impl Default for FieldMeta {
    fn default() -> Self {
        Self {
            field_id: FieldId(0),
            name: String::new(),
            data_type: DataType::String,
            op_type: OpType::Categorical,
            values: vec![],
            invalid_value_treatment: InvalidValueTreatment::ReturnInvalid,
            invalid_value_replacement: None,
            missing_value_replacement: None,
            missing_value_treatment: MissingValueTreatment::AsIs,
            outlier_treatment: OutlierTreatment::AsIs,
            low_value: None,
            high_value: None,
        }
    }
}

/// Outlier handling for continuous fields, per `MiningField/@outliers`.
///
/// Mirrors PMML XSD `OUTLIER-TREATMENT-METHOD`. Used by `pmml-evaluator` before
/// scoring to rewrite a value that lies outside `[low_value, high_value]`.
///
/// See [`FieldMeta::outlier_treatment`], [`FieldMeta::low_value`], [`FieldMeta::high_value`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutlierTreatment {
    /// Leave the value unchanged (`asIs`).
    AsIs,
    /// Treat outlying values as [`crate::base::Value::Missing`] (`asMissingValues`).
    AsMissingValues,
    /// Clamp to the nearer bound (`asExtremeValues`).
    AsExtremeValues,
}

/// How to handle a value that fails type or domain validation, per `MiningField/@invalidValueTreatment`.
///
/// Mirrors PMML XSD `INVALID-VALUE-TREATMENT-METHOD`. Lowering parses the
/// string via `parse_invalid_treatment` in [`mod@crate::ir::lower`]; default is
/// `ReturnInvalid` when absent.
///
/// See [`FieldMeta::invalid_value_treatment`] and
/// [`FieldMeta::invalid_value_replacement`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InvalidValueTreatment {
    /// Return an invalid result (score as missing / return error) (`returnInvalid`).
    ReturnInvalid,
    /// Use the value as-is despite validation (`asIs`).
    AsIs,
    /// Treat the value as missing (`asMissing`).
    AsMissing,
    /// Replace with [`FieldMeta::invalid_value_replacement`] (`asValue`).
    AsValue,
}

/// How to handle missing input values, per `MiningField/@missingValueTreatment`.
///
/// Mirrors PMML XSD `MISSING-VALUE-TREATMENT-METHOD`. When `AsValue`, the
/// replacement comes from [`FieldMeta::missing_value_replacement`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MissingValueTreatment {
    /// Propagate `Missing` unchanged (`asIs`).
    AsIs,
    /// Replace with mean of training column (`asMean`).
    AsMean,
    /// Replace with mode of training column (`asMode`).
    AsMode,
    /// Replace with median of training column (`asMedian`).
    AsMedian,
    /// Replace with [`FieldMeta::missing_value_replacement`] (`asValue`).
    AsValue,
    /// Fail scoring and return missing/invalid (`returnInvalid`).
    ReturnInvalid,
}

/// Role of a field in a `MiningSchema` (`MiningField/@usageType`).
///
/// `Active` fields are model inputs; `Predicted`/`Target` identify the output;
/// the others affect weighting or display but are not scored directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MiningFieldUsageType {
    /// Input feature (`active`).
    Active,
    /// Model output (`predicted`).
    Predicted,
    /// Alias for `Predicted` (`target` in PMML 4.4 `MiningSchema`).
    Target,
    /// Auxiliary column not used in scoring (`supplementary`).
    Supplementary,
    /// Grouping key for `MiningModel` (`group`).
    Group,
    /// Ordering key (`order`).
    Order,
    /// Per-row frequency weight (`frequencyWeight`).
    FrequencyWeight,
    /// Per-row analysis weight (`analysisWeight`).
    AnalysisWeight,
}

/// Flat mining schema after lowering.
///
/// Contains the active input fields (usage `active`) and at most one target,
/// plus per-field treatments already folded into each [`FieldMeta`]. Every
/// [`FieldId`] in `active_fields` / `target_field` / `field_metas` is present
/// in [`Ir::field_names`].
#[derive(Debug, Clone)]
pub struct MiningSchemaIr {
    /// Ordered active inputs (`usageType != target`). Scoring order follows this vec.
    pub active_fields: Vec<FieldId>,
    /// Predicted / target field, if declared (`usageType = target | predicted`).
    pub target_field: Option<FieldId>,
    /// Per-field metadata in `MiningField` order, with `MiningField` treatments folded in.
    pub field_metas: Vec<FieldMeta>,
    /// Global `missingValueReplacement` for backward compatibility.
    ///
    /// Set to the first field's `missing_value_replacement` if any. New code
    /// should read [`FieldMeta::missing_value_replacement`] instead.
    pub missing_value_replacement: Option<String>,
}

/// Derived field compiled to bytecode.
///
/// `TransformationDictionary/DerivedField` and model-local `DerivedField`s are
/// pooled, topologically sorted, and lowered to [`Op`] sequences. Evaluation
/// order follows the sorted `Ir.derived_fields` vector.
///
/// # Examples
///
/// ```
/// use pmmlruntime::base::{FieldId, DataType, OpType};
/// use pmmlruntime::ir::{DerivedFieldIr, Op};
/// let derived = DerivedFieldIr {
///     field_id: FieldId(2),
///     name: "log_age".into(),
///     data_type: DataType::Double,
///     op_type: OpType::Continuous,
///     bytecode: vec![Op::PushField(FieldId(0))],
/// };
/// assert_eq!(derived.name, "log_age");
/// ```
#[derive(Debug, Clone)]
pub struct DerivedFieldIr {
    /// Stable id assigned to the derived field itself.
    pub field_id: FieldId,
    /// `DerivedField/@name`.
    pub name: String,
    /// `DerivedField/@dataType`.
    pub data_type: DataType,
    /// `DerivedField/@optype`.
    pub op_type: OpType,
    /// Bytecode evaluated by `pmml-evaluator::vm::eval` in sorted order.
    pub bytecode: Vec<Op>,
}

/// Aggregate function for `Lag` (last `n` rows).
///
/// `None` returns a scalar lag; the others reduce a window of `n` values.
/// See [`Op::Lag`] and `PMML Lag` extension.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LagAggregate {
    /// No aggregation — return the single lagged value.
    None,
    /// Arithmetic mean over the window.
    Avg,
    /// Minimum over the window.
    Min,
    /// Maximum over the window.
    Max,
    /// Sum over the window.
    Sum,
    /// Product over the window.
    Product,
    /// Median over the window.
    Median,
    /// Standard deviation over the window.
    Stddev,
}

/// Bytecode for `Apply` / `MapValues` / `Discretize` and related expressions.
///
/// Evaluated by `pmml-evaluator::vm::eval`. Operands refer to
/// [`FieldId`] / [`SymbolId`] interned during lowering.
#[derive(Debug, Clone)]
pub enum Op {
    /// Push value of field `0` (`FieldRef` / implicit field) onto stack.
    PushField(FieldId),
    /// Push a constant (`Constant`): discrete symbol, continuous `f64`, or Missing.
    PushConst(SymbolIdOrContinuous),
    /// Call a pure builtin `function` with `arity` stack arguments.
    ///
    /// Arguments are popped in order produced by lowering; result is pushed.
    /// See [`BuiltinId`].
    CallBuiltin(BuiltinId, u8),
    /// Conditional jump when top-of-stack is `Missing` (for `IF` optimization).
    JumpIfMissing {
        /// Bytecode index to jump to when Missing.
        target: usize,
    },
    /// Single-input map-values table (`MapValues` with one `FieldColumnPair`).
    ///
    /// Evaluated via `pmml-evaluator` hash lookup on the input symbol.
    MapValues {
        /// Sorted map from input symbol to output symbol.
        table: Vec<(SymbolId, SymbolId)>,
        /// Value when no key matches; `None` yields Missing.
        default: Option<SymbolId>,
    },
    /// Multi-input map-values (`MapValues` with ≥2 inputs).
    MapValuesMulti {
        /// Input fields in table column order.
        inputs: Vec<FieldId>,
        /// Each row is `(Vec<input_symbols>, output_symbol)`.
        table: Vec<(Vec<SymbolId>, SymbolId)>,
        /// Default when no row matches.
        default: Option<SymbolId>,
    },
    /// Bin continuous value into discrete bins (`Discretize`).
    Discretize {
        /// Closed/open intervals sorted as in PMML; evaluated top-to-bottom.
        bins: Vec<DiscretizeBin>,
        /// Value when no bin matches and input is not missing.
        default_value: Option<SymbolId>,
        /// Value when input is Missing.
        map_missing_to: Option<SymbolId>,
    },
    /// Piecewise-linear normalization (`NormContinuous`).
    NormContinuous {
        /// Input field.
        field: FieldId,
        /// Ordered `LinearNorm` breakpoints.
        linear_norms: Vec<LinearNorm>,
    },
    /// Discrete indicator (`NormDiscrete`): `field == value ? 1 : 0`.
    NormDiscrete {
        /// Input field.
        field: FieldId,
        /// Value to compare.
        value: SymbolId,
        /// Replacement when input is Missing (typically `None` → Missing).
        map_missing_to: Option<f64>,
    },
    /// Temporal lag (`Lag`): value of `field` `n` rows ago, optionally aggregated.
    Lag {
        /// Input field.
        field: FieldId,
        /// Number of rows to look back.
        n: usize,
        /// Window aggregate.
        aggregate: LagAggregate,
    },
    /// Call a user-defined function from `TransformationDictionary/DefineFunction`.
    CallDefine {
        /// `DefineFunction/@name`.
        name: String,
        /// Declared arity.
        arity: u8,
    },
}

/// Constant or missing value pushed by [`Op::PushConst`].
///
/// `Continuous(f64)` carries an already-parsed `f64`; `Symbol(SymbolId)` is an
/// interned discrete string; `Missing` is PMML `Missing`.
#[derive(Debug, Clone, Copy)]
pub enum SymbolIdOrContinuous {
    /// Discrete constant (interned).
    Symbol(SymbolId),
    /// Numeric constant.
    Continuous(f64),
    /// PMML `Missing` (`Constant` with empty value or failed expression).
    Missing,
}

/// Single bin for [`Op::Discretize`].
///
/// PMML `Interval/@closure` controls inclusiveness of the bounds.
#[derive(Debug, Clone)]
pub struct DiscretizeBin {
    /// Output symbol for this bin (`DiscretizeBin/@binValue`).
    pub bin_value: SymbolId,
    /// Lower bound; `-inf` when omitted (`Interval/@leftMargin` absent).
    pub interval_low: f64,
    /// Upper bound; `+inf` when omitted.
    pub interval_high: f64,
    /// Whether `interval_low` is inclusive (closed).
    pub left_closed: bool,
    /// Whether `interval_high` is inclusive (closed).
    pub right_closed: bool,
}

/// Breakpoint for [`Op::NormContinuous`].
///
/// Two consecutive `LinearNorm`s define a line segment `norm = a*orig + b`.
#[derive(Debug, Clone, Copy)]
pub struct LinearNorm {
    /// Original value at breakpoint (`LinearNorm/@orig`).
    pub orig: f64,
    /// Normalized value at breakpoint (`LinearNorm/@norm`).
    pub norm: f64,
}

/// Pure builtin functions callable via `Apply` or dedicated PMML elements.
///
/// Lowering normalizes function names to canonical `fn` strings (for example,
/// `"add" | "+" → Add`) via `resolve_builtin` in [`mod@crate::ir::lower`].
/// Grouping below mirrors the JPMML function registry for discoverability.
///
/// # Examples
///
/// ```
/// use pmmlruntime::ir::BuiltinId;
/// let id = BuiltinId::Add;
/// assert_eq!(id as u8, BuiltinId::Add as u8);
/// // evaluators dispatch via match on BuiltinId
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BuiltinId {
    // ── Arithmetic ──────────────────────────────────────────────
    /// `add` / `+`: sum of all arguments.
    Add,
    /// `subtract` / `-`: first minus remaining.
    Sub,
    /// `multiply` / `*`: product.
    Mul,
    /// `divide` / `/`: first divided by second.
    Div,
    /// `pow`: `pow(base, exp)`.
    Pow,
    /// `log` / `ln`: natural logarithm.
    Log,
    /// `log10`: base-10 logarithm.
    Log10,
    /// Alias for `Log` (natural log).
    Ln,
    /// `exp`: `e^x`.
    Exp,
    /// `sqrt`: square root.
    Sqrt,
    /// `abs`: absolute value.
    Abs,
    /// `floor`: round down.
    Floor,
    /// `ceil`: round up.
    Ceil,
    /// `round`: to nearest integer (ties away from zero).
    Round,
    /// `remainder`: IEEE remainder.
    Remainder,
    // ── Trigonometry & hyperbolic ───────────────────────────────
    /// `sin`.
    Sin,
    /// `cos`.
    Cos,
    /// `tan`.
    Tan,
    /// `asin`.
    Asin,
    /// `acos`.
    Acos,
    /// `atan`.
    Atan,
    /// `sinh`.
    Sinh,
    /// `cosh`.
    Cosh,
    /// `tanh`.
    Tanh,
    // ── Minimum / maximum & aggregates ──────────────────────────
    /// `min`: minimum of arguments.
    Min,
    /// `max`: maximum of arguments.
    Max,
    /// `median`: median of arguments / table column.
    Median,
    /// `product`: product of arguments (distinct from [`Self::Mul`] which lower handles as variadic add-like).
    ProductOp,
    /// `sum`: sum of arguments.
    SumOp,
    /// `avg` / `average`: arithmetic mean.
    AvgOp,
    /// `mean`: alias for average.
    Mean,
    /// `stddev`: population standard deviation.
    StdDev,
    /// `variance`: population variance.
    Variance,
    // ── Modulo / rounding & C math ──────────────────────────────
    /// `modulo`: `a mod b`.
    Modulo,
    /// `rint`: round to nearest even.
    Rint,
    /// `expm1`: `exp(x) - 1` with high accuracy near 0.
    Expm1,
    /// `hypot`: `sqrt(x^2 + y^2)`.
    Hypot,
    /// `ln1p`: `ln(1 + x)` with high accuracy near 0.
    Ln1p,
    /// `atan2`: `atan2(y, x)`.
    Atan2,
    /// `cbrt`: cube root.
    Cbrt,
    /// `sign`: signum (-1, 0, 1).
    Sign,
    // ── String ──────────────────────────────────────────────────
    /// `uppercase` / `upperCase`.
    Uppercase,
    /// `lowercase` / `lowerCase`.
    Lowercase,
    /// `substring(string, pos, len)`.
    Substring,
    /// `trimBlanks`: trim leading/trailing blanks.
    TrimBlanks,
    /// `normalizeSpace`: trim + collapse whitespace.
    NormalizeSpace,
    /// `concat`: concatenate all arguments.
    Concat,
    /// `stringLength`: `len(s)`.
    StringLength,
    /// `replace(string, pattern, replacement)`.
    Replace,
    /// `matches(string, pattern)`: regex match.
    Matches,
    /// `formatNumber(number, pattern)`.
    FormatNumber,
    /// `formatDatetime(datetime, pattern)`.
    FormatDatetime,
    // ── Text index ──────────────────────────────────────────────
    /// `textIndex` / `TextIndex`: substring search index (1-based, 0 when not found).
    TextIndex,
    // ── Aggregate ───────────────────────────────────────────────
    /// `count` / `aggregateCount` over an inline table or batch.
    AggregateCount,
    /// `sum` / `aggregateSum`.
    AggregateSum,
    /// `average` / `avg` / `aggregateAverage`.
    AggregateAvg,
    /// `aggregateMin`.
    AggregateMin,
    /// `aggregateMax`.
    AggregateMax,
    /// `aggregateMultiset` (unique-count, not yet lowered as builtin).
    AggregateMultiset,
    // ── Temporal / Lag ──────────────────────────────────────────
    /// `lag`: previous row(s) value.
    Lag,
    // ── Date / time (chrono) ───────────────────────────────────
    /// `dateDaysSinceYear`.
    DateDaysSinceYear,
    /// `dateSecondsSinceYear`.
    DateSecondsSinceYear,
    /// `dateSecondsSinceMidnight`.
    DateSecondsSinceMidnight,
    /// `dateDaysSince[1960]` / `DataType::DateDaysSince1960` helper.
    DateDaysSince1960,
    /// `dateDaysSince[1970]`.
    DateDaysSince1970,
    /// `dateDaysSince[1980]`.
    DateDaysSince1980,
    /// `dateTimeSecondsSince[1960]`.
    DateTimeSecondsSince1960,
    /// `dateTimeSecondsSince[1970]`.
    DateTimeSecondsSince1970,
    /// `dateTimeSecondsSince[1980]`.
    DateTimeSecondsSince1980,
    /// `dateTimeSecondsSince[0]` (unsupported per JPMML, rejected in lower).
    DateTimeSecondsSince0,
    /// `timeSeconds`.
    TimeSeconds,
    // ── Distribution (statrs / libm) ────────────────────────────
    /// `normalCDF`.
    NormalCdf,
    /// `normalPDF`.
    NormalPdf,
    /// `normalIDF` (inverse CDF / quantile).
    NormalIdf,
    /// `stdNormalCDF`.
    StdNormalCdf,
    /// `stdNormalPDF`.
    StdNormalPdf,
    /// `stdNormalIDF`.
    StdNormalIdf,
    /// `erf`.
    ErfOp,
    // ── PMML Norm pseudo-builtins ───────────────────────────────
    /// `normContinuous` lowered as dedicated [`Op::NormContinuous`].
    NormContinuousOp,
    /// `normDiscrete` lowered as dedicated [`Op::NormDiscrete`].
    NormDiscreteOp,
    // ── Comparison / logical ────────────────────────────────────
    /// `equal`: `a == b`.
    Equal,
    /// `notEqual`: `a != b`.
    NotEqual,
    /// `lessThan`: `a < b`.
    LessThan,
    /// `lessOrEqual`: `a <= b`.
    LessOrEqual,
    /// `greaterThan`: `a > b`.
    GreaterThan,
    /// `greaterOrEqual`: `a >= b`.
    GreaterOrEqual,
    /// `and`: logical conjunction (all true).
    And,
    /// `or`: logical disjunction (any true).
    Or,
    /// `not`: logical negation.
    Not,
    /// `isMissing`: `value is Missing`.
    IsMissing,
    /// `isNotMissing`: `value is not Missing`.
    IsNotMissing,
    /// `isValid`: not missing and not outlier/invalid.
    IsValid,
    /// `isNotValid`: complement of `isValid`.
    IsNotValid,
    /// `isIn`: membership test (`x in {...}`).
    IsIn,
    /// `isNotIn`: negated membership.
    IsNotIn,
    // ── Conditional ─────────────────────────────────────────────
    /// `if`: `if(condition, then, else)`.
    If,
    // ── Misc ────────────────────────────────────────────────────
    /// `threshold(y, v)`: `y > v ? y : 0` helper for scorecard.
    Threshold,
    /// Unknown function → lowering emits `Missing` at runtime.
    Unknown,
}

/// Lowered PMML model — the sole scoring path selected per PMML file.
///
/// `RawPmml` contains at most one top-level model (plus optional Segmentation
/// segments). Lowering picks the present model and returns `MissingField` or
/// `UnsupportedMarkup` when none matches.
///
/// See each variant's struct for model-specific semantics and JPMML
/// comparability notes.
#[derive(Debug, Clone)]
pub enum ModelIr {
    /// `TreeModel` (classification or regression).
    Tree(TreeIr),
    /// `RegressionModel` (linear / logistic with normalization).
    Regression(RegressionIr),
    /// `MiningModel` (segmented / chained models).
    Mining(MiningIr),
    /// `Scorecard` (characteristics + attributes, reason codes).
    Scorecard(ScorecardIr),
    /// `ClusteringModel` (center-based, clusters + comparison measure).
    Clustering(ClusteringIr),
    /// `NaiveBayesModel` (BayesInputs / BayesOutput).
    NaiveBayes(NaiveBayesIr),
    /// `NearestNeighborModel` (k-NN with `InlineTable` instances).
    NearestNeighbor(NearestNeighborIr),
    /// `SupportVectorMachineModel` (vector fields + support vectors + kernels).
    SupportVectorMachine(SupportVectorMachineIr),
    /// `NeuralNetwork` (inputs → layers → neurons).
    NeuralNetwork(NeuralNetworkIr),
    /// `GeneralRegressionModel` (PPMatrix / ParamMatrix / factors / covariates).
    GeneralRegression(GeneralRegressionIr),
    /// `AssociationModel` (items / itemsets / rules).
    Association(AssociationIr),
    /// `RuleSetModel` (`RuleSet` with ordered simple rules).
    RuleSet(RuleSetIr),
}

/// Lowered `RegressionModel`.
///
/// Computes `y = intercept + Σ coeff * field^exponent` per `RegressionTable`,
/// then applies [`RegressionNormalizationMethod`].
#[derive(Debug, Clone)]
pub struct RegressionIr {
    /// `RegressionModel/@functionName` (`regression` / `classification`).
    pub function_name: String,
    /// Mining schema for this model (active fields + target).
    pub mining_schema: MiningSchemaIr,
    /// One table per target category (`targetCategory` discriminates multinomial).
    pub regression_tables: Vec<RegressionTableIr>,
    /// `RegressionModel/@normalizationMethod` (default `None`).
    pub normalization_method: RegressionNormalizationMethod,
    /// `Targets` rescaling / casting metadata.
    pub targets: Vec<TargetIr>,
    /// `Output` fields requested by the PMML.
    pub output: Vec<OutputFieldIr>,
}

/// One `RegressionTable` inside a [`RegressionIr`].
#[derive(Debug, Clone)]
pub struct RegressionTableIr {
    /// `RegressionTable/@intercept`.
    pub intercept: f64,
    /// `RegressionTable/@targetCategory` (interned when present, e.g., for softmax).
    pub target_category: Option<SymbolId>,
    /// Numeric predictors: `NumericPredictor/@name, exponent, coefficient`.
    pub numeric_predictors: Vec<NumericPredictorIr>,
    /// Categorical predictors: `CategoricalPredictor/@field, value, coefficient`.
    pub categorical_predictors: Vec<CategoricalPredictorIr>,
}

/// Numeric predictor contribution `coeff * field^exponent`.
#[derive(Debug, Clone)]
pub struct NumericPredictorIr {
    /// Input field.
    pub field: FieldId,
    /// `NumericPredictor/@coefficient`.
    pub coefficient: f64,
    /// `NumericPredictor/@exponent` (default 1).
    pub exponent: i32,
}

/// Categorical predictor contribution active when `field == value`.
#[derive(Debug, Clone)]
pub struct CategoricalPredictorIr {
    /// Input field.
    pub field: FieldId,
    /// Value that activates this predictor (interned).
    pub value: SymbolId,
    /// `CategoricalPredictor/@coefficient`.
    pub coefficient: f64,
}

/// Normalization applied after raw regression scores, per `RegressionModel/@normalizationMethod`.
///
/// Mirrors `REGRESSIONNORMALIZATIONMETHOD`. `None` returns raw scores directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegressionNormalizationMethod {
    /// No normalization (`none`).
    None,
    /// Divide by maximum (`simpleMax`): `x / max(x)`.
    SimpleMax,
    /// Softmax (`softmax`): `exp(x)/sum(exp(x))`.
    Softmax,
    /// Logit (`logit`): `1 / (1 + exp(-x))` (binary).
    Logit,
    /// Probit (`probit`): `Φ(x)` standard normal CDF.
    Probit,
    /// Complementary log-log (`cloglog`): `1 - exp(-exp(x))`.
    ClogLog,
    /// Exponential (`exp`): `exp(x)`.
    Exp,
    /// Double exponential (`loglog`): `exp(exp(x))`.
    Loglog,
    /// Cauchy (`cauchit`): `0.5 + atan(x)/π`.
    Cauchit,
}

/// Lowered `MiningModel` (segmented ensemble / `modelChain`).
#[derive(Debug, Clone)]
pub struct MiningIr {
    /// `MiningModel/@functionName`.
    pub function_name: String,
    /// Top-level mining schema.
    pub mining_schema: MiningSchemaIr,
    /// Segmentation defining how segment models are combined.
    pub segmentation: SegmentationIr,
    /// Targets for rescaling / casting.
    pub targets: Vec<TargetIr>,
    /// Output fields.
    pub output: Vec<OutputFieldIr>,
}

/// Segmentation of a [`MiningIr`].
#[derive(Debug, Clone)]
pub struct SegmentationIr {
    /// `Segmentation/@multipleModelMethod`.
    pub multiple_model_method: MultipleModelMethod,
    /// `Segmentation/@missingPredictionTreatment`.
    pub missing_prediction_treatment: MissingPredictionTreatment,
    /// Ordered segments; each has a predicate and a boxed model.
    pub segments: Vec<SegmentIr>,
}

/// Single segment inside a [`SegmentationIr`].
#[derive(Debug, Clone)]
pub struct SegmentIr {
    /// `Segment/@id` (may be `None` in older PMML).
    pub id: Option<String>,
    /// Predicate that selects this segment (`True` for the default segment).
    pub predicate: PredicateIr,
    /// `Segment/@weight` (default 1.0).
    pub weight: f64,
    /// Boxed sub-model (`Tree` or `Regression` currently).
    pub model: Box<ModelIr>,
}

/// How to combine predictions from multiple `MiningModel` segments.
///
/// Mirrors `MULTIPLE-MODEL-METHOD` in `pmml.xsd`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MultipleModelMethod {
    /// `majorityVote`.
    MajorityVote,
    /// `weightedMajorityVote`.
    WeightedMajorityVote,
    /// `average`.
    Average,
    /// `weightedAverage`.
    WeightedAverage,
    /// `median`.
    Median,
    /// `weightedMedian`.
    WeightedMedian,
    /// `max`.
    Max,
    /// `sum`.
    Sum,
    /// `weightedSum`.
    WeightedSum,
    /// `selectFirst`.
    SelectFirst,
    /// `selectAll`.
    SelectAll,
    /// `modelChain` — chain outputs as extra fields for later segments.
    ModelChain,
}

/// How a `MiningModel` handles a missing prediction from a segment.
///
/// Mirrors `MISSING-PREDICTION-TREATMENT`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MissingPredictionTreatment {
    /// Propagate missing (`returnMissing`).
    ReturnMissing,
    /// Skip the segment (`skipSegment`).
    SkipSegment,
    /// Continue to next segment (`continue`).
    Continue,
}

/// Lowered `Scorecard` model.
#[derive(Debug, Clone)]
pub struct ScorecardIr {
    /// `Scorecard/@functionName`.
    pub function_name: String,
    /// `Scorecard/@initialScore` (starting points).
    pub initial_score: f64,
    /// `Scorecard/@useReasonCodes`.
    pub use_reason_codes: bool,
    /// `Scorecard/@reasonCodeAlgorithm` (`pointsAbove` or `pointsBelow` etc.).
    pub reason_code_algorithm: String,
    /// Mining schema for inputs.
    pub mining_schema: MiningSchemaIr,
    /// Ordered characteristics.
    pub characteristics: Vec<CharacteristicIr>,
    /// Output fields (often `predictedValue` + `reasonCode`).
    pub output: Vec<OutputFieldIr>,
}

/// One `Characteristic` inside a [`ScorecardIr`].
#[derive(Debug, Clone)]
pub struct CharacteristicIr {
    /// `Characteristic/@name`.
    pub name: String,
    /// `Characteristic/@reasonCode` (when `useReasonCodes`).
    pub reason_code: Option<String>,
    /// `Characteristic/@baselineScore` (default 0.0).
    pub baseline_score: f64,
    /// Ordered attributes; first matching predicate contributes `partialScore`.
    pub attributes: Vec<AttributeIr>,
}

/// One `Attribute` inside a [`CharacteristicIr`].
#[derive(Debug, Clone)]
pub struct AttributeIr {
    /// `Attribute/@partialScore`.
    pub partial_score: f64,
    /// Predicate selecting this attribute.
    pub predicate: PredicateIr,
    /// `Attribute/@reasonCode` (overrides characteristic code when matched).
    pub reason_code: Option<String>,
}

/// Lowered `ClusteringModel`.
#[derive(Debug, Clone)]
pub struct ClusteringIr {
    /// `ClusteringModel/@functionName` (typically `clustering`).
    pub function_name: String,
    /// `ClusteringModel/@modelClass` (`centerBased` / `distributionBased`).
    pub model_class: String,
    /// `ClusteringModel/@numberOfClusters` (fallback to `clusters.len()`).
    pub number_of_clusters: usize,
    /// Mining schema (active fields must match `clusteringFields` subset).
    pub mining_schema: MiningSchemaIr,
    /// `ComparisonMeasure` kind (e.g., `euclidean`, `squaredEuclidean`).
    pub comparison_measure: String,
    /// Fields that define cluster coordinates.
    pub clustering_fields: Vec<FieldId>,
    /// Clusters with centroid / array.
    pub clusters: Vec<ClusterIr>,
    /// Output fields.
    pub output: Vec<OutputFieldIr>,
}

/// Single `Cluster` inside a [`ClusteringIr`].
#[derive(Debug, Clone)]
pub struct ClusterIr {
    /// Interned `Cluster/@name`.
    pub name: SymbolId,
    /// Display name (`Cluster/@name` raw string).
    pub name_str: String,
    /// Coordinate array aligned with `ClusteringIr.clustering_fields`.
    pub array: Vec<f64>,
}

/// Count of a single discrete target value, used in [`PairCountsIr`] and NaiveBayes.
#[derive(Debug, Clone)]
pub struct TargetValueCountIr {
    /// Interned `TargetValueCount/@value`.
    pub value: SymbolId,
    /// `TargetValueCount/@count`.
    pub count: f64,
}

/// Joint counts for one discrete input value vs all target values.
#[derive(Debug, Clone)]
pub struct PairCountsIr {
    /// Input value (e.g., `"sunny"`, interned).
    pub value: SymbolId,
    /// Per-target counts for this input value.
    pub target_counts: Vec<TargetValueCountIr>,
}

/// Gaussian statistics for a continuous input per target value.
///
/// Used when the Bayes input is continuous: `GaussianDistribution` per class.
#[derive(Debug, Clone)]
pub struct TargetValueStatIr {
    /// Target class value (interned).
    pub value: SymbolId,
    /// Mean `μ` (`GaussianDistribution/@mean`), if present.
    pub mean: Option<f64>,
    /// Variance `σ²` (`GaussianDistribution/@variance`), if present.
    pub variance: Option<f64>,
}

/// One `BayesInput` inside a [`NaiveBayesIr`].
#[derive(Debug, Clone)]
pub struct BayesInputIr {
    /// Input field.
    pub field: FieldId,
    /// Continuous per-class Gaussian stats (empty for discrete fields).
    pub target_value_stats: Vec<TargetValueStatIr>,
    /// Discrete joint counts (empty for continuous fields).
    pub pair_counts: Vec<PairCountsIr>,
}

/// Lowered `NaiveBayesModel`.
#[derive(Debug, Clone)]
pub struct NaiveBayesIr {
    /// `NaiveBayesModel/@functionName`.
    pub function_name: String,
    /// `NaiveBayesModel/@threshold` (count threshold for smoothing).
    pub threshold: f64,
    /// Mining schema.
    pub mining_schema: MiningSchemaIr,
    /// Output fields.
    pub output: Vec<OutputFieldIr>,
    /// Bayes inputs (one per field, mixed discrete / continuous).
    pub bayes_inputs: Vec<BayesInputIr>,
    /// Prior counts per target class.
    pub bayes_output_counts: Vec<TargetValueCountIr>,
}

/// Lowered `NearestNeighborModel` (k-NN).
#[derive(Debug, Clone)]
pub struct NearestNeighborIr {
    /// `NearestNeighborModel/@functionName`.
    pub function_name: String,
    /// `NearestNeighborModel/@numberOfNeighbors` (`k`).
    pub number_of_neighbors: usize,
    /// Mining schema.
    pub mining_schema: MiningSchemaIr,
    /// Output fields.
    pub output: Vec<OutputFieldIr>,
    /// `KNNInputs` fields (distance inputs).
    pub knn_inputs: Vec<FieldId>,
    /// Per-instance field values (`InlineTable` rows, as `FieldId → Value`).
    pub instances: Vec<std::collections::HashMap<crate::base::FieldId, crate::base::Value>>,
    /// Raw instance ids in row order (for entity id output).
    pub instance_ids: Vec<String>,
}

/// Lowered `SupportVectorMachineModel`.
#[derive(Debug, Clone)]
pub struct SupportVectorMachineIr {
    /// `SupportVectorMachineModel/@functionName`.
    pub function_name: String,
    /// Mining schema.
    pub mining_schema: MiningSchemaIr,
    /// Output fields.
    pub output: Vec<OutputFieldIr>,
    /// Ordered vector fields (`VectorDictionary/VectorFields/@field`).
    pub vector_fields: Vec<FieldId>,
    /// `(id, array)` pairs from `VectorDictionary` (id + dense vector).
    pub vector_instances: Vec<(String, Vec<f64>)>,
    /// `SupportVector/@vectorId` references in definition order.
    pub support_vectors: Vec<String>,
    /// `Coefficients/@value` in definition order.
    pub coefficients: Vec<f64>,
    /// Absolute value bias (`SupportVectorMachine/@absoluteValue` or `Coefficients` sum).
    pub absolute_value: f64,
    /// `Kernel/@gamma` (RBF `exp(-γ‖x-sv‖²)`).
    pub kernel_gamma: f64,
}

/// Single neural input (`NeuralInput/@id` → `FieldId`).
#[derive(Debug, Clone)]
pub struct NeuralInputIr {
    /// `NeuralInput/@id`.
    pub id: String,
    /// Backing field.
    pub field: FieldId,
}

/// Single neuron with bias and inbound weighted connections.
#[derive(Debug, Clone)]
pub struct NeuronIr {
    /// `Neuron/@id`.
    pub id: String,
    /// `Neuron/@bias` (default 0.0).
    pub bias: f64,
    /// Inbound `Con/@from → @weight`.
    pub cons: Vec<(String, f64)>,
}

/// One layer of neurons with a shared activation function.
#[derive(Debug, Clone)]
pub struct NeuralLayerIr {
    /// `NeuralLayer/@numberOfNeurons` (or `neurons.len()` fallback).
    pub number_of_neurons: usize,
    /// `NeuralLayer/@activationFunction` (e.g., `logistic`, `tanh`, `identity`).
    pub activation_function: String,
    /// Neurons in layer order.
    pub neurons: Vec<NeuronIr>,
}

/// Lowered `NeuralNetwork`.
#[derive(Debug, Clone)]
pub struct NeuralNetworkIr {
    /// `NeuralNetwork/@functionName`.
    pub function_name: String,
    /// Mining schema.
    pub mining_schema: MiningSchemaIr,
    /// Output fields.
    pub output: Vec<OutputFieldIr>,
    /// External inputs (`NeuralInputs`).
    pub neural_inputs: Vec<NeuralInputIr>,
    /// Layers from input-most to output-most.
    pub neural_layers: Vec<NeuralLayerIr>,
    /// `NeuralNetwork/@activationFunction` top-level default.
    pub activation_function: String,
}

/// Parameter of a [`GeneralRegressionIr`] (`Parameter/@name`, optional `@label`).
#[derive(Debug, Clone)]
pub struct ParameterIr {
    /// `Parameter/@name`.
    pub name: String,
    /// `Parameter/@label` (human-readable).
    pub label: Option<String>,
}

/// Factor (categorical covariate) for [`GeneralRegressionIr`].
#[derive(Debug, Clone)]
pub struct FactorIr {
    /// Field id for the factor (`Factor/@name`).
    pub name: FieldId,
    /// Ordered categories (`Factor/@categories` resolved to symbols).
    pub categories: Vec<SymbolId>,
    /// Contrast matrix `rows=categories, cols=parameters`.
    pub matrix: Vec<Vec<f64>>,
}

/// Cell in the predictor-to-parameter matrix (`PPCell`).
#[derive(Debug, Clone)]
pub struct PPCellIr {
    /// `PPCell/@value` (category value, interned).
    pub value: SymbolId,
    /// `PPCell/@predictorName`.
    pub predictor_name: String,
    /// `PPCell/@parameterName`.
    pub parameter_name: String,
}

/// Cell in the parameter matrix (`PCell`).
#[derive(Debug, Clone)]
pub struct PCellIr {
    /// `PCell/@targetCategory` (interned when present).
    pub target_category: Option<SymbolId>,
    /// `PCell/@parameterName`.
    pub parameter_name: String,
    /// `PCell/@beta`.
    pub beta: f64,
}

/// Lowered `GeneralRegressionModel` (GLM / multinomial logistic).
#[derive(Debug, Clone)]
pub struct GeneralRegressionIr {
    /// `GeneralRegressionModel/@functionName`.
    pub function_name: String,
    /// Mining schema.
    pub mining_schema: MiningSchemaIr,
    /// Output fields.
    pub output: Vec<OutputFieldIr>,
    /// `GeneralRegressionModel/@modelType` (`regression`, `generalLinear`, `multinomialLogistic`, …).
    pub model_type: Option<String>,
    /// `GeneralRegressionModel/@targetVariableName`.
    pub target_variable_name: Option<String>,
    /// `GeneralRegressionModel/@targetReferenceCategory`.
    pub target_reference_category: Option<SymbolId>,
    /// `ParameterList` in order.
    pub parameters: Vec<ParameterIr>,
    /// `FactorList`.
    pub factors: Vec<FactorIr>,
    /// `CovariateList` (continuous predictors) as field ids.
    pub covariates: Vec<FieldId>,
    /// `PPMatrix` cells.
    pub pp_matrix: Vec<PPCellIr>,
    /// `ParamMatrix` cells (betas).
    pub param_matrix: Vec<PCellIr>,
}

/// Item in an [`AssociationIr`] (`Item/@id` and category value).
#[derive(Debug, Clone)]
pub struct ItemIr {
    /// `Item/@id`.
    pub id: String,
    /// `Item/@value` category (interned).
    pub value: SymbolId,
}

/// Itemset (`Itemset/@id` with item refs) in an [`AssociationIr`].
#[derive(Debug, Clone)]
pub struct ItemsetIr {
    /// `Itemset/@id`.
    pub id: String,
    /// `Itemset/ItemRef/@itemRef` in order.
    pub item_ids: Vec<String>,
}

/// Association rule: `antecedent => consequent` with support/confidence/lift.
#[derive(Debug, Clone)]
pub struct AssociationRuleIr {
    /// `AssociationRule/@antecedent` (itemset id).
    pub antecedent: String,
    /// `AssociationRule/@consequent` (itemset id).
    pub consequent: String,
    /// `AssociationRule/@support`.
    pub support: f64,
    /// `AssociationRule/@confidence`.
    pub confidence: f64,
    /// `AssociationRule/@lift`.
    pub lift: f64,
}

/// Lowered `AssociationModel` (association rules).
#[derive(Debug, Clone)]
pub struct AssociationIr {
    /// `AssociationModel/@functionName` (typically `associationRules`).
    pub function_name: String,
    /// Mining schema.
    pub mining_schema: MiningSchemaIr,
    /// Output fields.
    pub output: Vec<OutputFieldIr>,
    /// `AssociationModel/Item`s.
    pub items: Vec<ItemIr>,
    /// `AssociationModel/Itemset`s.
    pub itemsets: Vec<ItemsetIr>,
    /// `AssociationModel/AssociationRule`s.
    pub rules: Vec<AssociationRuleIr>,
}

/// Single simple rule (`SimpleRule/@id, @score` with predicate).
#[derive(Debug, Clone)]
pub struct SimpleRuleIr {
    /// `SimpleRule/@id` (may be `None`).
    pub id: Option<String>,
    /// `SimpleRule/@score` (interned category or numeric string).
    pub score: SymbolId,
    /// Predicate that activates the rule.
    pub predicate: PredicateIr,
}

/// Lowered `RuleSetModel` (ordered rules + optional default).
#[derive(Debug, Clone)]
pub struct RuleSetIr {
    /// `RuleSetModel/@functionName`.
    pub function_name: String,
    /// Mining schema.
    pub mining_schema: MiningSchemaIr,
    /// Output fields.
    pub output: Vec<OutputFieldIr>,
    /// `RuleSet/@defaultScore` (interned) when no rule fires.
    pub default_score: Option<SymbolId>,
    /// Rules in PMML order (first match wins in evaluator).
    pub rules: Vec<SimpleRuleIr>,
}

/// Lowered `TreeModel`.
#[derive(Debug, Clone)]
pub struct TreeIr {
    /// `TreeModel/@functionName` (`classification` / `regression`).
    pub function_name: String,
    /// `TreeModel/@missingValueStrategy` (how to route missing during traversal).
    pub missing_value_strategy: MissingValueStrategy,
    /// `TreeModel/@noTrueChildStrategy` (what to return when no child predicate holds).
    pub no_true_child_strategy: NoTrueChildStrategy,
    /// Flat node list; root at index 0. See [`NodeIr`].
    pub nodes: Vec<NodeIr>,
    /// Mining schema for this tree (active inputs + optional target).
    pub mining_schema: MiningSchemaIr,
    /// `Targets` (rescaling / casting after tree score).
    pub targets: Vec<TargetIr>,
    /// `Output` fields (probability, predictedDisplayValue, etc.).
    pub output: Vec<OutputFieldIr>,
}

/// How to handle `Missing` during tree traversal (`TreeModel/@missingValueStrategy`).
///
/// Mirrors JPMML behaviour. The unsupported variants `WeightedConfidence` and
/// `AggregateNodes` are never produced by lowering (they would have been rejected
/// earlier), but are preserved in the enum for `verify_ir` parity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MissingValueStrategy {
    /// Use the last non-missing prediction on the path (`lastPrediction`).
    LastPrediction,
    /// Return `Missing` / null prediction (`nullPrediction`).
    NullPrediction,
    /// Follow `Node/@defaultChild` (`defaultChild`).
    DefaultChild,
    /// No special handling (`none`).
    None,
    /// JPMML-unsupported: weighted confidence fallback (`weightedConfidence`).
    WeightedConfidence,
    /// JPMML-unsupported: aggregate over leaves (`aggregateNodes`).
    AggregateNodes,
}

/// How to handle the case that no child predicate is true (`TreeModel/@noTrueChildStrategy`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoTrueChildStrategy {
    /// Return `Missing` / null (`returnNullPrediction`).
    ReturnNullPrediction,
    /// Return last prediction on the path (`returnLastPrediction`).
    ReturnLastPrediction,
}

/// Single node in a [`TreeIr`] (`Node/@id, @score` + predicate + children).
///
/// Nodes are stored flat: children are indices into `TreeIr.nodes`. `default_child`
/// is an index (not an id) resolved during lowering.
#[derive(Debug, Clone)]
pub struct NodeIr {
    /// `Node/@id` (stable identifier for `defaultChild` resolution).
    pub id: Option<String>,
    /// `Node/@score` (`Continuous(f64)` for regression, `Symbol` for classification).
    pub score: Option<SymbolIdOrContinuous>,
    /// Predicate tested at this node (`True` for the root).
    pub predicate: PredicateIr,
    /// Indices of child nodes in `TreeIr.nodes` (DFS order from `flatten_node`).
    pub children: Vec<usize>,
    /// Index of the `defaultChild` when `missingValueStrategy = defaultChild`.
    pub default_child: Option<usize>,
    /// `ScoreDistribution` (class probabilities per node, for `Output/probability`).
    pub score_distributions: Vec<ScoreDistributionIr>,
}

/// Predicate tested at a tree node or segment / rule.
#[derive(Debug, Clone)]
pub enum PredicateIr {
    /// Unconditional true (`True` / root predicate).
    True,
    /// `SimplePredicate`: `field operator value`.
    Simple {
        /// Field under test.
        field: FieldId,
        /// Operator (`equal`, `lessThan`, `isMissing`, etc.).
        operator: SimpleOperator,
        /// Value or `Missing` when operator is `isMissing` / `isNotMissing`.
        value: SymbolIdOrContinuous,
    },
    /// `SimpleSetPredicate`: `field isIn {values}` or `isNotIn`.
    SimpleSet {
        /// Field under test.
        field: FieldId,
        /// When `true` → `isIn`; `false` → `isNotIn`.
        is_in: bool,
        /// Set of values (already coerced per field `DataType`).
        array: Vec<SymbolIdOrContinuous>,
    },
    /// `CompoundPredicate`: `and` / `or` / `xor` / `surrogate` over sub-predicates.
    Compound {
        /// Logical operator.
        operator: CompoundOperator,
        /// Boxed sub-predicates, pooled in a `SmallVec` for typical arities 1–4.
        predicates: SmallVec<[Box<PredicateIr>; 4]>,
    },
}

/// Operator for a [`PredicateIr::Simple`] (`SimplePredicate/@operator`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimpleOperator {
    /// `equal`.
    Equal,
    /// `notEqual`.
    NotEqual,
    /// `lessThan`.
    LessThan,
    /// `lessOrEqual`.
    LessOrEqual,
    /// `greaterThan`.
    GreaterThan,
    /// `greaterOrEqual`.
    GreaterOrEqual,
    /// `isMissing`.
    IsMissing,
    /// `isNotMissing`.
    IsNotMissing,
}

/// Operator for a [`PredicateIr::Compound`] (`CompoundPredicate/@booleanOperator`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompoundOperator {
    /// `and`.
    And,
    /// `or`.
    Or,
    /// `xor`.
    Xor,
    /// `surrogate`.
    Surrogate,
}

/// Class distribution at a node (`ScoreDistribution/@value, @recordCount`).
#[derive(Debug, Clone)]
pub struct ScoreDistributionIr {
    /// Class value (interned).
    pub value: SymbolId,
    /// `recordCount` (weight, may be 0).
    pub record_count: f64,
}

/// Integer cast method for a [`TargetIr`] (`Target/@castInteger` may carry `Round | Ceiling | Floor`).
///
/// Mirrors JPMML `CastInteger`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CastIntegerMethod {
    /// Round to nearest (`round`).
    Round,
    /// Round toward `+∞` (`ceiling`).
    Ceiling,
    /// Round toward `-∞` (`floor`).
    Floor,
}

/// Single `TargetValue` entry inside a [`TargetIr`].
#[derive(Debug, Clone)]
pub struct TargetValueIr {
    /// `TargetValue/@value` (interned) when present.
    pub value: Option<SymbolId>,
    /// Raw string of `TargetValue/@value` (preserved for round-trip).
    pub value_str: Option<String>,
    /// `TargetValue/@displayValue`.
    pub display_value: Option<String>,
    /// `TargetValue/@priorProbability`.
    pub prior_probability: Option<f64>,
    /// `TargetValue/@defaultValue` (used for `MiningModel/modelChain`).
    pub default_value: Option<f64>,
}

/// Lowered `Target` (`Targets/Target`).
///
/// Controls post-processing of the raw model score: rescaling (`rescaleFactor`,
/// `rescaleConstant`), optional clamping (`min`, `max`), and optional integer
/// casting. Evaluator applies these in JPMML order.
#[derive(Debug, Clone)]
pub struct TargetIr {
    /// `Target/@field` resolved to `FieldId` when declared (synthetic if needed).
    pub field: Option<FieldId>,
    /// Canonical target field name (raw `Target/@field` or `"target"` fallback).
    pub field_name: String,
    /// `Target/@opType` when explicitly declared.
    pub op_type: Option<OpType>,
    /// `Target/@rescaleConstant` (additive, default 0.0).
    pub rescale_constant: f64,
    /// `Target/@rescaleFactor` (multiplicative, default 1.0).
    pub rescale_factor: f64,
    /// `true` when any `castInteger` was declared (backward compat flag).
    pub cast_integer: bool,
    /// Integer cast method, if declared.
    pub cast_method: Option<CastIntegerMethod>,
    /// `Target/@min` (clamp low).
    pub min: Option<f64>,
    /// `Target/@max` (clamp high).
    pub max: Option<f64>,
    /// `TargetValue` entries in document order.
    pub target_values: Vec<TargetValueIr>,
}

/// Requested output feature for association rules (`OutputField/@ruleFeature`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuleFeature {
    /// `antecedent`.
    Antecedent,
    /// `consequent`.
    Consequent,
    /// `rule` (antecedent + consequent).
    Rule,
    /// `ruleId`.
    RuleId,
    /// `confidence`.
    Confidence,
    /// `support`.
    Support,
    /// `lift`.
    Lift,
    /// `leverage`.
    Leverage,
    /// `affinity`.
    Affinity,
}

/// Algorithm for ranking / recommending association rules (`OutputField/@algorithm`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Algorithm {
    /// `recommendation`.
    Recommendation,
    /// `exclusiveRecommendation`.
    ExclusiveRecommendation,
    /// `ruleAssociation`.
    RuleAssociation,
}

/// Basis for ranking association rules (`OutputField/@rankBasis`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RankBasis {
    /// Rank by `confidence`.
    Confidence,
    /// Rank by `support`.
    Support,
    /// Rank by `lift`.
    Lift,
    /// Rank by `leverage`.
    Leverage,
    /// Rank by `affinity`.
    Affinity,
}

/// Order for ranking (`OutputField/@rankOrder`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RankOrder {
    /// Highest-rank first.
    Descending,
    /// Lowest-rank first.
    Ascending,
}

/// Lowered `OutputField` (`Output/OutputField`).
///
/// Describes a requested output: feature, rank / multi-valued handling, and
/// optional expression bytecode for `transformedValue` / `decision`.
///
/// # Examples
///
/// ```
/// use pmmlruntime::ir::{OutputFieldIr, RankBasis, RankOrder};
/// use pmmlruntime::base::ResultFeature;
/// let of = OutputFieldIr {
///     name: "probability".into(),
///     feature: ResultFeature::Probability,
///     value: None,
///     field: None,
///     target_field: None,
///     data_type: None,
///     op_type: None,
///     rule_feature: None,
///     algorithm: None,
///     rank: 1,
///     rank_basis: RankBasis::Confidence,
///     rank_order: RankOrder::Descending,
///     is_multi_valued: false,
///     segment_id: None,
///     is_final_result: true,
///     display_name: None,
///     expression_bytecode: None,
/// };
/// assert_eq!(of.feature, ResultFeature::Probability);
/// ```
#[derive(Debug, Clone)]
pub struct OutputFieldIr {
    /// `OutputField/@name`.
    pub name: String,
    /// `OutputField/@feature` (default `predictedValue`).
    pub feature: ResultFeature,
    /// `OutputField/@value` category (interned) when feature is probability etc. for a specific value.
    pub value: Option<SymbolId>,
    /// `OutputField/@field` or resolved alias for probability / affinity fields.
    pub field: Option<FieldId>,
    /// `OutputField/@targetField` (explicit target).
    pub target_field: Option<FieldId>,
    /// `OutputField/@dataType` when declared (otherwise inferred from feature).
    pub data_type: Option<DataType>,
    /// `OutputField/@opType` when declared.
    pub op_type: Option<OpType>,
    /// `OutputField/@ruleFeature` for association outputs.
    pub rule_feature: Option<RuleFeature>,
    /// `OutputField/@algorithm` for rule ranking.
    pub algorithm: Option<Algorithm>,
    /// `OutputField/@rank` (default 1).
    pub rank: i32,
    /// `OutputField/@rankBasis` (default `confidence`).
    pub rank_basis: RankBasis,
    /// `OutputField/@rankOrder` (default `descending`).
    pub rank_order: RankOrder,
    /// `OutputField/@isMultiValued` (`1` / `true` → `true`).
    pub is_multi_valued: bool,
    /// `OutputField/@segmentId` restricting output to a MiningModel segment.
    pub segment_id: Option<String>,
    /// `OutputField/@isFinalResult` (default `true`; `false` exposes intermediate Segmentation outputs).
    pub is_final_result: bool,
    /// `OutputField/@displayName`.
    pub display_name: Option<String>,
    /// Expression bytecode for `transformedValue` / `decision` (future use, `None` today).
    pub expression_bytecode: Option<Vec<Op>>,
}

/// Vendor extension (`Extension/@extender, @name, @value`).
///
/// Extensions are stored by lowering but never evaluated (D1 graceful handling).
/// See [`Ir::extensions`] and [`crate::ir::verify::verify_raw`].
#[derive(Debug, Clone)]
pub struct ExtensionIr {
    /// `Extension/@extender` (vendor, e.g., `"KNIME"`).
    pub extender: Option<String>,
    /// `Extension/@name` (vendor key).
    pub name: Option<String>,
    /// `Extension/@value` (payload string).
    pub value: Option<String>,
}

/// Top-level optimized representation consumed by the hot path.
///
/// Produced once by [`crate::ir::lower::lower()`] and held as `Arc<Ir>` in
/// `pmml-session`. All [`FieldId`]s and [`SymbolId`]s in `data_dictionary`,
/// `derived_fields`, and `model` are keys in `field_names` / `symbol_names`.
///
/// # Invariants
///
/// - Every [`FieldId`] referenced in `mining_schema` / `DerivedFieldIr` is
///   present in `field_names`.
/// - `num_fields()` equals `data_dictionary.len() + derived_fields.len()` (see
///   [`Ir::num_fields`]).
///
/// # Examples
///
/// ```
/// use pmmlruntime::xml::unmarshal;
/// use pmmlruntime::ir::lower;
/// let xml = br#"<PMML version="4.4"><Header/><DataDictionary><DataField name="x" dataType="double" optype="continuous"/></DataDictionary><TreeModel functionName="classification"><MiningSchema><MiningField name="x"/></MiningSchema><Node score="a"><True/></Node></TreeModel></PMML>"#;
/// let raw = unmarshal(xml).unwrap();
/// let ir = lower(raw).unwrap();
/// assert_eq!(ir.data_dictionary.len(), 1);
/// assert_eq!(ir.num_fields(), 1);
/// ```
#[derive(Debug, Clone)]
pub struct Ir {
    /// DataDictionary fields in document order (always includes every `DataField`).
    pub data_dictionary: Vec<FieldMeta>,
    /// Topologically-sorted derived fields from `TransformationDictionary` + model-local transforms.
    pub derived_fields: Vec<DerivedFieldIr>,
    /// The single scoring model for this PMML file.
    pub model: ModelIr,
    /// Snapshot of `field_name → FieldId` at lowering time for pretty-printing and session layout.
    pub field_names: std::collections::HashMap<FieldId, String>,
    /// Snapshot of `symbol_string → SymbolId` at lowering time (discrete values, category names, scores).
    pub symbol_names: std::collections::HashMap<SymbolId, String>,
    /// Vendor extensions stored but not evaluated (plan D1 graceful handling).
    pub extensions: Vec<ExtensionIr>,
    /// Audit: PMML 4.4 `pmml.xsd` element coverage (304 of 304 addressed, see `docs/PLAN.md` §1.5).
    pub element_coverage: usize,
}

impl Ir {
    /// Total field count: `data_dictionary.len() + derived_fields.len()`.
    ///
    /// Determines the length of the hot-path `&mut [crate::base::Value]` array
    /// allocated in `pmml-session` (actually `max_field_id` vs `num_fields()+4`,
    /// but never less than 16). See crate-level docs.
    ///
    /// # Examples
    ///
    /// ```
    /// use pmmlruntime::xml::unmarshal;
    /// use pmmlruntime::ir::lower;
    /// let xml = br#"<PMML version="4.4"><Header/><DataDictionary><DataField name="a" dataType="double" optype="continuous"/></DataDictionary><RegressionModel functionName="regression"><MiningSchema><MiningField name="a"/></MiningSchema><RegressionTable intercept="0"/></RegressionModel></PMML>"#;
    /// let ir = lower(unmarshal(xml).unwrap()).unwrap();
    /// assert_eq!(ir.num_fields(), 1);
    /// ```
    #[must_use]
    pub fn num_fields(&self) -> usize {
        self.data_dictionary.len() + self.derived_fields.len()
    }
}
