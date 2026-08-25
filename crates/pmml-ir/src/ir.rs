//! IR — optimized, post-lower representation (hot path reads this).

use pmml_core::field::{DataType, OpType, ResultFeature};
use pmml_core::{FieldId, SymbolId};
use smallvec::SmallVec;

/// Field metadata after DataDictionary + MiningSchema lowering.
#[derive(Debug, Clone)]
pub struct FieldMeta {
    pub field_id: FieldId,
    pub name: String,
    pub data_type: DataType,
    pub op_type: OpType,
    pub values: Vec<SymbolId>, // allowed discrete values (for validation)
    // MiningSchema per-field treatments (JPMML: invalid/outlier/missing per MiningField)
    pub invalid_value_treatment: InvalidValueTreatment,
    pub invalid_value_replacement: Option<String>,
    pub missing_value_replacement: Option<String>,
    pub missing_value_treatment: MissingValueTreatment,
    pub outlier_treatment: OutlierTreatment,
    pub low_value: Option<f64>,
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

/// MiningSchema treatment enums — per PMML XSD (OUTLIER-TREATMENT-METHOD, INVALID-VALUE-TREATMENT-METHOD, MISSING-VALUE-TREATMENT-METHOD)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutlierTreatment {
    AsIs,
    AsMissingValues,
    AsExtremeValues,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InvalidValueTreatment {
    ReturnInvalid,
    AsIs,
    AsMissing,
    AsValue,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MissingValueTreatment {
    AsIs,
    AsMean,
    AsMode,
    AsMedian,
    AsValue,
    ReturnInvalid,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MiningFieldUsageType {
    Active,
    Predicted,
    Target,
    Supplementary,
    Group,
    Order,
    FrequencyWeight,
    AnalysisWeight,
}

/// MiningSchema IR — flat.
#[derive(Debug, Clone)]
pub struct MiningSchemaIr {
    pub active_fields: Vec<FieldId>, // fields with usageType != target
    pub target_field: Option<FieldId>,
    pub field_metas: Vec<FieldMeta>, // one per active+target (with per-field treatments)
    pub missing_value_replacement: Option<String>, // per field, simplified global (kept for backward compat)
}

/// DerivedField IR — bytecode for expression.
#[derive(Debug, Clone)]
pub struct DerivedFieldIr {
    pub field_id: FieldId,
    pub name: String,
    pub data_type: DataType,
    pub op_type: OpType,
    pub bytecode: Vec<Op>,
}

/// Bytecode for Apply/MapValues etc — evaluated by vm::eval.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LagAggregate {
    None,
    Avg,
    Min,
    Max,
    Sum,
    Product,
    Median,
    Stddev,
}

#[derive(Debug, Clone)]
pub enum Op {
    PushField(FieldId),
    PushConst(SymbolIdOrContinuous),
    CallBuiltin(BuiltinId, u8), // builtin + arity
    JumpIfMissing {
        target: usize,
    },
    MapValues {
        table: Vec<(SymbolId, SymbolId)>,
        default: Option<SymbolId>,
    },
    MapValuesMulti {
        inputs: Vec<FieldId>,
        table: Vec<(Vec<SymbolId>, SymbolId)>,
        default: Option<SymbolId>,
    },
    Discretize {
        bins: Vec<DiscretizeBin>,
        default_value: Option<SymbolId>,
        map_missing_to: Option<SymbolId>,
    },
    NormContinuous {
        field: FieldId,
        linear_norms: Vec<LinearNorm>,
    },
    NormDiscrete {
        field: FieldId,
        value: SymbolId,
        map_missing_to: Option<f64>,
    },
    Lag {
        field: FieldId,
        n: usize,
        aggregate: LagAggregate,
    },
    // Aggregate, TextIndex are modeled as CallBuiltin with dedicated BuiltinIds
    CallDefine {
        name: String,
        arity: u8,
    },
}

#[derive(Debug, Clone, Copy)]
pub enum SymbolIdOrContinuous {
    Symbol(SymbolId),
    Continuous(f64),
    Missing,
}

#[derive(Debug, Clone)]
pub struct DiscretizeBin {
    pub bin_value: SymbolId,
    pub interval_low: f64,
    pub interval_high: f64,
    pub left_closed: bool,
    pub right_closed: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct LinearNorm {
    pub orig: f64,
    pub norm: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BuiltinId {
    // Arithmetic
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Log,
    Log10,
    Ln,
    Exp,
    Sqrt,
    Abs,
    Floor,
    Ceil,
    Round,
    Remainder,
    // Math
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Sinh,
    Cosh,
    Tanh,
    // Min/Max + statistical aggregates
    Min,
    Max,
    Median,
    ProductOp,
    SumOp,
    AvgOp,
    Mean,
    StdDev,
    Variance,
    // Modulo / rounding
    Modulo,
    Rint,
    Expm1,
    Hypot,
    Ln1p,
    Atan2,
    Cbrt,
    Sign,
    // String
    Uppercase,
    Lowercase,
    Substring,
    TrimBlanks,
    NormalizeSpace,
    Concat,
    StringLength,
    Replace,
    Matches,
    FormatNumber,
    FormatDatetime,
    // TextIndex (distinct from string)
    TextIndex,
    // Aggregate (count/sum/avg/min/max over inline table or batch)
    AggregateCount,
    AggregateSum,
    AggregateAvg,
    AggregateMin,
    AggregateMax,
    AggregateMultiset,
    // Temporal / Sequence
    Lag,
    // Date / time builtins (chrono)
    DateDaysSinceYear,
    DateSecondsSinceYear,
    DateSecondsSinceMidnight,
    DateDaysSince1960,
    DateDaysSince1970,
    DateDaysSince1980,
    DateTimeSecondsSince1960,
    DateTimeSecondsSince1970,
    DateTimeSecondsSince1980,
    DateTimeSecondsSince0,
    TimeSeconds,
    // Distribution (statrs / libm)
    NormalCdf,
    NormalPdf,
    NormalIdf,
    StdNormalCdf,
    StdNormalPdf,
    StdNormalIdf,
    ErfOp,
    // Norm
    NormContinuousOp,
    NormDiscreteOp,
    // Comparison / Logical (via Apply)
    Equal,
    NotEqual,
    LessThan,
    LessOrEqual,
    GreaterThan,
    GreaterOrEqual,
    And,
    Or,
    Not,
    IsMissing,
    IsNotMissing,
    IsValid,
    IsNotValid,
    IsIn,
    IsNotIn,
    // Conditional
    If,
    // Misc
    Threshold,
    // Unknown fallback
    Unknown,
}

/// Model IR — all supported PMML 4.4 models.
#[derive(Debug, Clone)]
pub enum ModelIr {
    Tree(TreeIr),
    Regression(RegressionIr),
    Mining(MiningIr),
    Scorecard(ScorecardIr),
    Clustering(ClusteringIr),
    NaiveBayes(NaiveBayesIr),
    NearestNeighbor(NearestNeighborIr),
    SupportVectorMachine(SupportVectorMachineIr),
    NeuralNetwork(NeuralNetworkIr),
    GeneralRegression(GeneralRegressionIr),
    Association(AssociationIr),
    RuleSet(RuleSetIr),
}

#[derive(Debug, Clone)]
pub struct RegressionIr {
    pub function_name: String,
    pub mining_schema: MiningSchemaIr,
    pub regression_tables: Vec<RegressionTableIr>,
    pub normalization_method: RegressionNormalizationMethod,
    pub targets: Vec<TargetIr>,
    pub output: Vec<OutputFieldIr>,
}

#[derive(Debug, Clone)]
pub struct RegressionTableIr {
    pub intercept: f64,
    pub target_category: Option<SymbolId>,
    pub numeric_predictors: Vec<NumericPredictorIr>,
    pub categorical_predictors: Vec<CategoricalPredictorIr>,
}

#[derive(Debug, Clone)]
pub struct NumericPredictorIr {
    pub field: FieldId,
    pub coefficient: f64,
    pub exponent: i32,
}

#[derive(Debug, Clone)]
pub struct CategoricalPredictorIr {
    pub field: FieldId,
    pub value: SymbolId,
    pub coefficient: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegressionNormalizationMethod {
    None,
    SimpleMax,
    Softmax,
    Logit,
    Probit,
    ClogLog,
    Exp,
    Loglog,
    Cauchit,
}

#[derive(Debug, Clone)]
pub struct MiningIr {
    pub function_name: String,
    pub mining_schema: MiningSchemaIr,
    pub segmentation: SegmentationIr,
    pub targets: Vec<TargetIr>,
    pub output: Vec<OutputFieldIr>,
}

#[derive(Debug, Clone)]
pub struct SegmentationIr {
    pub multiple_model_method: MultipleModelMethod,
    pub missing_prediction_treatment: MissingPredictionTreatment,
    pub segments: Vec<SegmentIr>,
}

#[derive(Debug, Clone)]
pub struct SegmentIr {
    pub id: Option<String>,
    pub predicate: PredicateIr,
    pub weight: f64,
    pub model: Box<ModelIr>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MultipleModelMethod {
    MajorityVote,
    WeightedMajorityVote,
    Average,
    WeightedAverage,
    Median,
    WeightedMedian,
    Max,
    Sum,
    WeightedSum,
    SelectFirst,
    SelectAll,
    ModelChain,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MissingPredictionTreatment {
    ReturnMissing,
    SkipSegment,
    Continue,
}

#[derive(Debug, Clone)]
pub struct ScorecardIr {
    pub function_name: String,
    pub initial_score: f64,
    pub use_reason_codes: bool,
    pub reason_code_algorithm: String,
    pub mining_schema: MiningSchemaIr,
    pub characteristics: Vec<CharacteristicIr>,
    pub output: Vec<OutputFieldIr>,
}

#[derive(Debug, Clone)]
pub struct CharacteristicIr {
    pub name: String,
    pub reason_code: Option<String>,
    pub baseline_score: f64,
    pub attributes: Vec<AttributeIr>,
}

#[derive(Debug, Clone)]
pub struct AttributeIr {
    pub partial_score: f64,
    pub predicate: PredicateIr,
    pub reason_code: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ClusteringIr {
    pub function_name: String,
    pub model_class: String,
    pub number_of_clusters: usize,
    pub mining_schema: MiningSchemaIr,
    pub comparison_measure: String,
    pub clustering_fields: Vec<FieldId>,
    pub clusters: Vec<ClusterIr>,
    pub output: Vec<OutputFieldIr>,
}

#[derive(Debug, Clone)]
pub struct ClusterIr {
    pub name: pmml_core::SymbolId,
    pub name_str: String,
    pub array: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct TargetValueCountIr {
    pub value: SymbolId,
    pub count: f64,
}

#[derive(Debug, Clone)]
pub struct PairCountsIr {
    pub value: SymbolId,
    pub target_counts: Vec<TargetValueCountIr>,
}

#[derive(Debug, Clone)]
pub struct TargetValueStatIr {
    pub value: SymbolId,
    pub mean: Option<f64>,
    pub variance: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct BayesInputIr {
    pub field: FieldId,
    pub target_value_stats: Vec<TargetValueStatIr>,
    pub pair_counts: Vec<PairCountsIr>,
}

#[derive(Debug, Clone)]
pub struct NaiveBayesIr {
    pub function_name: String,
    pub threshold: f64,
    pub mining_schema: MiningSchemaIr,
    pub output: Vec<OutputFieldIr>,
    pub bayes_inputs: Vec<BayesInputIr>,
    pub bayes_output_counts: Vec<TargetValueCountIr>,
}

#[derive(Debug, Clone)]
pub struct NearestNeighborIr {
    pub function_name: String,
    pub number_of_neighbors: usize,
    pub mining_schema: MiningSchemaIr,
    pub output: Vec<OutputFieldIr>,
    pub knn_inputs: Vec<FieldId>,
    pub instances: Vec<std::collections::HashMap<pmml_core::FieldId, pmml_core::Value>>,
    pub instance_ids: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct SupportVectorMachineIr {
    pub function_name: String,
    pub mining_schema: MiningSchemaIr,
    pub output: Vec<OutputFieldIr>,
    pub vector_fields: Vec<FieldId>,
    pub vector_instances: Vec<(String, Vec<f64>)>,
    pub support_vectors: Vec<String>,
    pub coefficients: Vec<f64>,
    pub absolute_value: f64,
    pub kernel_gamma: f64,
}

#[derive(Debug, Clone)]
pub struct NeuralInputIr {
    pub id: String,
    pub field: FieldId,
}

#[derive(Debug, Clone)]
pub struct NeuronIr {
    pub id: String,
    pub bias: f64,
    pub cons: Vec<(String, f64)>, // from -> weight
}

#[derive(Debug, Clone)]
pub struct NeuralLayerIr {
    pub number_of_neurons: usize,
    pub activation_function: String,
    pub neurons: Vec<NeuronIr>,
}

#[derive(Debug, Clone)]
pub struct NeuralNetworkIr {
    pub function_name: String,
    pub mining_schema: MiningSchemaIr,
    pub output: Vec<OutputFieldIr>,
    pub neural_inputs: Vec<NeuralInputIr>,
    pub neural_layers: Vec<NeuralLayerIr>,
    pub activation_function: String,
}

#[derive(Debug, Clone)]
pub struct ParameterIr {
    pub name: String,
    pub label: Option<String>,
}

#[derive(Debug, Clone)]
pub struct FactorIr {
    pub name: FieldId,
    pub categories: Vec<SymbolId>,
    pub matrix: Vec<Vec<f64>>,
}

#[derive(Debug, Clone)]
pub struct PPCellIr {
    pub value: SymbolId,
    pub predictor_name: String,
    pub parameter_name: String,
}

#[derive(Debug, Clone)]
pub struct PCellIr {
    pub target_category: Option<SymbolId>,
    pub parameter_name: String,
    pub beta: f64,
}

#[derive(Debug, Clone)]
pub struct GeneralRegressionIr {
    pub function_name: String,
    pub mining_schema: MiningSchemaIr,
    pub output: Vec<OutputFieldIr>,
    pub model_type: Option<String>,
    pub target_variable_name: Option<String>,
    pub target_reference_category: Option<SymbolId>,
    pub parameters: Vec<ParameterIr>,
    pub factors: Vec<FactorIr>,
    pub covariates: Vec<FieldId>,
    pub pp_matrix: Vec<PPCellIr>,
    pub param_matrix: Vec<PCellIr>,
}

#[derive(Debug, Clone)]
pub struct ItemIr {
    pub id: String,
    pub value: SymbolId,
}

#[derive(Debug, Clone)]
pub struct ItemsetIr {
    pub id: String,
    pub item_ids: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct AssociationRuleIr {
    pub antecedent: String,
    pub consequent: String,
    pub support: f64,
    pub confidence: f64,
    pub lift: f64,
}

#[derive(Debug, Clone)]
pub struct AssociationIr {
    pub function_name: String,
    pub mining_schema: MiningSchemaIr,
    pub output: Vec<OutputFieldIr>,
    pub items: Vec<ItemIr>,
    pub itemsets: Vec<ItemsetIr>,
    pub rules: Vec<AssociationRuleIr>,
}

#[derive(Debug, Clone)]
pub struct SimpleRuleIr {
    pub id: Option<String>,
    pub score: SymbolId,
    pub predicate: PredicateIr,
}

#[derive(Debug, Clone)]
pub struct RuleSetIr {
    pub function_name: String,
    pub mining_schema: MiningSchemaIr,
    pub output: Vec<OutputFieldIr>,
    pub default_score: Option<SymbolId>,
    pub rules: Vec<SimpleRuleIr>,
}

#[derive(Debug, Clone)]
pub struct TreeIr {
    pub function_name: String, // classification/regression
    pub missing_value_strategy: MissingValueStrategy,
    pub no_true_child_strategy: NoTrueChildStrategy,
    pub nodes: Vec<NodeIr>, // flat, root at 0
    pub mining_schema: MiningSchemaIr,
    pub targets: Vec<TargetIr>,
    pub output: Vec<OutputFieldIr>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MissingValueStrategy {
    LastPrediction,
    NullPrediction,
    DefaultChild,
    None,
    // Explicitly unsupported per JPMML (must throw UnsupportedMarkup, not fallback)
    WeightedConfidence,
    AggregateNodes,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoTrueChildStrategy {
    ReturnNullPrediction,
    ReturnLastPrediction,
}

#[derive(Debug, Clone)]
pub struct NodeIr {
    pub id: Option<String>,
    pub score: Option<SymbolIdOrContinuous>,
    pub predicate: PredicateIr,
    pub children: Vec<usize>, // indices into TreeIr::nodes
    /// Index of default child for missingValueStrategy=DefaultChild (JPMML parity)
    pub default_child: Option<usize>,
    pub score_distributions: Vec<ScoreDistributionIr>,
}

#[derive(Debug, Clone)]
pub enum PredicateIr {
    True,
    Simple {
        field: FieldId,
        operator: SimpleOperator,
        value: SymbolIdOrContinuous,
    },
    SimpleSet {
        field: FieldId,
        is_in: bool,
        array: Vec<SymbolIdOrContinuous>,
    },
    Compound {
        operator: CompoundOperator,
        // SmallVec pooled for typical 1-4 predicates per compound (E2) — Boxed to avoid infinite size
        predicates: SmallVec<[Box<PredicateIr>; 4]>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimpleOperator {
    Equal,
    NotEqual,
    LessThan,
    LessOrEqual,
    GreaterThan,
    GreaterOrEqual,
    IsMissing,
    IsNotMissing,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompoundOperator {
    And,
    Or,
    Xor,
    Surrogate,
}

#[derive(Debug, Clone)]
pub struct ScoreDistributionIr {
    pub value: SymbolId,
    pub record_count: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CastIntegerMethod {
    Round,
    Ceiling,
    Floor,
}

#[derive(Debug, Clone)]
pub struct TargetValueIr {
    pub value: Option<SymbolId>,
    pub value_str: Option<String>,
    pub display_value: Option<String>,
    pub prior_probability: Option<f64>,
    pub default_value: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct TargetIr {
    pub field: Option<FieldId>,
    pub field_name: String,
    pub op_type: Option<OpType>,
    pub rescale_constant: f64,
    pub rescale_factor: f64,
    pub cast_integer: bool, // kept for backward compat (true if any cast)
    pub cast_method: Option<CastIntegerMethod>,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub target_values: Vec<TargetValueIr>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuleFeature {
    Antecedent,
    Consequent,
    Rule,
    RuleId,
    Confidence,
    Support,
    Lift,
    Leverage,
    Affinity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Algorithm {
    Recommendation,
    ExclusiveRecommendation,
    RuleAssociation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RankBasis {
    Confidence,
    Support,
    Lift,
    Leverage,
    Affinity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RankOrder {
    Descending,
    Ascending,
}

#[derive(Debug, Clone)]
pub struct OutputFieldIr {
    pub name: String,
    pub feature: ResultFeature,
    pub value: Option<SymbolId>,
    pub field: Option<FieldId>, // for probability etc (field containing category)
    pub target_field: Option<FieldId>,
    pub data_type: Option<DataType>,
    pub op_type: Option<OpType>,
    pub rule_feature: Option<RuleFeature>,
    pub algorithm: Option<Algorithm>,
    pub rank: i32,
    pub rank_basis: RankBasis,
    pub rank_order: RankOrder,
    pub is_multi_valued: bool,
    pub segment_id: Option<String>,
    pub is_final_result: bool,
    pub display_name: Option<String>,
    // For transformedValue/decision: optional expression bytecode (evaluated via vm)
    // Minimal: store raw expression not yet compiled; evaluator may use it if present.
    // For now keep None — future: Vec<Op>
    pub expression_bytecode: Option<Vec<Op>>,
}

#[derive(Debug, Clone)]
pub struct ExtensionIr {
    pub extender: Option<String>,
    pub name: Option<String>,
    pub value: Option<String>,
}

#[derive(Debug, Clone)]
pub struct Ir {
    pub data_dictionary: Vec<FieldMeta>,
    pub derived_fields: Vec<DerivedFieldIr>,
    pub model: ModelIr,
    // interner snapshot for symbol resolve (clone)
    pub field_names: std::collections::HashMap<FieldId, String>,
    pub symbol_names: std::collections::HashMap<SymbolId, String>,
    /// Vendor extensions — stored but not evaluated (plan D1 graceful handling)
    pub extensions: Vec<ExtensionIr>,
    /// Audit: PMML 4.4 element count covered (304) — see docs/PLAN.md
    pub element_coverage: usize,
}

impl Ir {
    pub fn num_fields(&self) -> usize {
        self.data_dictionary.len() + self.derived_fields.len()
    }
}
