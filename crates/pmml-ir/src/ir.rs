//! IR — optimized, post-lower representation (hot path reads this).

use pmml_core::field::{DataType, OpType, ResultFeature};
use pmml_core::{FieldId, SymbolId};

/// Field metadata after DataDictionary + MiningSchema lowering.
#[derive(Debug, Clone)]
pub struct FieldMeta {
    pub field_id: FieldId,
    pub name: String,
    pub data_type: DataType,
    pub op_type: OpType,
    pub values: Vec<SymbolId>, // allowed discrete values (for validation)
}

/// MiningSchema IR — flat.
#[derive(Debug, Clone)]
pub struct MiningSchemaIr {
    pub active_fields: Vec<FieldId>, // fields with usageType != target
    pub target_field: Option<FieldId>,
    pub field_metas: Vec<FieldMeta>, // one per active+target
    pub missing_value_replacement: Option<String>, // per field, simplified global?
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
#[derive(Debug, Clone)]
pub enum Op {
    PushField(FieldId),
    PushConst(SymbolIdOrContinuous),
    CallBuiltin(BuiltinId, u8), // builtin + arity
    JumpIfMissing { target: usize },
    MapValues { table: Vec<(SymbolId, SymbolId)> },
    // future: Discretize, etc
}

#[derive(Debug, Clone, Copy)]
pub enum SymbolIdOrContinuous {
    Symbol(SymbolId),
    Continuous(f64),
    Missing,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BuiltinId {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Log,
    Exp,
    Sqrt,
    Abs,
    Min,
    Max,
    // ... 100 total, stub for v1
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
    // stub
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
        predicates: Vec<PredicateIr>,
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

#[derive(Debug, Clone)]
pub struct TargetIr {
    pub field: FieldId,
    pub rescale_constant: f64,
    pub rescale_factor: f64,
    pub cast_integer: bool,
}

#[derive(Debug, Clone)]
pub struct OutputFieldIr {
    pub name: String,
    pub feature: ResultFeature,
    pub value: Option<SymbolId>,
    pub field: Option<FieldId>, // for probability etc
}

#[derive(Debug, Clone)]
pub struct Ir {
    pub data_dictionary: Vec<FieldMeta>,
    pub derived_fields: Vec<DerivedFieldIr>,
    pub model: ModelIr,
    // interner snapshot for symbol resolve (clone)
    pub field_names: std::collections::HashMap<FieldId, String>,
    pub symbol_names: std::collections::HashMap<SymbolId, String>,
}

impl Ir {
    pub fn num_fields(&self) -> usize {
        self.data_dictionary.len() + self.derived_fields.len()
    }
}
