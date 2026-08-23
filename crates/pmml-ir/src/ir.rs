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

/// Model IR — single variant for v1 (Tree).
#[derive(Debug, Clone)]
pub enum ModelIr {
    Tree(TreeIr),
    // stubs for future
    Regression,
    Mining,
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
