//! Model evaluators — 12 pure `(&IrStruct, &[Value]) -> Value` scorers.
//!
//! Each submodule implements one PMML model family. All evaluators are pure functions
//! over `&[Value]` (no `Session` state), `Send + Sync`, and benchable at ~402 ns
//! single-row for trees. `MiningModel` delegates to the other evaluators per segment.
//!
//! | Module | PMML element | Primary function | Return |
//! |---|---|---|
//! | [`tree`] | `TreeModel` | [`evaluate_tree`](crate::engine::models::evaluate_tree) | `Discrete`/`Continuous` or `Missing` |
//! | [`regression`] | `RegressionModel` | [`evaluate_regression`](crate::engine::models::evaluate_regression) | `Continuous` (or `Discrete` for multi-table) |
//! | [`mining`] | `MiningModel` | [`evaluate_mining`](crate::engine::models::evaluate_mining) | combined segment prediction |
//! | [`general_regression`] | `GeneralRegressionModel` | [`evaluate_general_regression`](crate::engine::models::general_regression::evaluate_general_regression) | `Discrete` via softmax |
//! | [`scorecard`] | `Scorecard` | [`evaluate_scorecard`](crate::engine::models::evaluate_scorecard) | `Continuous` sum |
//! | [`clustering`] | `ClusteringModel` | [`evaluate_clustering`](crate::engine::models::evaluate_clustering) | `Discrete` cluster id |
//! | [`naive_bayes`] | `NaiveBayesModel` | [`evaluate_naive_bayes`](crate::engine::models::evaluate_naive_bayes) | `Discrete` with threshold |
//! | [`nearest_neighbor`] | `NearestNeighborModel` | [`evaluate_nearest_neighbor`](crate::engine::models::evaluate_nearest_neighbor) | voted `Discrete` |
//! | [`neural_network`] | `NeuralNetwork` | [`evaluate_neural_network`](crate::engine::models::evaluate_neural_network) | `Continuous` |
//! | [`support_vector_machine`] | `SupportVectorMachineModel` | [`evaluate_support_vector_machine`](crate::engine::models::evaluate_support_vector_machine) | `Continuous` RBF sum |
//! | [`association`] | `AssociationModel` | [`evaluate_association`](crate::engine::models::evaluate_association) | `Discrete` consequent |
//! | [`rule_set`] | `RuleSetModel` | [`evaluate_rule_set`](crate::engine::models::evaluate_rule_set) | first firing `Discrete` |
//!
//! Re-exported at crate root for ergonomic `crate::engine::models::*`.

pub mod association;
pub mod clustering;
pub mod general_regression;
pub mod mining;
pub mod naive_bayes;
pub mod nearest_neighbor;
pub mod neural_network;
pub mod regression;
pub mod rule_set;
pub mod scorecard;
pub mod support_vector_machine;
pub mod tree;

pub use association::evaluate_association;
pub use clustering::evaluate_clustering;
pub use general_regression::{evaluate_general_regression, evaluate_general_regression_with_probs};
pub use mining::evaluate_mining;
pub use naive_bayes::evaluate_naive_bayes;
pub use nearest_neighbor::evaluate_nearest_neighbor;
pub use neural_network::evaluate_neural_network;
pub use regression::evaluate_regression;
pub use rule_set::evaluate_rule_set;
pub use scorecard::evaluate_scorecard;
pub use support_vector_machine::evaluate_support_vector_machine;
pub use tree::evaluate_tree;
