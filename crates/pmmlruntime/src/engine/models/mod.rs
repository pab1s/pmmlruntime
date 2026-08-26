//! Model evaluators — 19 pure `(&IrStruct, &[Value]) -> Value` scorers.
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
//! | [`anomaly_detection`] | `AnomalyDetectionModel` | [`evaluate_anomaly_detection`](crate::engine::models::evaluate_anomaly_detection) | `Continuous` anomaly score |
//! | [`baseline`] | `BaselineModel` | [`evaluate_baseline`](crate::engine::models::evaluate_baseline) | `Continuous` test statistic |
//! | [`gaussian_process`] | `GaussianProcessModel` | [`evaluate_gaussian_process`](crate::engine::models::evaluate_gaussian_process) | kernel weighted |
//! | [`text`] | `TextModel` | [`evaluate_text`](crate::engine::models::evaluate_text) | `Discrete` document id |
//! | [`time_series`] | `TimeSeriesModel` | [`evaluate_time_series`](crate::engine::models::evaluate_time_series) | `Continuous` forecast |
//! | [`sequence`] | `SequenceModel` | [`evaluate_sequence`](crate::engine::models::evaluate_sequence) | `Discrete` sequence |
//! | [`bayesian_network`] | `BayesianNetworkModel` | [`evaluate_bayesian_network`](crate::engine::models::evaluate_bayesian_network) | `Discrete` posterior |
//!
//! Re-exported at crate root for ergonomic `crate::engine::models::*`.

pub mod anomaly_detection;
pub mod association;
pub mod baseline;
pub mod bayesian_network;
pub mod clustering;
pub mod gaussian_process;
pub mod general_regression;
pub mod mining;
pub mod naive_bayes;
pub mod nearest_neighbor;
pub mod neural_network;
pub mod regression;
pub mod rule_set;
pub mod scorecard;
pub mod sequence;
pub mod support_vector_machine;
pub mod text;
pub mod time_series;
pub mod tree;

pub use anomaly_detection::evaluate_anomaly_detection;
pub use association::evaluate_association;
pub use baseline::evaluate_baseline;
pub use bayesian_network::evaluate_bayesian_network;
pub use clustering::evaluate_clustering;
pub use gaussian_process::evaluate_gaussian_process;
pub use general_regression::{evaluate_general_regression, evaluate_general_regression_with_probs};
pub use mining::evaluate_mining;
pub use naive_bayes::evaluate_naive_bayes;
pub use nearest_neighbor::evaluate_nearest_neighbor;
pub use neural_network::evaluate_neural_network;
pub use regression::evaluate_regression;
pub use rule_set::evaluate_rule_set;
pub use scorecard::evaluate_scorecard;
pub use sequence::evaluate_sequence;
pub use support_vector_machine::evaluate_support_vector_machine;
pub use text::{best_document_index, evaluate_text};
pub use time_series::evaluate_time_series;
pub use tree::evaluate_tree;
