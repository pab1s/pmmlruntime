pub mod clustering;
pub mod mining;
pub mod naive_bayes;
pub mod nearest_neighbor;
pub mod regression;
pub mod scorecard;
pub mod tree;

pub use clustering::evaluate_clustering;
pub use mining::evaluate_mining;
pub use naive_bayes::evaluate_naive_bayes;
pub use nearest_neighbor::evaluate_nearest_neighbor;
pub use regression::evaluate_regression;
pub use scorecard::evaluate_scorecard;
pub use tree::evaluate_tree;
