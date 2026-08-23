pub mod mining;
pub mod regression;
pub mod tree;

pub use mining::evaluate_mining;
pub use regression::evaluate_regression;
pub use tree::evaluate_tree;
