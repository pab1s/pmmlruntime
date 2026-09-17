//! TimeSeriesModel evaluation — stub for an unsupported model.
//!
//! PMML `TimeSeriesModel` (ARIMA/ExponentialSmoothing) is valid per `pmml.xsd`
//! but not yet scored by this runtime. The IR (`TimeSeriesIr`) is preserved for
//! inspection, and scoring delegates here which returns [`Value::Missing`] to
//! signal unsupported markup without erroring during `Session::from_bytes`.
//! Future work will implement step-ahead forecasting; see `docs/ARCHITECTURE.md`.
//!
//! # What belongs here
//!
//! - [`evaluate_time_series`] — stub entry point `(&TimeSeriesIr, &[Value]) -> Value`.

use crate::base::Value;
use crate::ir::TimeSeriesIr;

/// Evaluates a [`TimeSeriesIr`] against a dense `values` array.
///
/// Takes the `model` and `values` indexed by [`FieldId`](crate::base::FieldId).
/// Currently a stub that always returns [`Value::Missing`] because
/// `TimeSeriesModel` forecasting is not yet implemented. The lowerer preserves
/// `TimeSeriesIr` so `verify_ir` can check the unsupported path.
///
/// Returns [`Value::Missing`] in all cases.
pub fn evaluate_time_series(_model: &TimeSeriesIr, _values: &[Value]) -> Value {
    Value::Missing
}
