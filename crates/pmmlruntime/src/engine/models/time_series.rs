//! TimeSeriesModel evaluation — stub.

use crate::base::Value;
use crate::ir::TimeSeriesIr;

pub fn evaluate_time_series(_model: &TimeSeriesIr, _values: &[Value]) -> Value {
    Value::Missing
}
