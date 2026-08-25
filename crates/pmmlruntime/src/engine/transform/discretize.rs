//! Discretize binning — continuous → discrete via interval matching.
//!
//! Implements `Discretize` semantics from PMML 4.4: a continuous input is tested
//! against an ordered list of intervals (`low`/`high` with open/closed closure).
//! The first matching interval yields a discrete symbol. This module provides
//! a standalone helper exposed for unit tests; the hot-path `Op::Discretize`
//! evaluation in `transform::vm` uses [`crate::ir::DiscretizeBin`] directly.
//!
//! # What belongs here
//!
//! - [`eval_discretize`] — pure function mapping `Value::Continuous(x)` to a bin.
//!
//! # Performance
//!
//! `O(bins)` linear scan. Bins are typically 2–7 intervals; binary search would
//! not improve the constant factor for this size.

use crate::base::Value;

/// Map a continuous [`Value`] into a bin defined by intervals.
///
/// Tests `value` against `bins` in order. Each bin is `(low, high, left_closed, right_closed)`:
/// the interval is inclusive (`>=`/`<=`) when closed, exclusive (`>`/`<`) otherwise.
/// The first matching bin returns `Continuous(idx as f64)` where `idx` is the bin's position;
/// the VM layer then maps `idx` to the actual [`crate::base::SymbolId`] (`bin_value`) or to
/// `default_value` / `map_missing_to`.
///
/// # Parameters
///
/// - `value`: Input value. Only [`Value::Continuous`] is binned; `Missing` or `Discrete` returns `Missing`.
/// - `bins`: Ordered intervals to test top-to-bottom. `low == -inf` / `high == inf` are permitted via `f64::NEG_INFINITY` / `INFINITY`.
///
/// # Returns
///
/// `Continuous(bin_index)` when a bin matches, otherwise [`Value::Missing`]. `Missing` input always yields `Missing`.
///
/// # Panics
///
/// Never panics.
///
/// # Performance
///
/// `O(bins)` linear scan. No allocation.
///
/// # Examples
///
/// ```
/// use pmmlruntime::base::Value;
/// use pmmlruntime::engine::transform::discretize::eval_discretize;
///
/// let bins = vec![(0.0, 10.0, true, false), (10.0, 20.0, true, true)];
/// assert_eq!(eval_discretize(Value::Continuous(5.0), &bins), Value::Continuous(0.0));
/// assert_eq!(eval_discretize(Value::Continuous(10.0), &bins), Value::Continuous(1.0));
/// assert_eq!(eval_discretize(Value::Continuous(25.0), &bins), Value::Missing);
/// assert_eq!(eval_discretize(Value::Missing, &bins), Value::Missing);
/// ```
pub fn eval_discretize(value: Value, bins: &[(f64, f64, bool, bool)]) -> Value {
    let x = match value {
        Value::Continuous(f) => f,
        _ => return Value::Missing,
    };
    for (idx, (low, high, left_closed, right_closed)) in bins.iter().enumerate() {
        let left_ok = if *left_closed { x >= *low } else { x > *low };
        let right_ok = if *right_closed { x <= *high } else { x < *high };
        if left_ok && right_ok {
            // Return bin index as Continuous for generic helper; vm maps to SymbolId
            return Value::Continuous(idx as f64);
        }
    }
    Value::Missing
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discretize_simple() {
        let bins = vec![(0.0, 10.0, true, false), (10.0, 20.0, true, true)];
        assert_eq!(
            eval_discretize(Value::Continuous(5.0), &bins),
            Value::Continuous(0.0)
        );
        assert_eq!(
            eval_discretize(Value::Continuous(10.0), &bins),
            Value::Continuous(1.0)
        );
        assert_eq!(
            eval_discretize(Value::Continuous(20.0), &bins),
            Value::Continuous(1.0)
        );
        assert_eq!(
            eval_discretize(Value::Continuous(25.0), &bins),
            Value::Missing
        );
        assert_eq!(eval_discretize(Value::Missing, &bins), Value::Missing);
    }
}
