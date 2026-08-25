//! Discretize — binning continuous to discrete (full per plan D2).

use pmml_core::Value;

/// Evaluate discretize: map continuous `value` into a bin defined by intervals.
/// `_bins` is slice of (low, high, left_closed, right_closed). Returns `Missing` if not found.
/// For full SymbolId mapping, caller (vm) handles bin_value mapping; this helper returns bin index as Continuous.
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
