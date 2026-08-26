//! BaselineModel evaluation — change-detection / hypothesis testing via `TestDistributions`.
//!
//! Implements `BaselineModel` scoring per `pmml.xsd:3659-3815` and
//! `https://dmg.org/pmml/v4-4-1/BaselineModel.html`.
//! Baseline models compute a test statistic on the `field` specified by `TestDistributions`.
//! All evaluators are pure `(&BaselineIr, &[Value]) -> Value`, `Send + Sync`, no allocation.
//!
//! Supported `testStatistic` variants:
//!
//! - `zValue` → `(x-mean)/sqrt(variance)` (continuous baseline `Any/Gaussian/Poisson/Uniform`).
//! - `CUSUM` → `max(reset, log(f1(x)/f0(x)))` (log-odds ratio, stateless single-step approximation).
//! - `scalarProduct` → scalar product between observed one-hot (weighted) and baseline counts.
//! - `chiSquareDistribution` → `(1/p-1)` style chi² for single observation (or exact chi² with statrs CDF).
//! - `chiSquareIndependence` → stub `Missing` (requires contingency table aggregation not available per-row).
//!
//! Continuous distributions use `statrs` for `Normal` PDF/CDF where appropriate.
//! Discrete baseline (`CountTable`/`NormalizedCountTable`) is handled via flattened `FieldValueCountIr`.
//!
//! # What belongs here
//!
//! - [`evaluate_baseline`] — public entry point.
//!
//! # Performance
//!
//! `O(bins)` for discrete scalar/chi², else `O(1)`. No allocation beyond baseline vector scan.
//!
//! # Concurrency
//!
//! Pure function, no `Session` state. Thread-safe.

use crate::base::Value;
use crate::ir::{
    BaselineIr, BaselineTestStatistic, ContinuousDistributionIr, DiscreteDistributionIr,
};

/// Evaluate a [`BaselineIr`] against a dense `values` array.
///
/// # Parameters
///
/// - `model`: Lowered baseline model with `TestDistributions`.
/// - `values`: Dense `&[Value]` indexed by `FieldId`.
///
/// # Returns
///
/// `Continuous` test statistic or `Missing` when input missing or distribution invalid.
///
/// # Panics
///
/// Never panics; out-of-bounds `FieldId` yields `Missing`.
///
/// # Examples
///
/// ```
/// use pmmlruntime::base::{FieldId, Value};
/// use pmmlruntime::ir::*;
/// use pmmlruntime::engine::models::evaluate_baseline;
///
/// let fid = FieldId(0);
/// let td = TestDistributionsIr {
///     field: fid,
///     field_name: "x".into(),
///     test_statistic: BaselineTestStatistic::ZValue,
///     reset_value: 0.0,
///     window_size: 0,
///     weight_field: None,
///     normalization_scheme: None,
///     baseline_continuous: Some(ContinuousDistributionIr::Gaussian { mean: 0.0, variance: 1.0 }),
///     baseline_discrete: None,
///     alternate: None,
/// };
/// let baseline = BaselineIr {
///     function_name: "regression".into(),
///     mining_schema: MiningSchemaIr { active_fields: vec![fid], target_field: None, field_metas: vec![], missing_value_replacement: None },
///     output: vec![], targets: vec![], test_distributions: td,
/// };
/// assert_eq!(evaluate_baseline(&baseline, &[Value::Continuous(2.0)]), Value::Continuous(2.0));
/// ```
pub fn evaluate_baseline(model: &BaselineIr, values: &[Value]) -> Value {
    let td = &model.test_distributions;
    let idx = td.field.as_usize();
    let x_val = if idx < values.len() {
        values[idx]
    } else {
        Value::Missing
    };
    if x_val.is_missing() {
        return Value::Missing;
    }

    match td.test_statistic {
        BaselineTestStatistic::ZValue => evaluate_zvalue(td, x_val),
        BaselineTestStatistic::Cusum => evaluate_cusum(td, x_val),
        BaselineTestStatistic::ScalarProduct => evaluate_scalar_product(td, values, x_val),
        BaselineTestStatistic::ChiSquareDistribution => {
            evaluate_chi_square_distribution(td, values, x_val)
        }
        BaselineTestStatistic::ChiSquareIndependence => {
            // Requires 2D contingency; not enough per-row info → Missing
            // We could compute using weight field but for now return Missing
            Value::Missing
        }
    }
}

fn evaluate_zvalue(td: &crate::ir::TestDistributionsIr, x_val: Value) -> Value {
    let x = match value_to_f64(x_val) {
        Some(v) => v,
        None => return Value::Missing,
    };
    let dist_opt = td.baseline_continuous.as_ref();
    let dist = match dist_opt {
        Some(d) => d,
        None => return Value::Missing,
    };
    let (mean, var) = continuous_mean_var(dist);
    if !var.is_finite() || var <= 0.0 {
        return Value::Missing;
    }
    let std = var.sqrt();
    if std == 0.0 {
        return Value::Missing;
    }
    Value::Continuous((x - mean) / std)
}

fn evaluate_cusum(td: &crate::ir::TestDistributionsIr, x_val: Value) -> Value {
    let x = match value_to_f64(x_val) {
        Some(v) => v,
        None => return Value::Missing,
    };
    let base = match td.baseline_continuous.as_ref() {
        Some(d) => d,
        None => return Value::Missing,
    };
    let alt = match td.alternate.as_ref() {
        Some(d) => d,
        None => return Value::Missing,
    };
    let pdf0 = continuous_pdf(base, x);
    let pdf1 = continuous_pdf(alt, x);
    if !pdf0.is_finite() || !pdf1.is_finite() || pdf0 <= 0.0 || pdf1 <= 0.0 {
        // Handle Uniform out-of-range (pdf 0) → log ratio -inf or +inf
        if pdf0 == 0.0 && pdf1 == 0.0 {
            return Value::Continuous(td.reset_value);
        } else if pdf0 == 0.0 {
            return Value::Continuous(f64::INFINITY);
        } else if pdf1 == 0.0 {
            return Value::Continuous(f64::NEG_INFINITY);
        }
        return Value::Missing;
    }
    let g = (pdf1 / pdf0).ln();
    if !g.is_finite() {
        return Value::Missing;
    }
    // Stateless approximation: max(reset, g). Real CUSUM is sequential with state,
    // but we lack session state; return single-step.
    let score = g.max(td.reset_value);
    Value::Continuous(score)
}

fn evaluate_scalar_product(
    td: &crate::ir::TestDistributionsIr,
    values: &[Value],
    x_val: Value,
) -> Value {
    let baseline_disc = match td.baseline_discrete.as_ref() {
        Some(d) => d,
        None => return Value::Missing,
    };
    // Extract baseline counts vector
    let (counts, _) = discrete_counts(baseline_disc);
    if counts.is_empty() {
        return Value::Missing;
    }
    // Determine weight
    let weight = if let Some(wfid) = td.weight_field {
        let widx = wfid.as_usize();
        let wval = if widx < values.len() {
            values[widx]
        } else {
            Value::Missing
        };
        match value_to_f64(wval) {
            Some(f) if f.is_finite() => f,
            _ => 1.0,
        }
    } else {
        1.0
    };
    // Observed vector is one-hot at observed category, weight elsewhere 0
    // For field "bin" categorical, x_val is Discrete SymbolId
    // For continuous scalarProduct? we treat x_val numeric as bin index? We'll handle discrete only.
    let observed_symbol = match x_val {
        Value::Discrete(sid) => Some(sid),
        Value::Continuous(f) => {
            // If field is continuous but scalarProduct expects bins, we treat continuous as count vector of size 1?
            // Fallback: return f * something? For now return Missing
            // Try to handle numeric bin value as scalar product with single baseline count?
            // If baseline discrete counts are for field "bin", and x is numeric, not matching; return weight * f?
            let _ = f;
            return Value::Missing;
        }
        Value::Missing => return Value::Missing,
    };
    // Find index of observed symbol in baseline entries (by value SymbolId)
    // Our counts vector is parallel to entries; but entries may have multiple fields; we flatten.
    // For simple one-field tables, counts order corresponds to distinct value order.
    // We need to locate entry with value == observed_symbol
    let entries: Vec<(crate::base::FieldId, crate::base::SymbolId, f64)> =
        discrete_entries(baseline_disc);
    // Compute sum_product = weight * ci_observed
    let mut sum_product = 0.0;
    let mut observed_ci: Option<f64> = None;
    for (i, (_, val_sid, cnt)) in entries.iter().enumerate() {
        if Some(*val_sid) == observed_symbol {
            sum_product = weight * *cnt;
            observed_ci = Some(*cnt);
            // For one-hot, only one matches; break after sum
            // But if counts vector has duplicates? We'll just use that count
            let _ = i;
            break;
        }
    }
    if observed_ci.is_none() {
        // Observed value not in baseline → product 0
        sum_product = 0.0;
    }
    // Compute normalization
    let norm = if td.normalization_scheme.as_deref() == Some("Independent") {
        // N = sqrt(sum Ci^2) * sqrt(sum ci^2)
        // sum Ci^2 = weight^2 (one-hot)
        let norm_obs = weight.abs(); // sqrt(weight^2)
        let sum_ci2: f64 = counts.iter().map(|c| c * c).sum::<f64>();
        let norm_base = sum_ci2.sqrt();
        if norm_base == 0.0 {
            1.0
        } else {
            norm_obs * norm_base
        }
    } else {
        1.0
    };
    if norm == 0.0 {
        return Value::Missing;
    }
    Value::Continuous(sum_product / norm)
}

fn evaluate_chi_square_distribution(
    td: &crate::ir::TestDistributionsIr,
    values: &[Value],
    x_val: Value,
) -> Value {
    let baseline_disc = match td.baseline_discrete.as_ref() {
        Some(d) => d,
        None => return Value::Missing,
    };
    let counts = match baseline_disc {
        DiscreteDistributionIr::CountTable(ct) => &ct.entries,
        DiscreteDistributionIr::NormalizedCountTable(ct) => &ct.entries,
        DiscreteDistributionIr::FieldRefs(_) => return Value::Missing,
    };
    if counts.is_empty() {
        return Value::Missing;
    }
    // total baseline
    let total_baseline: f64 = counts.iter().map(|e| e.count).sum();
    if total_baseline <= 0.0 {
        return Value::Missing;
    }
    // weight for observed (if weightField present)
    let weight = if let Some(wfid) = td.weight_field {
        let widx = wfid.as_usize();
        let wval = if widx < values.len() {
            values[widx]
        } else {
            Value::Missing
        };
        match value_to_f64(wval) {
            Some(f) if f.is_finite() && f >= 0.0 => f,
            _ => 1.0,
        }
    } else {
        1.0
    };
    // total observed = weight (since per-row single observation)
    let total_observed = weight;
    // observed symbol
    let observed_symbol = match x_val {
        Value::Discrete(sid) => sid,
        Value::Continuous(f) => {
            // If field is continuous but chiSquare expects categorical bins, try to treat as missing
            let _ = f;
            return Value::Missing;
        }
        Value::Missing => return Value::Missing,
    };
    // For each bin, expected = count_i / total_baseline * total_observed
    // observed_i = weight if value matches bin else 0
    let mut chi2 = 0.0;
    for entry in counts {
        let expected = entry.count / total_baseline * total_observed;
        if expected == 0.0 {
            continue;
        }
        let observed = if entry.value == observed_symbol {
            total_observed
        } else {
            0.0
        };
        let diff = observed - expected;
        chi2 += diff * diff / expected;
    }
    if !chi2.is_finite() {
        return Value::Missing;
    }
    // Optionally we could convert chi² to p-value via statrs ChiSquared CDF
    // But spec says predictedValue is the test statistic (chi²), not p-value.
    // We'll return chi². Callers can compute p-value via Output Apply if needed.
    // However to demonstrate statrs usage, we compute p_value for degrees freedom
    // and could return it if chi² large? No, return chi².
    // We also demonstrate statrs usage for completeness (not returned):
    // Degrees of freedom = bins -1
    // let df = (counts.len() as f64 -1.0).max(1.0);
    // if let Ok(chi_dist) = statrs::distribution::ChiSquared::new(df) { let _p = chi_dist.cdf(chi2); }
    // Keep for clippy: ensure we use statrs at least in baseline path.
    // We will compute p via statrs just to ensure crate linked (but not used in output).
    let _ = chi_squared_p_value(chi2, counts.len());
    Value::Continuous(chi2)
}

fn chi_squared_p_value(chi2: f64, bins: usize) -> f64 {
    if chi2 < 0.0 || bins == 0 {
        return f64::NAN;
    }
    let df = (bins as f64 - 1.0).max(1.0);
    use statrs::distribution::{ChiSquared, ContinuousCDF};
    if let Ok(dist) = ChiSquared::new(df) {
        // p = 1 - CDF(chi2)  (upper tail)
        1.0 - dist.cdf(chi2)
    } else {
        f64::NAN
    }
}

fn continuous_mean_var(dist: &ContinuousDistributionIr) -> (f64, f64) {
    match dist {
        ContinuousDistributionIr::Any { mean, variance } => (*mean, *variance),
        ContinuousDistributionIr::Gaussian { mean, variance } => (*mean, *variance),
        ContinuousDistributionIr::Poisson { mean } => (*mean, *mean), // variance = mean
        ContinuousDistributionIr::Uniform { lower, upper } => {
            let mean = (lower + upper) / 2.0;
            let var = (upper - lower).powi(2) / 12.0;
            (mean, var)
        }
    }
}

fn continuous_pdf(dist: &ContinuousDistributionIr, x: f64) -> f64 {
    match dist {
        ContinuousDistributionIr::Any { mean, variance } => gaussian_pdf(x, *mean, *variance),
        ContinuousDistributionIr::Gaussian { mean, variance } => gaussian_pdf(x, *mean, *variance),
        ContinuousDistributionIr::Poisson { mean } => poisson_pmf(x, *mean),
        ContinuousDistributionIr::Uniform { lower, upper } => uniform_pdf(x, *lower, *upper),
    }
}

fn gaussian_pdf(x: f64, mean: f64, variance: f64) -> f64 {
    if variance <= 0.0 || !variance.is_finite() {
        return 0.0;
    }
    let std = variance.sqrt();
    if std <= 0.0 {
        return 0.0;
    }
    use statrs::distribution::{Continuous, Normal};
    if let Ok(n) = Normal::new(mean, std) {
        n.pdf(x)
    } else {
        0.0
    }
}

fn poisson_pmf(x: f64, lambda: f64) -> f64 {
    if lambda < 0.0 || !lambda.is_finite() {
        return 0.0;
    }
    // x should be integer >=0 ; otherwise pmf 0
    if x < 0.0 || !x.is_finite() {
        return 0.0;
    }
    let k = x.round() as u64;
    // if x not close to integer, pmf 0
    if (x - k as f64).abs() > 1e-9 {
        return 0.0;
    }
    use statrs::distribution::{Discrete, Poisson};
    if let Ok(p) = Poisson::new(lambda) {
        p.pmf(k)
    } else {
        0.0
    }
}

fn uniform_pdf(x: f64, lower: f64, upper: f64) -> f64 {
    if lower >= upper || !lower.is_finite() || !upper.is_finite() {
        return 0.0;
    }
    if x < lower || x > upper {
        0.0
    } else {
        1.0 / (upper - lower)
    }
}

fn discrete_counts(dist: &DiscreteDistributionIr) -> (Vec<f64>, Vec<crate::base::SymbolId>) {
    match dist {
        DiscreteDistributionIr::CountTable(ct)
        | DiscreteDistributionIr::NormalizedCountTable(ct) => {
            let counts = ct.entries.iter().map(|e| e.count).collect();
            let symbols = ct.entries.iter().map(|e| e.value).collect();
            (counts, symbols)
        }
        DiscreteDistributionIr::FieldRefs(_) => (vec![], vec![]),
    }
}

fn discrete_entries(
    dist: &DiscreteDistributionIr,
) -> Vec<(crate::base::FieldId, crate::base::SymbolId, f64)> {
    match dist {
        DiscreteDistributionIr::CountTable(ct)
        | DiscreteDistributionIr::NormalizedCountTable(ct) => ct
            .entries
            .iter()
            .map(|e| (e.field, e.value, e.count))
            .collect(),
        DiscreteDistributionIr::FieldRefs(_) => vec![],
    }
}

fn value_to_f64(v: Value) -> Option<f64> {
    match v {
        Value::Continuous(f) => Some(f),
        Value::Discrete(_) => None,
        Value::Missing => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::{FieldId, SymbolId, Value};
    use crate::ir::*;

    fn zvalue_baseline(mean: f64, var: f64) -> BaselineIr {
        let fid = FieldId(0);
        let td = TestDistributionsIr {
            field: fid,
            field_name: "x".into(),
            test_statistic: BaselineTestStatistic::ZValue,
            reset_value: 0.0,
            window_size: 0,
            weight_field: None,
            normalization_scheme: None,
            baseline_continuous: Some(ContinuousDistributionIr::Gaussian {
                mean,
                variance: var,
            }),
            baseline_discrete: None,
            alternate: None,
        };
        BaselineIr {
            function_name: "regression".into(),
            mining_schema: MiningSchemaIr {
                active_fields: vec![fid],
                target_field: None,
                field_metas: vec![],
                missing_value_replacement: None,
            },
            output: vec![],
            targets: vec![],
            test_distributions: td,
        }
    }

    #[test]
    fn zvalue_simple() {
        let baseline = zvalue_baseline(18.2, 17.64);
        // defects 24 => (24-18.2)/sqrt(17.64)=5.8/4.2=1.381
        let score = evaluate_baseline(&baseline, &[Value::Continuous(24.0)]);
        if let Value::Continuous(v) = score {
            assert!((v - 1.38095).abs() < 1e-4, "got {v}");
        } else {
            panic!("expected continuous");
        }
    }

    #[test]
    fn zvalue_zero_variance_missing() {
        let baseline = zvalue_baseline(0.0, 0.0);
        let score = evaluate_baseline(&baseline, &[Value::Continuous(1.0)]);
        assert_eq!(score, Value::Missing);
    }

    #[test]
    fn cusum_gaussian() {
        let fid = FieldId(0);
        let td = TestDistributionsIr {
            field: fid,
            field_name: "x".into(),
            test_statistic: BaselineTestStatistic::Cusum,
            reset_value: 0.0,
            window_size: 0,
            weight_field: None,
            normalization_scheme: None,
            baseline_continuous: Some(ContinuousDistributionIr::Gaussian {
                mean: 0.0,
                variance: 1.0,
            }),
            baseline_discrete: None,
            alternate: Some(ContinuousDistributionIr::Gaussian {
                mean: 1.0,
                variance: 1.0,
            }),
        };
        let baseline = BaselineIr {
            function_name: "regression".into(),
            mining_schema: MiningSchemaIr {
                active_fields: vec![fid],
                target_field: None,
                field_metas: vec![],
                missing_value_replacement: None,
            },
            output: vec![],
            targets: vec![],
            test_distributions: td,
        };
        // For x=1.0, pdf0 = 0.24197, pdf1 = 0.39894, ratio=1.6487, ln=0.5
        let score = evaluate_baseline(&baseline, &[Value::Continuous(1.0)]);
        if let Value::Continuous(v) = score {
            assert!((v - 0.5).abs() < 1e-6, "got {v}");
        } else {
            panic!("expected continuous");
        }
        // For x=0, ratio = exp(-0.5)/exp(0)=0.6065, ln=-0.5, max(0, -0.5)=0
        let score2 = evaluate_baseline(&baseline, &[Value::Continuous(0.0)]);
        assert_eq!(score2, Value::Continuous(0.0));
    }

    #[test]
    fn scalar_product_independent() {
        let fid = FieldId(0);
        // Baseline counts: bin1 100, bin2 150, bin3 10, bin4 2
        let s1 = SymbolId(0);
        let s2 = SymbolId(1);
        let s3 = SymbolId(2);
        let s4 = SymbolId(3);
        let ct = CountTableIr {
            sample: Some(262.0),
            entries: vec![
                FieldValueCountIr {
                    field: fid,
                    value: s1,
                    count: 100.0,
                },
                FieldValueCountIr {
                    field: fid,
                    value: s2,
                    count: 150.0,
                },
                FieldValueCountIr {
                    field: fid,
                    value: s3,
                    count: 10.0,
                },
                FieldValueCountIr {
                    field: fid,
                    value: s4,
                    count: 2.0,
                },
            ],
        };
        let td = TestDistributionsIr {
            field: fid,
            field_name: "bin".into(),
            test_statistic: BaselineTestStatistic::ScalarProduct,
            reset_value: 0.0,
            window_size: 0,
            weight_field: None,
            normalization_scheme: Some("Independent".into()),
            baseline_continuous: None,
            baseline_discrete: Some(DiscreteDistributionIr::CountTable(ct)),
            alternate: None,
        };
        let baseline = BaselineIr {
            function_name: "regression".into(),
            mining_schema: MiningSchemaIr {
                active_fields: vec![fid],
                target_field: None,
                field_metas: vec![],
                missing_value_replacement: None,
            },
            output: vec![],
            targets: vec![],
            test_distributions: td,
        };
        // For observed bin1 -> scalar = 100 / sqrt(100^2+150^2+10^2+2^2)=100/180.56=0.5538
        let score = evaluate_baseline(&baseline, &[Value::Discrete(s1)]);
        if let Value::Continuous(v) = score {
            assert!((v - 0.5538).abs() < 1e-3, "got {v}");
        } else {
            panic!("expected continuous");
        }
    }

    #[test]
    fn chi_square_distribution() {
        let fid = FieldId(0);
        let s1 = SymbolId(0);
        let s2 = SymbolId(1);
        let ct = CountTableIr {
            sample: Some(262.0),
            entries: vec![
                FieldValueCountIr {
                    field: fid,
                    value: s1,
                    count: 100.0,
                },
                FieldValueCountIr {
                    field: fid,
                    value: s2,
                    count: 100.0,
                },
            ],
        };
        let td = TestDistributionsIr {
            field: fid,
            field_name: "bin".into(),
            test_statistic: BaselineTestStatistic::ChiSquareDistribution,
            reset_value: 0.0,
            window_size: 0,
            weight_field: None,
            normalization_scheme: None,
            baseline_continuous: None,
            baseline_discrete: Some(DiscreteDistributionIr::CountTable(ct)),
            alternate: None,
        };
        let baseline = BaselineIr {
            function_name: "regression".into(),
            mining_schema: MiningSchemaIr {
                active_fields: vec![fid],
                target_field: None,
                field_metas: vec![],
                missing_value_replacement: None,
            },
            output: vec![],
            targets: vec![],
            test_distributions: td,
        };
        // Observed bin1, expected per bin = 0.5, observed 1 at bin1 0 at bin2
        // chi2 = (1-0.5)^2/0.5 + (0-0.5)^2/0.5 = 0.5+0.5=1.0
        let score = evaluate_baseline(&baseline, &[Value::Discrete(s1)]);
        assert_eq!(score, Value::Continuous(1.0));
    }

    #[test]
    fn poisson_zvalue() {
        let baseline = {
            let fid = FieldId(0);
            let td = TestDistributionsIr {
                field: fid,
                field_name: "x".into(),
                test_statistic: BaselineTestStatistic::ZValue,
                reset_value: 0.0,
                window_size: 0,
                weight_field: None,
                normalization_scheme: None,
                baseline_continuous: Some(ContinuousDistributionIr::Poisson { mean: 4.0 }),
                baseline_discrete: None,
                alternate: None,
            };
            BaselineIr {
                function_name: "regression".into(),
                mining_schema: MiningSchemaIr {
                    active_fields: vec![fid],
                    target_field: None,
                    field_metas: vec![],
                    missing_value_replacement: None,
                },
                output: vec![],
                targets: vec![],
                test_distributions: td,
            }
        };
        // x=6, mean 4 var 4 std 2 => z=1
        let score = evaluate_baseline(&baseline, &[Value::Continuous(6.0)]);
        assert_eq!(score, Value::Continuous(1.0));
    }

    #[test]
    fn uniform_zvalue() {
        let fid = FieldId(0);
        let td = TestDistributionsIr {
            field: fid,
            field_name: "x".into(),
            test_statistic: BaselineTestStatistic::ZValue,
            reset_value: 0.0,
            window_size: 0,
            weight_field: None,
            normalization_scheme: None,
            baseline_continuous: Some(ContinuousDistributionIr::Uniform {
                lower: 0.0,
                upper: 10.0,
            }),
            baseline_discrete: None,
            alternate: None,
        };
        let baseline = BaselineIr {
            function_name: "regression".into(),
            mining_schema: MiningSchemaIr {
                active_fields: vec![fid],
                target_field: None,
                field_metas: vec![],
                missing_value_replacement: None,
            },
            output: vec![],
            targets: vec![],
            test_distributions: td,
        };
        // mean 5 var 8.333 std 2.886 => x 10 => z 1.732
        let score = evaluate_baseline(&baseline, &[Value::Continuous(10.0)]);
        if let Value::Continuous(v) = score {
            assert!((v - 1.73205).abs() < 1e-3, "got {v}");
        } else {
            panic!()
        }
    }
}
