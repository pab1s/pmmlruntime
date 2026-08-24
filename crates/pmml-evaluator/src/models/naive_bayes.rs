use pmml_core::Value;
use pmml_ir::ir::NaiveBayesIr;

/// Gaussian PDF
fn gaussian_pdf(x: f64, mean: f64, variance: f64) -> f64 {
    if variance <= 0.0 {
        return 0.0;
    }
    let denom = (2.0 * std::f64::consts::PI * variance).sqrt();
    let num = (-((x - mean) * (x - mean)) / (2.0 * variance)).exp();
    num / denom
}

pub fn evaluate_naive_bayes(nb: &NaiveBayesIr, values: &[Value]) -> Value {
    if nb.bayes_output_counts.is_empty() {
        return Value::Missing;
    }

    // Prior counts from BayesOutput
    let total_prior: f64 = nb.bayes_output_counts.iter().map(|c| c.count).sum();
    if total_prior == 0.0 {
        // Fallback to uniform
        return Value::Discrete(nb.bayes_output_counts[0].value);
    }

    // Compute log probabilities per target for threshold handling
    let mut log_probs: std::collections::HashMap<pmml_core::SymbolId, f64> = std::collections::HashMap::new();
    let mut best_score = f64::NEG_INFINITY;
    let mut best_value: Option<pmml_core::SymbolId> = None;

    for target_count in &nb.bayes_output_counts {
        let prior = target_count.count / total_prior;
        if prior <= 0.0 {
            log_probs.insert(target_count.value, f64::NEG_INFINITY);
            continue;
        }
        let mut log_prob = prior.ln();

        for bayes_input in &nb.bayes_inputs {
            let fid = bayes_input.field.as_usize();
            let actual = if fid < values.len() {
                values[fid]
            } else {
                Value::Missing
            };
            if actual.is_missing() {
                continue;
            }

            // Check if this BayesInput has target_value_stats (continuous with Gaussian)
            // Or pair_counts (categorical)
            let mut found = false;
            // Try Gaussian first
            for tvs in &bayes_input.target_value_stats {
                if tvs.value == target_count.value {
                    if let (Value::Continuous(x), Some(mean), Some(var)) =
                        (actual, tvs.mean, tvs.variance)
                    {
                        let pdf = gaussian_pdf(x, mean, var);
                        if pdf > 0.0 {
                            log_prob += pdf.ln();
                        } else {
                            log_prob += f64::NEG_INFINITY;
                        }
                        found = true;
                        break;
                    }
                }
            }
            if found {
                continue;
            }
            // Try PairCounts (categorical)
            for pc in &bayes_input.pair_counts {
                // Need to check if actual discrete value equals pc.value
                let matches = match actual {
                    Value::Discrete(sid) => sid == pc.value,
                    Value::Continuous(_f) => {
                        // For categorical double like "1.0", need to handle as discrete string?
                        // Try to compare as string via symbol? For v1, if actual is continuous but expected discrete, try to see if f as string matches
                        // For simplicity, if actual is continuous and pc.value corresponds to discrete string representation of f, we can try
                        // But for now, just not match
                        false
                    }
                    _ => false,
                };
                if matches {
                    // Find count for this target
                    let mut count = 0.0;
                    let mut total_for_input = 0.0;
                    for tc in &pc.target_counts {
                        total_for_input += tc.count;
                        if tc.value == target_count.value {
                            count = tc.count;
                        }
                    }
                    if total_for_input > 0.0 {
                        let prob = count / total_for_input;
                        if prob > 0.0 {
                            log_prob += prob.ln();
                        } else {
                            log_prob += f64::NEG_INFINITY;
                        }
                    }
                    found = true;
                    break;
                }
            }
            if !found {
                // No matching BayesInput for this target, treat as uniform? Skip
            }
        }

        log_probs.insert(target_count.value, log_prob);
        if log_prob > best_score {
            best_score = log_prob;
            best_value = Some(target_count.value);
        }
    }

    // Threshold handling: if best probability < threshold, return Missing (per PMML spec)
    // threshold is e.g., 0.001 to avoid low-confidence predictions
    if nb.threshold > 0.0 {
        if let Some(best_sid) = best_value {
            if let Some(&best_log) = log_probs.get(&best_sid) {
                // Compute normalized probability via softmax of log probs (exp(log)/sum exp)
                let mut sum_exp = 0.0;
                let mut max_log = f64::NEG_INFINITY;
                for &lp in log_probs.values() {
                    if lp > max_log {
                        max_log = lp;
                    }
                }
                // Subtract max for numerical stability
                for &lp in log_probs.values() {
                    if lp.is_finite() {
                        sum_exp += (lp - max_log).exp();
                    }
                }
                let best_prob = if sum_exp > 0.0 {
                    (best_log - max_log).exp() / sum_exp
                } else {
                    0.0
                };
                if best_prob < nb.threshold {
                    return Value::Missing;
                }
            }
        }
    }

    if let Some(v) = best_value {
        Value::Discrete(v)
    } else {
        Value::Missing
    }
}
