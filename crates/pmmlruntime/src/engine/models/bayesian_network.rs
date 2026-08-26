//! BayesianNetworkModel evaluation — exact inference via enumeration.
//!
//! Implements `BayesianNetworkModel` where each `BayesianNetworkNodes` entry is either `DiscreteNode`
//! or `ContinuousNode`. `DiscreteNode`s carry `ValueProbability` marginals and `DiscreteConditionalProbability`
//! tables (parents → value probs). `ContinuousNode`s carry `ContinuousDistribution`s (Normal/Lognormal/Uniform/Triangular)
//! with `Mean`/`Variance`/`Lower`/`Upper` expressions that may reference parent fields.
//!
//! Scoring (inference) is exact enumeration over unobserved discrete variables:
//!
//! 1. Evidence is taken from `MiningSchema.active_fields` where input `values[FieldId]` is not `Missing`.
//! 2. Discrete domains are the union of `value_probabilities` and conditional table probabilities.
//! 3. All assignments to unobserved discrete nodes are enumerated; for each assignment:
//!    - Continuous unobserved nodes' expected means are computed via `eval_ops` on their distribution `Mean` expressions.
//!    - Node-local `DerivedField`s (e.g. `C3_Discretized` from `C3`) are re-evaluated via `eval_ops` to obtain discrete parent values.
//!    - Joint weight = Π discrete probs × Π pdf(continuous evidence) is computed.
//! 4. For a discrete target, posterior = weight per value / total weight, predicted = argmax.
//!    For a continuous target, posterior mean = Σ weight·expected / total weight.

use crate::base::{FieldId, Value};
use crate::ir::{
    BayesianContinuousDistributionIr, BayesianNetworkIr, BayesianNodeIr, ContinuousBayesianNodeIr,
    DiscreteBayesianNodeIr, DiscreteConditionalTableIr,
};
use std::collections::HashMap;

fn find_discrete_table<'a>(
    dn: &'a DiscreteBayesianNodeIr,
    temp: &[Value],
) -> Option<&'a DiscreteConditionalTableIr> {
    if dn.conditional_tables.is_empty() {
        return None;
    }
    for ct in &dn.conditional_tables {
        let mut matches = true;
        for pv in &ct.parent_values {
            let idx = pv.parent.as_usize();
            let actual = if idx < temp.len() {
                temp[idx]
            } else {
                Value::Missing
            };
            match actual {
                Value::Discrete(sid) => {
                    if sid != pv.value {
                        matches = false;
                        break;
                    }
                }
                _ => {
                    matches = false;
                    break;
                }
            }
        }
        if matches {
            return Some(ct);
        }
    }
    None
}

fn find_continuous_dist<'a>(
    cn: &'a ContinuousBayesianNodeIr,
    temp: &[Value],
) -> Option<&'a BayesianContinuousDistributionIr> {
    if !cn.conditional_tables.is_empty() {
        for ct in &cn.conditional_tables {
            let mut matches = true;
            for pv in &ct.parent_values {
                let idx = pv.parent.as_usize();
                let actual = if idx < temp.len() {
                    temp[idx]
                } else {
                    Value::Missing
                };
                match actual {
                    Value::Discrete(sid) => {
                        if sid != pv.value {
                            matches = false;
                            break;
                        }
                    }
                    _ => {
                        matches = false;
                        break;
                    }
                }
            }
            if matches {
                if let Some(d) = ct.distributions.first() {
                    return Some(d);
                }
            }
        }
        cn.distributions.first()
    } else {
        cn.distributions.first()
    }
}

/// Evaluate a [`BayesianNetworkIr`] against a dense `values` array.
pub fn evaluate_bayesian_network(model: &BayesianNetworkIr, values: &[Value]) -> Value {
    // Determine primary target field
    let target_fid_opt = model.mining_schema.target_field.or_else(|| {
        // fallback: first node's field if mining schema has no target
        model.nodes.first().map(|n| match n {
            BayesianNodeIr::Discrete(d) => d.field,
            BayesianNodeIr::Continuous(c) => c.field,
        })
    });
    let target_fid = match target_fid_opt {
        Some(f) => f,
        None => return Value::Missing,
    };

    // Determine target node type
    let target_is_discrete = model.nodes.iter().any(|n| match n {
        BayesianNodeIr::Discrete(d) => d.field == target_fid,
        BayesianNodeIr::Continuous(_) => false,
    });
    let target_is_continuous = !target_is_discrete
        && model.nodes.iter().any(|n| match n {
            BayesianNodeIr::Continuous(c) => c.field == target_fid,
            BayesianNodeIr::Discrete(_) => false,
        });

    // Build node lookup maps
    let mut discrete_nodes: Vec<&DiscreteBayesianNodeIr> = Vec::new();
    let mut continuous_nodes: Vec<&ContinuousBayesianNodeIr> = Vec::new();
    for node in &model.nodes {
        match node {
            BayesianNodeIr::Discrete(d) => discrete_nodes.push(d),
            BayesianNodeIr::Continuous(c) => continuous_nodes.push(c),
        }
    }

    // Build evidence map for Bayesian node fields that are active and observed
    // Need to know which FieldIds correspond to node fields; then evidence is values[FieldId] if not Missing
    // Also need to collect field->node for quick lookup
    let mut evidence: HashMap<FieldId, Value> = HashMap::new();
    for &fid in &model.mining_schema.active_fields {
        let idx = fid.as_usize();
        if idx < values.len() {
            let v = values[idx];
            if !v.is_missing() {
                // Check if fid is a Bayesian node field (or derived? derived evidence not needed)
                let is_node_field = model.nodes.iter().any(|n| match n {
                    BayesianNodeIr::Discrete(d) => d.field == fid,
                    BayesianNodeIr::Continuous(c) => c.field == fid,
                });
                if is_node_field {
                    evidence.insert(fid, v);
                } else {
                    // Could be group/order field, ignore? But also might be evidence for derived parents that are not nodes
                    // For now ignore
                }
            }
        }
    }
    // Also consider any node field that has observed value even if not in active_fields (e.g., continuous evidence passed as discrete via active? Already covered)
    // But in case values contains observed for a target field that is also active? Not.

    // Build discrete domains
    let mut discrete_domains: HashMap<FieldId, Vec<crate::base::SymbolId>> = HashMap::new();
    for dn in &discrete_nodes {
        let mut set = std::collections::HashSet::new();
        let mut vec = Vec::new();
        for vp in &dn.value_probabilities {
            if set.insert(vp.value) {
                vec.push(vp.value);
            }
        }
        for ct in &dn.conditional_tables {
            for vp in &ct.value_probabilities {
                if set.insert(vp.value) {
                    vec.push(vp.value);
                }
            }
        }
        // Also include values from DataDictionary? but we have at least from tables
        // If still empty, fallback to single value? Should not happen
        if vec.is_empty() {
            // Try to get from evidence? No
            // leave empty -> enumeration will skip? Better to keep empty and treat as 0 combos?
        }
        discrete_domains.insert(dn.field, vec);
    }

    // Unobserved discrete fields = discrete nodes where evidence missing
    let mut unobserved: Vec<(FieldId, Vec<crate::base::SymbolId>)> = Vec::new();
    for dn in &discrete_nodes {
        if !evidence.contains_key(&dn.field) {
            if let Some(dom) = discrete_domains.get(&dn.field) {
                if !dom.is_empty() {
                    unobserved.push((dn.field, dom.clone()));
                }
            }
        }
    }

    // If no unobserved discrete, we have single assignment
    // Enumerate
    let total_combos: usize = if unobserved.is_empty() {
        1
    } else {
        unobserved
            .iter()
            .map(|(_, dom)| dom.len())
            .product::<usize>()
    };
    // Guard against explosion: cap at 1M? For our tests it's small
    if total_combos > 1_000_000 {
        // fallback to marginal most probable
        return fallback_marginal(model, target_fid);
    }

    // Helper to evaluate param ops to f64
    let eval_param = |ops: &[crate::ir::Op], temp: &[Value]| -> f64 {
        let v = crate::engine::transform::vm::eval_ops(ops, temp);
        match v {
            Value::Continuous(f) => f,
            Value::Discrete(sid) => {
                // Try to parse discrete symbol as f64 if it looks numeric (for cases where Constant was string "2"? But Constant for continuous should be Continuous, not Discrete)
                // We'll try to lookup symbol string if possible via thread local? But we don't have mapping here; assume 0
                let _ = sid;
                0.0
            }
            Value::Missing => f64::NAN,
        }
    };

    // find_discrete_table and find_continuous_dist are top-level helpers

    // Helpers for probability lookup
    let prob_for_discrete =
        |dn: &DiscreteBayesianNodeIr, sid: crate::base::SymbolId, temp: &[Value]| -> f64 {
            if let Some(ct) = find_discrete_table(dn, temp) {
                for vp in &ct.value_probabilities {
                    if vp.value == sid {
                        return vp.probability;
                    }
                }
                0.0
            } else {
                for vp in &dn.value_probabilities {
                    if vp.value == sid {
                        return vp.probability;
                    }
                }
                0.0
            }
        };

    let pdf_for_continuous = |cn: &ContinuousBayesianNodeIr, x: f64, temp: &[Value]| -> f64 {
        let dist = match find_continuous_dist(cn, temp) {
            Some(d) => d,
            None => return 0.0,
        };
        match dist {
            BayesianContinuousDistributionIr::Normal { mean, variance } => {
                let m = eval_param(mean, temp);
                let v = eval_param(variance, temp);
                if !m.is_finite() || !v.is_finite() || v <= 0.0 {
                    return 0.0;
                }
                let std = v.sqrt();
                // Use statrs if available? We'll compute directly
                let denom = std * (2.0 * std::f64::consts::PI).sqrt();
                let num = (-0.5 * ((x - m) / std).powi(2)).exp();
                num / denom
            }
            BayesianContinuousDistributionIr::Lognormal { mean, variance } => {
                let m = eval_param(mean, temp);
                let v = eval_param(variance, temp);
                if !m.is_finite() || !v.is_finite() || v <= 0.0 || x <= 0.0 {
                    return 0.0;
                }
                let std = v.sqrt();
                let denom = x * std * (2.0 * std::f64::consts::PI).sqrt();
                let num = (-0.5 * ((x.ln() - m) / std).powi(2)).exp();
                num / denom
            }
            BayesianContinuousDistributionIr::Uniform { lower, upper } => {
                let l = eval_param(lower, temp);
                let u = eval_param(upper, temp);
                if !l.is_finite() || !u.is_finite() || u <= l {
                    return 0.0;
                }
                if x >= l && x <= u {
                    1.0 / (u - l)
                } else {
                    0.0
                }
            }
            BayesianContinuousDistributionIr::Triangular { mean, lower, upper } => {
                let m = eval_param(mean, temp);
                let l = eval_param(lower, temp);
                let u = eval_param(upper, temp);
                if !l.is_finite() || !u.is_finite() || !m.is_finite() || u <= l {
                    return 0.0;
                }
                if x < l || x > u {
                    return 0.0;
                }
                if x <= m {
                    if (m - l).abs() < 1e-9 {
                        return 0.0;
                    }
                    2.0 * (x - l) / ((u - l) * (m - l))
                } else {
                    if (u - m).abs() < 1e-9 {
                        return 0.0;
                    }
                    2.0 * (u - x) / ((u - l) * (u - m))
                }
            }
        }
    };

    let mean_for_continuous = |cn: &ContinuousBayesianNodeIr, temp: &[Value]| -> f64 {
        let dist = match find_continuous_dist(cn, temp) {
            Some(d) => d,
            None => return 0.0,
        };
        match dist {
            BayesianContinuousDistributionIr::Normal { mean, .. } => eval_param(mean, temp),
            BayesianContinuousDistributionIr::Lognormal { mean, variance } => {
                let m = eval_param(mean, temp);
                let v = eval_param(variance, temp);
                if m.is_finite() && v.is_finite() {
                    (m + v / 2.0).exp()
                } else {
                    m
                }
            }
            BayesianContinuousDistributionIr::Uniform { lower, upper } => {
                let l = eval_param(lower, temp);
                let u = eval_param(upper, temp);
                if l.is_finite() && u.is_finite() {
                    (l + u) / 2.0
                } else {
                    0.0
                }
            }
            BayesianContinuousDistributionIr::Triangular { mean, lower, upper } => {
                let m = eval_param(mean, temp);
                if m.is_finite() {
                    m
                } else {
                    let l = eval_param(lower, temp);
                    let u = eval_param(upper, temp);
                    if l.is_finite() && u.is_finite() {
                        (l + u + m) / 3.0
                    } else {
                        0.0
                    }
                }
            }
        }
    };

    // Accumulators
    let mut total_weight = 0.0;
    let mut discrete_sums: HashMap<crate::base::SymbolId, f64> = HashMap::new();
    let mut continuous_weighted_sum = 0.0;

    // Precompute target discrete domain if needed
    let target_domain_opt = if target_is_discrete {
        discrete_domains.get(&target_fid).cloned()
    } else {
        None
    };

    // Enumerate
    for combo_idx in 0..total_combos {
        // Build temp values
        let mut temp = values.to_vec();
        // Ensure temp length covers all FieldIds up to max needed (if values shorter, extend with Missing)
        let needed_len = {
            let mut max_id = 0usize;
            for node in &model.nodes {
                let fid = match node {
                    BayesianNodeIr::Discrete(d) => d.field.as_usize(),
                    BayesianNodeIr::Continuous(c) => c.field.as_usize(),
                };
                if fid > max_id {
                    max_id = fid;
                }
                // also check derived fields
                let dfs = match node {
                    BayesianNodeIr::Discrete(d) => &d.derived_fields,
                    BayesianNodeIr::Continuous(c) => &c.derived_fields,
                };
                for df in dfs {
                    let id = df.field_id.as_usize();
                    if id > max_id {
                        max_id = id;
                    }
                }
            }
            // also consider mining schema fields
            for &fid in &model.mining_schema.active_fields {
                if fid.as_usize() > max_id {
                    max_id = fid.as_usize();
                }
            }
            if let Some(tf) = model.mining_schema.target_field {
                if tf.as_usize() > max_id {
                    max_id = tf.as_usize();
                }
            }
            max_id + 1
        };
        if temp.len() < needed_len {
            temp.resize(needed_len, Value::Missing);
        }

        // Decode combo into assignments for unobserved discrete nodes
        let mut rem = combo_idx;
        for (fid, domain) in &unobserved {
            let dlen = domain.len();
            let idx = rem % dlen;
            rem /= dlen;
            let sid = domain[idx];
            let fidx = fid.as_usize();
            if fidx < temp.len() {
                temp[fidx] = Value::Discrete(sid);
            }
        }
        // For discrete nodes that are observed, their values already in temp (evidence). Ensure they are preserved (they are from base values)
        // But also for discrete nodes where evidence missing but domain assignment gave value, we have set.

        // Compute continuous expected values for unobserved continuous nodes in order
        for cn in &continuous_nodes {
            let fidx = cn.field.as_usize();
            if fidx < temp.len() && !temp[fidx].is_missing() {
                continue; // observed
            }
            // Find distribution and compute mean
            let mean = mean_for_continuous(cn, &temp);
            if fidx < temp.len() {
                if mean.is_finite() {
                    temp[fidx] = Value::Continuous(mean);
                } else {
                    temp[fidx] = Value::Missing;
                }
            }
        }

        // Re-evaluate derived fields for all nodes (to compute discretization parents)
        for node in &model.nodes {
            let dfs = match node {
                BayesianNodeIr::Discrete(d) => &d.derived_fields,
                BayesianNodeIr::Continuous(c) => &c.derived_fields,
            };
            for df in dfs {
                let v = crate::engine::transform::vm::eval_ops(&df.bytecode, &temp);
                let idx = df.field_id.as_usize();
                if idx < temp.len() {
                    temp[idx] = v;
                }
            }
        }

        // Compute joint weight
        let mut w = 1.0;

        // Discrete nodes contribution
        for dn in &discrete_nodes {
            let fidx = dn.field.as_usize();
            let actual = if fidx < temp.len() {
                temp[fidx]
            } else {
                Value::Missing
            };
            let sid = match actual {
                Value::Discrete(s) => s,
                Value::Missing => {
                    w = 0.0;
                    break;
                }
                Value::Continuous(_) => {
                    w = 0.0;
                    break;
                }
            };
            let p = prob_for_discrete(dn, sid, &temp);
            w *= p;
            if w == 0.0 {
                break;
            }
        }
        if w == 0.0 {
            continue;
        }
        // Continuous observed evidence contribution (pdf)
        for cn in &continuous_nodes {
            // Check if this continuous node is observed evidence (evidence map contains it)
            if let Some(ev) = evidence.get(&cn.field) {
                let x = match ev {
                    Value::Continuous(f) => *f,
                    Value::Discrete(_) => continue, // discrete evidence for continuous node? ignore
                    Value::Missing => continue,
                };
                let pdf = pdf_for_continuous(cn, x, &temp);
                // If pdf is 0, weight becomes 0
                if pdf <= 0.0 || !pdf.is_finite() {
                    w = 0.0;
                    break;
                }
                w *= pdf;
                if w == 0.0 {
                    break;
                }
            }
        }
        if w == 0.0 {
            continue;
        }

        total_weight += w;
        if target_is_discrete {
            let tidx = target_fid.as_usize();
            let tv = if tidx < temp.len() {
                temp[tidx]
            } else {
                Value::Missing
            };
            if let Value::Discrete(sid) = tv {
                *discrete_sums.entry(sid).or_insert(0.0) += w;
            }
        } else if target_is_continuous {
            let tidx = target_fid.as_usize();
            let tv = if tidx < temp.len() {
                temp[tidx]
            } else {
                Value::Missing
            };
            if let Value::Continuous(f) = tv {
                if f.is_finite() {
                    continuous_weighted_sum += w * f;
                }
            } else {
                // If target continuous value is Missing (maybe not computed), try mean_for_continuous
                if let Some(cn) = continuous_nodes.iter().find(|c| c.field == target_fid) {
                    let m = mean_for_continuous(cn, &temp);
                    if m.is_finite() {
                        continuous_weighted_sum += w * m;
                    }
                }
            }
        }
    }

    if total_weight == 0.0 || !total_weight.is_finite() {
        return fallback_marginal(model, target_fid);
    }

    if target_is_discrete {
        // Find max posterior
        let mut best_sid: Option<crate::base::SymbolId> = None;
        let mut best_prob = -1.0;
        if let Some(domain) = target_domain_opt {
            for sid in domain {
                let sum = discrete_sums.get(&sid).copied().unwrap_or(0.0);
                let prob = sum / total_weight;
                if prob > best_prob {
                    best_prob = prob;
                    best_sid = Some(sid);
                }
            }
        } else {
            for (sid, sum) in &discrete_sums {
                let prob = sum / total_weight;
                if prob > best_prob {
                    best_prob = prob;
                    best_sid = Some(*sid);
                }
            }
        }
        if let Some(sid) = best_sid {
            return Value::Discrete(sid);
        } else {
            return fallback_marginal(model, target_fid);
        }
    } else if target_is_continuous {
        let mean = continuous_weighted_sum / total_weight;
        if mean.is_finite() {
            return Value::Continuous(mean);
        } else {
            return Value::Missing;
        }
    } else {
        // Target field not found among Bayesian nodes - fallback to marginal of first node?
        fallback_marginal(model, target_fid)
    }
}

fn fallback_marginal(model: &BayesianNetworkIr, target: FieldId) -> Value {
    for node in &model.nodes {
        match node {
            BayesianNodeIr::Discrete(d) if d.field == target => {
                // return most probable marginal
                let mut best: Option<(crate::base::SymbolId, f64)> = None;
                for vp in &d.value_probabilities {
                    if best.is_none() || vp.probability > best.unwrap().1 {
                        best = Some((vp.value, vp.probability));
                    }
                }
                if let Some((sid, _)) = best {
                    return Value::Discrete(sid);
                }
                // fallback to first conditional value if no marginal
                if let Some(ct) = d.conditional_tables.first() {
                    if let Some(vp) = ct.value_probabilities.first() {
                        return Value::Discrete(vp.value);
                    }
                }
                return Value::Missing;
            }
            BayesianNodeIr::Continuous(c) if c.field == target => {
                if let Some(dist) = c.distributions.first() {
                    // Need dummy values for eval? Use empty
                    let dummy = vec![Value::Missing; 64];
                    let m = match dist {
                        BayesianContinuousDistributionIr::Normal { mean, .. } => {
                            crate::engine::transform::vm::eval_ops(mean, &dummy)
                        }
                        BayesianContinuousDistributionIr::Lognormal { mean, .. } => {
                            crate::engine::transform::vm::eval_ops(mean, &dummy)
                        }
                        BayesianContinuousDistributionIr::Uniform { lower, upper } => {
                            let l = crate::engine::transform::vm::eval_ops(lower, &dummy);
                            let u = crate::engine::transform::vm::eval_ops(upper, &dummy);
                            match (l, u) {
                                (Value::Continuous(lf), Value::Continuous(uf)) => {
                                    Value::Continuous((lf + uf) / 2.0)
                                }
                                _ => Value::Missing,
                            }
                        }
                        BayesianContinuousDistributionIr::Triangular { mean, .. } => {
                            crate::engine::transform::vm::eval_ops(mean, &dummy)
                        }
                    };
                    return m;
                }
                return Value::Continuous(0.0);
            }
            _ => {}
        }
    }
    Value::Missing
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::{FieldId, SymbolId, Value};
    use crate::ir::*;

    fn make_simple_bn() -> BayesianNetworkIr {
        // Simple BN: A->C, B->C ; A,B roots with priors, C conditional as in spec exact example
        let fid_a = FieldId(0);
        let fid_b = FieldId(1);
        let fid_c = FieldId(2);
        let sid_0 = SymbolId(10);
        let sid_1 = SymbolId(11);
        let sid_2 = SymbolId(12);
        // Discrete nodes A,B,C each categorical "0","1","2"
        let node_a = DiscreteBayesianNodeIr {
            name: "A".into(),
            field: fid_a,
            count: None,
            value_probabilities: vec![
                BayesianValueProbabilityIr {
                    value: sid_0,
                    probability: 0.4,
                },
                BayesianValueProbabilityIr {
                    value: sid_1,
                    probability: 0.6,
                },
            ],
            conditional_tables: vec![],
            derived_fields: vec![],
        };
        let node_b = DiscreteBayesianNodeIr {
            name: "B".into(),
            field: fid_b,
            count: None,
            value_probabilities: vec![
                BayesianValueProbabilityIr {
                    value: sid_0,
                    probability: 0.7,
                },
                BayesianValueProbabilityIr {
                    value: sid_1,
                    probability: 0.3,
                },
            ],
            conditional_tables: vec![],
            derived_fields: vec![],
        };
        let node_c = DiscreteBayesianNodeIr {
            name: "C".into(),
            field: fid_c,
            count: None,
            value_probabilities: vec![],
            conditional_tables: vec![
                DiscreteConditionalTableIr {
                    parent_values: vec![
                        BayesianParentValueIr {
                            parent: fid_a,
                            value: sid_0,
                        },
                        BayesianParentValueIr {
                            parent: fid_b,
                            value: sid_0,
                        },
                    ],
                    value_probabilities: vec![
                        BayesianValueProbabilityIr {
                            value: sid_0,
                            probability: 0.7,
                        },
                        BayesianValueProbabilityIr {
                            value: sid_1,
                            probability: 0.2,
                        },
                        BayesianValueProbabilityIr {
                            value: sid_2,
                            probability: 0.1,
                        },
                    ],
                    count: None,
                },
                DiscreteConditionalTableIr {
                    parent_values: vec![
                        BayesianParentValueIr {
                            parent: fid_a,
                            value: sid_0,
                        },
                        BayesianParentValueIr {
                            parent: fid_b,
                            value: sid_1,
                        },
                    ],
                    value_probabilities: vec![
                        BayesianValueProbabilityIr {
                            value: sid_0,
                            probability: 0.6,
                        },
                        BayesianValueProbabilityIr {
                            value: sid_1,
                            probability: 0.2,
                        },
                        BayesianValueProbabilityIr {
                            value: sid_2,
                            probability: 0.2,
                        },
                    ],
                    count: None,
                },
                DiscreteConditionalTableIr {
                    parent_values: vec![
                        BayesianParentValueIr {
                            parent: fid_a,
                            value: sid_1,
                        },
                        BayesianParentValueIr {
                            parent: fid_b,
                            value: sid_0,
                        },
                    ],
                    value_probabilities: vec![
                        BayesianValueProbabilityIr {
                            value: sid_0,
                            probability: 0.4,
                        },
                        BayesianValueProbabilityIr {
                            value: sid_1,
                            probability: 0.3,
                        },
                        BayesianValueProbabilityIr {
                            value: sid_2,
                            probability: 0.3,
                        },
                    ],
                    count: None,
                },
                DiscreteConditionalTableIr {
                    parent_values: vec![
                        BayesianParentValueIr {
                            parent: fid_a,
                            value: sid_1,
                        },
                        BayesianParentValueIr {
                            parent: fid_b,
                            value: sid_1,
                        },
                    ],
                    value_probabilities: vec![
                        BayesianValueProbabilityIr {
                            value: sid_0,
                            probability: 0.3,
                        },
                        BayesianValueProbabilityIr {
                            value: sid_1,
                            probability: 0.3,
                        },
                        BayesianValueProbabilityIr {
                            value: sid_2,
                            probability: 0.4,
                        },
                    ],
                    count: None,
                },
            ],
            derived_fields: vec![],
        };
        BayesianNetworkIr {
            function_name: "regression".into(),
            model_name: None,
            algorithm_name: None,
            model_type: Some("General".into()),
            inference_method: Some("Exact".into()),
            is_scorable: true,
            mining_schema: MiningSchemaIr {
                active_fields: vec![fid_c],
                target_field: Some(fid_a),
                field_metas: vec![],
                missing_value_replacement: None,
            },
            output: vec![],
            targets: vec![],
            nodes: vec![
                BayesianNodeIr::Discrete(node_a),
                BayesianNodeIr::Discrete(node_b),
                BayesianNodeIr::Discrete(node_c),
            ],
        }
    }

    #[test]
    fn bayesian_exact_inference() {
        let bn = make_simple_bn();
        // Evidence C=2 should give posterior for A: P(A=0|C=2)=0.208, P(A=1|C=2)=0.792 => predicts 1
        let mut vals = vec![Value::Missing; 3];
        vals[2] = Value::Discrete(SymbolId(12)); // C=2
        let pred = evaluate_bayesian_network(&bn, &vals);
        assert_eq!(pred, Value::Discrete(SymbolId(11))); // A=1
    }

    #[test]
    fn bayesian_no_evidence_marginal() {
        let bn = make_simple_bn();
        let vals = vec![Value::Missing; 3];
        let pred = evaluate_bayesian_network(&bn, &vals);
        // No evidence, marginal for A is 0.4 vs 0.6 => predicts 1
        assert_eq!(pred, Value::Discrete(SymbolId(11)));
    }
}
