//! Support-vector machine evaluation — RBF kernel dot product.
//!
//! Implements `SupportVectorMachineModel` with an RBF (`exp(-γ‖x - sv‖²)`) kernel.
//! Inputs are the ordered `VectorFields` (`VectorDictionary/VectorFields/@field`), each
//! must be `Continuous` (otherwise `Missing`). The raw score is `absoluteValue + Σ coeff[i] * K(input, sv_i)`
//! where `sv_i` is the dense vector from `VectorDictionary/VectorInstances` looked up by
//! `SupportVector/@vectorId` and `coeff` is `Coefficients/@value`. For regression the raw sum is returned
//! as `Continuous`; classification threshold mapping is not yet implemented.
//!
//! # What belongs here
//!
//! - [`evaluate_support_vector_machine`] — the single public entry point.
//!
//! # Performance
//!
//! `O(support_vectors * dims)` where `dims = vector_fields.len()`. No allocation beyond `HashMap` for `vectorId → array`.

use pmml_core::Value;
use pmml_ir::ir::SupportVectorMachineIr;
use std::collections::HashMap;

/// Evaluate a [`SupportVectorMachineIr`] against a dense `values` array.
///
/// Builds the input vector `x` from `svm.vector_fields` (all must be `Continuous`; any `Missing`
/// or `Discrete` yields `Missing`), then computes `sum = absoluteValue + Σ coeff[i] * exp(-γ ‖x - sv_i‖²)`.
/// Dimensionality mismatches (`sv_i.len() != dims`) are skipped. `γ` is `SupportVectorMachine/@gamma`
/// (RBF kernel width).
///
/// # Parameters
///
/// - `svm`: Lowered SVM model (`SupportVectorMachineIr`) with `vector_fields`, `vector_instances` (`id → array`),
///   `support_vectors` (ordered `vectorId`s), `coefficients`, `absolute_value`, `kernel_gamma`.
/// - `values`: Dense `&[Value]` indexed by [`FieldId`](pmml_core::FieldId). Out-of-bounds → `Missing`.
///
/// # Returns
///
/// `Continuous(sum)` on success, `Missing` when `vector_fields` or `support_vectors` is empty,
/// or when any required input is `Missing`/`Discrete`.
///
/// # Panics
///
/// Never panics. All `FieldId` indexing and `support_vectors`/`coefficients` length mismatches are guarded.
///
/// # Performance
///
/// `O(support_vectors * dims)` with one `exp` per support vector. Hash lookup for `vectorId → array`.
///
/// # Examples
///
/// ```
/// use pmml_core::{FieldId, Value};
/// use pmml_ir::ir::*;
/// use pmml_evaluator::models::evaluate_support_vector_machine;
///
/// let f = FieldId(0);
/// let svm = SupportVectorMachineIr {
///     function_name: "regression".into(),
///     mining_schema: MiningSchemaIr { active_fields: vec![f], target_field: None, field_metas: vec![], missing_value_replacement: None },
///     output: vec![],
///     vector_fields: vec![f],
///     vector_instances: vec![("sv1".into(), vec![0.0]), ("sv2".into(), vec![1.0])],
///     support_vectors: vec!["sv1".into(), "sv2".into()],
///     coefficients: vec![1.0, -1.0],
///     absolute_value: 0.0,
///     kernel_gamma: 1.0,
/// };
/// let pred = evaluate_support_vector_machine(&svm, &[Value::Continuous(0.0)]);
/// match pred { Value::Continuous(v) => assert!(v.is_finite()), _ => panic!("expected continuous") }
/// ```
pub fn evaluate_support_vector_machine(svm: &SupportVectorMachineIr, values: &[Value]) -> Value {
    if svm.vector_fields.is_empty() || svm.support_vectors.is_empty() {
        return Value::Missing;
    }

    // Build input vector for vector_fields
    let mut input_vec: Vec<f64> = Vec::new();
    for &fid in &svm.vector_fields {
        let idx = fid.as_usize();
        let v = if idx < values.len() {
            values[idx]
        } else {
            Value::Missing
        };
        match v {
            Value::Continuous(f) => input_vec.push(f),
            Value::Missing => return Value::Missing,
            Value::Discrete(_sid) => {
                // For categorical via vector, try to parse? For now 0
                // This shouldn't happen for continuous vector fields
                return Value::Missing;
            }
        }
    }

    // Build map from vectorId to array
    let mut vec_map: HashMap<String, &[f64]> = HashMap::new();
    for (id, arr) in &svm.vector_instances {
        vec_map.insert(id.clone(), arr.as_slice());
    }

    let gamma = svm.kernel_gamma;
    let mut sum = svm.absolute_value;

    for (i, sv_id) in svm.support_vectors.iter().enumerate() {
        if i >= svm.coefficients.len() {
            break;
        }
        let coeff = svm.coefficients[i];
        if let Some(sv_arr) = vec_map.get(sv_id) {
            if sv_arr.len() != input_vec.len() {
                continue;
            }
            let mut dist2: f64 = 0.0;
            for (a, b) in input_vec.iter().zip(sv_arr.iter()) {
                let diff = a - b;
                dist2 += diff * diff;
            }
            let k = (-gamma * dist2).exp();
            sum += coeff * k;
        }
    }

    Value::Continuous(sum)
}
