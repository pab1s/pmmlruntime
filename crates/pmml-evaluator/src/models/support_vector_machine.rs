use pmml_core::Value;
use pmml_ir::ir::SupportVectorMachineIr;
use std::collections::HashMap;

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
            Value::Discrete(sid) => {
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
