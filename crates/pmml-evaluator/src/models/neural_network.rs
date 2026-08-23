use pmml_core::Value;
use pmml_ir::ir::{NeuralNetworkIr, NeuronIr};

fn activation(func: &str, x: f64) -> f64 {
    match func.to_lowercase().as_str() {
        "logistic" | "sigmoid" => 1.0 / (1.0 + (-x).exp()),
        "tanh" => x.tanh(),
        "identity" | "linear" => x,
        "exponential" => x.exp(),
        "square" => x * x,
        "squareroot" => x.sqrt(),
        "sine" => x.sin(),
        "cosine" => x.cos(),
        "elliot" => 0.5 * (x / (1.0 + x.abs())) + 0.5,
        "arctan" => x.atan(),
        "radialbasis" => (-x * x).exp(),
        _ => 1.0 / (1.0 + (-x).exp()), // default logistic
    }
}

pub fn evaluate_neural_network(nn: &NeuralNetworkIr, values: &[Value]) -> Value {
    if nn.neural_layers.is_empty() || nn.neural_inputs.is_empty() {
        return Value::Missing;
    }

    // Map from id to computed value
    let mut computed: std::collections::HashMap<String, f64> = std::collections::HashMap::new();

    // First, set input values from neural_inputs
    for input in &nn.neural_inputs {
        let fid = input.field.as_usize();
        let v = if fid < values.len() {
            values[fid]
        } else {
            Value::Missing
        };
        let f = match v {
            Value::Continuous(x) => x,
            Value::Discrete(_) => 0.0,
            Value::Missing => 0.0,
        };
        computed.insert(input.id.clone(), f);
    }

    let mut last_layer_outputs: Vec<f64> = Vec::new();

    for layer in &nn.neural_layers {
        let mut next_outputs = Vec::new();
        for neuron in &layer.neurons {
            let mut sum = neuron.bias;
            for (from, weight) in &neuron.cons {
                if let Some(&val) = computed.get(from) {
                    sum += val * weight;
                } else {
                    // If from is not yet computed, it might be an input that wasn't in neural_inputs? Try to find by field?
                    // For v1, we assume all cons from are previously computed ids
                    sum += 0.0;
                }
            }
            let act = activation(&layer.activation_function, sum);
            computed.insert(neuron.id.clone(), act);
            next_outputs.push(act);
        }
        last_layer_outputs = next_outputs;
    }

    // Output is last layer's first neuron (for regression) or the max for classification?
    // For our simple fixture, last layer has 1 neuron with identity, so output is that neuron's value
    if let Some(&out) = last_layer_outputs.first() {
        // Check functionName: if regression, return continuous; if classification, need to map to discrete?
        // For v1, we return Continuous
        Value::Continuous(out)
    } else {
        Value::Missing
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmml_core::{FieldId, Value};
    use pmml_ir::ir::*;

    #[test]
    fn simple_nn() {
        let f1 = FieldId(0);
        let f2 = FieldId(1);
        let nn = NeuralNetworkIr {
            function_name: "regression".into(),
            mining_schema: MiningSchemaIr {
                active_fields: vec![f1, f2],
                target_field: None,
                field_metas: vec![],
                missing_value_replacement: None,
            },
            output: vec![],
            neural_inputs: vec![
                NeuralInputIr {
                    id: "0".into(),
                    field: f1,
                },
                NeuralInputIr {
                    id: "1".into(),
                    field: f2,
                },
            ],
            neural_layers: vec![
                NeuralLayerIr {
                    number_of_neurons: 2,
                    activation_function: "logistic".into(),
                    neurons: vec![
                        NeuronIr {
                            id: "hidden1".into(),
                            bias: 0.0,
                            cons: vec![("0".into(), 1.0), ("1".into(), 1.0)],
                        },
                        NeuronIr {
                            id: "hidden2".into(),
                            bias: 0.0,
                            cons: vec![("0".into(), 0.5), ("1".into(), 0.5)],
                        },
                    ],
                },
                NeuralLayerIr {
                    number_of_neurons: 1,
                    activation_function: "identity".into(),
                    neurons: vec![NeuronIr {
                        id: "output".into(),
                        bias: 0.0,
                        cons: vec![("hidden1".into(), 1.0), ("hidden2".into(), 1.0)],
                    }],
                },
            ],
            activation_function: "logistic".into(),
        };
        let vals = vec![Value::Continuous(0.5), Value::Continuous(0.5)];
        let out = evaluate_neural_network(&nn, &vals);
        match out {
            Value::Continuous(f) => {
                // hidden1: logistic(1.0) ~ 0.731, hidden2: logistic(0.5) ~0.622, output 1.353
                assert!((f - 1.353).abs() < 0.01);
            }
            _ => panic!("expected continuous"),
        }
    }
}
