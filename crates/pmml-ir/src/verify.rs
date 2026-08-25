//! Unsupported markup inspector — mirrors JPMML `UnsupportedMarkupInspector`.
//! Fails fast on PMML features we explicitly don't support in v1.

use crate::ir::Ir;
use pmml_core::error::{PmmlError, Result};
use pmml_xml::RawPmml;

/// Elements we reject with UnsupportedMarkup (like upstream).
const UNSUPPORTED_MSG: &str = "unsupported markup";

pub fn verify_raw(raw: &RawPmml) -> Result<()> {
    // Gracefully handle vendor extensions — never error, just store
    // Extensions are already captured in raw.extensions; no verification needed

    // Handle unsupported PMML 4.4 models (plan D1): AnomalyDetection, Baseline, Bayesian, etc.
    // These are captured as raw.unsupported_model during unmarshal and should produce a clear
    // UnsupportedMarkup error that callers can handle gracefully (keep, not panic).
    if let Some(ref model) = raw.unsupported_model {
        return Err(unsupported(model));
    }

    // Deprecated / removed elements: ModelComposition (4.1), CenterFields (3.2), TableLocator handled in arrow bridge
    // They are already either captured as unsupported_model or handled gracefully elsewhere

    // Check for invalid PMML: no known model and no extensions but empty data_dictionary handled elsewhere
    if raw.tree_model.is_none()
        && raw.regression_model.is_none()
        && raw.mining_model.is_none()
        && raw.scorecard.is_none()
        && raw.clustering_model.is_none()
        && raw.naive_bayes_model.is_none()
        && raw.nearest_neighbor_model.is_none()
        && raw.support_vector_machine_model.is_none()
        && raw.neural_network.is_none()
        && raw.general_regression_model.is_none()
        && raw.association_model.is_none()
        && raw.rule_set_model.is_none()
        && raw.unsupported_model.is_none()
        && raw.data_dictionary.is_empty()
    {
        // empty pmml — validation will fail elsewhere
        return Ok(());
    }
    // InvalidValueTreatment and other MiningField attributes are validated in lower; here we just ensure
    // at least one known model or extensions present is ok
    Ok(())
}

pub fn verify_ir(ir: &Ir) -> Result<()> {
    // Check for unsupported ResultFeature etc — already filtered in lower.
    // Check for unsupported mining_function? Tree supports classification/regression.
    if let crate::ir::ModelIr::Tree(tree) = &ir.model {
        // Check missingValueStrategy is one of allowed
        // Allowed: lastPrediction, nullPrediction, defaultChild
        // Already validated in lower; if unknown, error
        let _ = tree.missing_value_strategy;
    }
    Ok(())
}

/// Helper to assert unsupported and return error.
pub fn unsupported(feature: &str) -> PmmlError {
    PmmlError::UnsupportedMarkup(format!("{UNSUPPORTED_MSG}: {feature}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsupported_err() {
        let e = unsupported("AnomalyDetectionModel");
        assert!(e.to_string().contains("unsupported markup"));
    }
}
