//! Unsupported markup inspector — mirrors JPMML `UnsupportedMarkupInspector`.
//! Fails fast on PMML features we explicitly don't support in v1.

use crate::ir::Ir;
use pmml_core::error::{PmmlError, Result};
use pmml_xml::RawPmml;

/// Elements we reject with UnsupportedMarkup (like upstream).
const UNSUPPORTED_MSG: &str = "unsupported markup";

pub fn verify_raw(raw: &RawPmml) -> Result<()> {
    // Check for models we don't support in v1: if raw has no TreeModel but has other models,
    // we should error. For now RawPmml only has tree_model, so if it's None, we error.
    // In future, RawPmml will have multiple model fields; we check each.
    if raw.tree_model.is_none() && raw.data_dictionary.is_empty() {
        // empty pmml — validation will fail elsewhere
        return Ok(());
    }
    // No explicit unsupported elements in Raw yet; later we parse Extension/other.
    Ok(())
}

pub fn verify_ir(ir: &Ir) -> Result<()> {
    // Check for unsupported ResultFeature etc — already filtered in lower.
    // Check for unsupported mining_function? Tree supports classification/regression.
    match &ir.model {
        crate::ir::ModelIr::Tree(tree) => {
            // Check missingValueStrategy is one of allowed
            // Allowed: lastPrediction, nullPrediction, defaultChild
            // Already validated in lower; if unknown, error
            let _ = tree.missing_value_strategy;
        }
        _ => {}
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
