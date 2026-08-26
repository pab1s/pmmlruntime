//! Unsupported-markup inspector — fails fast on explicitly unsupported PMML 4.4.
//!
//! Mirrors `org.jpmml.evaluator.UnsupportedMarkupInspector` (JPMML). Called as
//! `verify_raw` before lowering and `verify_ir` after lowering. Vendor
//! [`crate::ir::ExtensionIr`] is always allowed (stored, not evaluated).

use crate::base::error::{PmmlError, Result};
use crate::ir::Ir;
use crate::xml::RawPmml;

/// Prefix used for every [`PmmlError::UnsupportedMarkup`] message from this module.
const UNSUPPORTED_MSG: &str = "unsupported markup";

/// Verifies a raw PMML document before lowering.
///
/// Checks `RawPmml.unsupported_model` (populated by `pmml-xml` when the XML
/// contains `AnomalyDetectionModel`, `BaselineModel`, `BayesianNetworkModel`,
/// `GaussianProcessModel`, `SequenceModel`, `TextModel`, or `TimeSeriesModel`).
/// Vendor extensions are always allowed.
///
/// # Errors
///
/// Returns `PmmlError::UnsupportedMarkup` when `raw.unsupported_model` is `Some`,
/// with a message of the form `"unsupported markup: {feature}"`.
///
/// # Examples
///
/// ```
/// use pmmlruntime::xml::unmarshal;
/// use pmmlruntime::ir::verify_raw;
///
/// let xml = br#"<PMML version="4.4"><Header/><DataDictionary><DataField name="x" dataType="double" optype="continuous"/></DataDictionary><TreeModel functionName="classification"><MiningSchema><MiningField name="x"/></MiningSchema><Node score="a"><True/></Node></TreeModel></PMML>"#;
/// let raw = unmarshal(xml).unwrap();
/// assert!(verify_raw(&raw).is_ok());
/// ```
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
        && raw.anomaly_detection_model.is_none()
        && raw.baseline_model.is_none()
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

/// Verifies an already-lowered [`Ir`] for unsupported constructs.
///
/// Rejects JPMML-unsupported markup: `TreeModel/@missingValueStrategy`
/// `weightedConfidence`/`aggregateNodes` and `ClusteringModel/@modelClass`
/// `distributionBased` (features.md:42,51). `lower::parse_missing_strategy`
/// preserves those variants so this check can fail fast like
/// `org.jpmml.evaluator.UnsupportedMarkupInspector`.
///
/// # Errors
///
/// Returns `PmmlError::UnsupportedMarkup` for the two Tree strategies and
/// for `distributionBased` clustering. Vendor `Extension` is always `Ok`.
///
/// # Examples
///
/// ```
/// use pmmlruntime::xml::unmarshal;
/// use pmmlruntime::ir::{lower, verify_ir};
/// let xml = br#"<PMML version="4.4"><Header/><DataDictionary><DataField name="x" dataType="double" optype="continuous"/></DataDictionary><TreeModel functionName="classification"><MiningSchema><MiningField name="x"/></MiningSchema><Node score="a"><True/></Node></TreeModel></PMML>"#;
/// let ir = lower(unmarshal(xml).unwrap()).unwrap();
/// assert!(verify_ir(&ir).is_ok());
/// ```
pub fn verify_ir(ir: &Ir) -> Result<()> {
    match &ir.model {
        crate::ir::ModelIr::Tree(tree) => match tree.missing_value_strategy {
            crate::ir::MissingValueStrategy::WeightedConfidence => {
                return Err(unsupported(
                    "TreeModel/@missingValueStrategy='weightedConfidence'",
                ))
            }
            crate::ir::MissingValueStrategy::AggregateNodes => {
                return Err(unsupported(
                    "TreeModel/@missingValueStrategy='aggregateNodes'",
                ))
            }
            _ => {}
        },
        crate::ir::ModelIr::Clustering(cl) if cl.model_class == "distributionBased" => {
            return Err(unsupported(
                "ClusteringModel/@modelClass='distributionBased'",
            ));
        }
        _ => {}
    }
    Ok(())
}

/// Strict alias — same as [`verify_ir`] (kept for callers that used `verify_ir_strict`).
pub fn verify_ir_strict(ir: &Ir) -> Result<()> {
    verify_ir(ir)
}

/// Constructs a `PmmlError::UnsupportedMarkup` with the standard prefix.
///
/// # Examples
///
/// ```
/// use pmmlruntime::ir::verify::unsupported;
/// let err = unsupported("AnomalyDetectionModel");
/// assert!(err.to_string().contains("unsupported markup"));
/// ```
#[must_use]
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
