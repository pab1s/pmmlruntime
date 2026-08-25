//! PMML field-type enums derived from `pmml.xsd:4490`.
//!
//! All `FromStr` impls are **case-sensitive per spec** (lowercase). Parsing `"String"`
//! or `"DOUBLE"` returns `Err`. String values round-trip via [`DataType::as_str`]
//! (and `OpType`/`MiningFunction`/`ResultFeature` equivalents).
//!
//! These enums are used in `pmml-ir::FieldMeta` and `pmml-evaluator` dispatch.

use std::str::FromStr;

/// PMML `DATATYPE` (16 values per `pmml.xsd`).
///
/// See `DataDictionary/DataField/@dataType`. The two `*Since\[0\]` variants are
/// unsupported per JPMML and rejected early in `pmml-ir::lower`.
///
/// # Examples
///
/// ```
/// use pmmlruntime::base::DataType;
/// assert_eq!("double".parse::<DataType>().unwrap(), DataType::Double);
/// assert_eq!(DataType::DateDaysSince1970.as_str(), "dateDaysSince[1970]");
/// assert!("DOUBLE".parse::<DataType>().is_err()); // case-sensitive
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DataType {
    String,
    Integer,
    Float,
    Double,
    Boolean,
    Date,
    Time,
    DateTime,
    DateDaysSince0,
    DateDaysSince1960,
    DateDaysSince1970,
    DateDaysSince1980,
    TimeSeconds,
    DateTimeSecondsSince0,
    DateTimeSecondsSince1960,
    DateTimeSecondsSince1970,
    DateTimeSecondsSince1980,
}

impl FromStr for DataType {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "string" => Self::String,
            "integer" => Self::Integer,
            "float" => Self::Float,
            "double" => Self::Double,
            "boolean" => Self::Boolean,
            "date" => Self::Date,
            "time" => Self::Time,
            "dateTime" => Self::DateTime,
            "dateDaysSince[0]" => Self::DateDaysSince0,
            "dateDaysSince[1960]" => Self::DateDaysSince1960,
            "dateDaysSince[1970]" => Self::DateDaysSince1970,
            "dateDaysSince[1980]" => Self::DateDaysSince1980,
            "timeSeconds" => Self::TimeSeconds,
            "dateTimeSecondsSince[0]" => Self::DateTimeSecondsSince0,
            "dateTimeSecondsSince[1960]" => Self::DateTimeSecondsSince1960,
            "dateTimeSecondsSince[1970]" => Self::DateTimeSecondsSince1970,
            "dateTimeSecondsSince[1980]" => Self::DateTimeSecondsSince1980,
            _ => return Err(format!("unknown DATATYPE: {s}")),
        })
    }
}

impl DataType {
    /// Returns the PMML XSD string for this variant.
    ///
    /// # Examples
    ///
    /// ```
    /// use pmmlruntime::base::DataType;
    /// assert_eq!(DataType::String.as_str(), "string");
    /// ```
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::String => "string",
            Self::Integer => "integer",
            Self::Float => "float",
            Self::Double => "double",
            Self::Boolean => "boolean",
            Self::Date => "date",
            Self::Time => "time",
            Self::DateTime => "dateTime",
            Self::DateDaysSince0 => "dateDaysSince[0]",
            Self::DateDaysSince1960 => "dateDaysSince[1960]",
            Self::DateDaysSince1970 => "dateDaysSince[1970]",
            Self::DateDaysSince1980 => "dateDaysSince[1980]",
            Self::TimeSeconds => "timeSeconds",
            Self::DateTimeSecondsSince0 => "dateTimeSecondsSince[0]",
            Self::DateTimeSecondsSince1960 => "dateTimeSecondsSince[1960]",
            Self::DateTimeSecondsSince1970 => "dateTimeSecondsSince[1970]",
            Self::DateTimeSecondsSince1980 => "dateTimeSecondsSince[1980]",
        }
    }

    /// Whether this type is unsupported per JPMML.
    ///
    /// Only `dateDaysSince[0]` and `dateTimeSecondsSince[0]` are unsupported
    /// (epoch `0` has no defined calendar). All other variants return `false`.
    ///
    /// # Examples
    ///
    /// ```
    /// use pmmlruntime::base::DataType;
    /// assert!(DataType::DateDaysSince0.is_unsupported());
    /// assert!(!DataType::Double.is_unsupported());
    /// ```
    #[must_use]
    pub fn is_unsupported(self) -> bool {
        matches!(self, Self::DateDaysSince0 | Self::DateTimeSecondsSince0)
    }
}

/// PMML `OPTYPE` (3 values per `pmml.xsd`).
///
/// `categorical`/`ordinal` → [`crate::Value::Discrete`], `continuous` → [`crate::Value::Continuous`].
///
/// # Examples
///
/// ```
/// use pmmlruntime::base::OpType;
/// assert_eq!("continuous".parse::<OpType>().unwrap(), OpType::Continuous);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OpType {
    Categorical,
    Ordinal,
    Continuous,
}

impl FromStr for OpType {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "categorical" => Self::Categorical,
            "ordinal" => Self::Ordinal,
            "continuous" => Self::Continuous,
            _ => return Err(format!("unknown OPTYPE: {s}")),
        })
    }
}

impl OpType {
    /// Returns the PMML XSD string (`"categorical"`, `"ordinal"`, `"continuous"`).
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Categorical => "categorical",
            Self::Ordinal => "ordinal",
            Self::Continuous => "continuous",
        }
    }
}

/// PMML `MINING-FUNCTION` (7 values per `pmml.xsd`).
///
/// Maps to the model family; e.g., `TreeModel` typically has `classification` or `regression`.
///
/// # Examples
///
/// ```
/// use pmmlruntime::base::MiningFunction;
/// assert_eq!("classification".parse::<MiningFunction>().unwrap(), MiningFunction::Classification);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MiningFunction {
    AssociationRules,
    Sequences,
    Classification,
    Regression,
    Clustering,
    TimeSeries,
    Mixed,
}

impl FromStr for MiningFunction {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "associationRules" => Self::AssociationRules,
            "sequences" => Self::Sequences,
            "classification" => Self::Classification,
            "regression" => Self::Regression,
            "clustering" => Self::Clustering,
            "timeSeries" => Self::TimeSeries,
            "mixed" => Self::Mixed,
            _ => return Err(format!("unknown MINING-FUNCTION: {s}")),
        })
    }
}

/// PMML `RESULT-FEATURE` (26 values per spec 4.4, `OutputField/@feature`).
///
/// Four values are unsupported per JPMML and `is_unsupported` returns `true`:
/// `confidenceIntervalLower`, `confidenceIntervalUpper`, `standardError`, `standardDeviation`.
///
/// # Examples
///
/// ```
/// use pmmlruntime::base::ResultFeature;
/// assert_eq!("predictedValue".parse::<ResultFeature>().unwrap(), ResultFeature::PredictedValue);
/// assert!(ResultFeature::StandardError.is_unsupported());
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ResultFeature {
    PredictedValue,
    PredictedDisplayValue,
    TransformedValue,
    Decision,
    Probability,
    Affinity,
    Residual,
    StandardError,
    StandardDeviation,
    ClusterId,
    ClusterAffinity,
    EntityId,
    EntityAffinity,
    Warning,
    RuleValue,
    ReasonCode,
    Antecedent,
    Consequent,
    Rule,
    RuleId,
    Confidence,
    Support,
    Lift,
    Leverage,
    ConfidenceIntervalLower,
    ConfidenceIntervalUpper,
}

impl FromStr for ResultFeature {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "predictedValue" => Self::PredictedValue,
            "predictedDisplayValue" => Self::PredictedDisplayValue,
            "transformedValue" => Self::TransformedValue,
            "decision" => Self::Decision,
            "probability" => Self::Probability,
            "affinity" => Self::Affinity,
            "residual" => Self::Residual,
            "standardError" => Self::StandardError,
            "standardDeviation" => Self::StandardDeviation,
            "clusterId" => Self::ClusterId,
            "clusterAffinity" => Self::ClusterAffinity,
            "entityId" => Self::EntityId,
            "entityAffinity" => Self::EntityAffinity,
            "warning" => Self::Warning,
            "ruleValue" => Self::RuleValue,
            "reasonCode" => Self::ReasonCode,
            "antecedent" => Self::Antecedent,
            "consequent" => Self::Consequent,
            "rule" => Self::Rule,
            "ruleId" => Self::RuleId,
            "confidence" => Self::Confidence,
            "support" => Self::Support,
            "lift" => Self::Lift,
            "leverage" => Self::Leverage,
            "confidenceIntervalLower" => Self::ConfidenceIntervalLower,
            "confidenceIntervalUpper" => Self::ConfidenceIntervalUpper,
            _ => return Err(format!("unknown RESULT-FEATURE: {s}")),
        })
    }
}

impl ResultFeature {
    /// Whether this feature is unsupported per JPMML (4 of 26).
    #[must_use]
    pub fn is_unsupported(self) -> bool {
        matches!(
            self,
            Self::ConfidenceIntervalLower
                | Self::ConfidenceIntervalUpper
                | Self::StandardError
                | Self::StandardDeviation
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn datatype_roundtrip() {
        for s in ["string", "double", "dateDaysSince[1970]", "boolean"] {
            let dt: DataType = s.parse().unwrap();
            assert_eq!(dt.as_str(), s);
        }
    }

    #[test]
    fn optype_parse() {
        assert_eq!("continuous".parse::<OpType>().unwrap(), OpType::Continuous);
    }

    #[test]
    fn result_feature_unsupported() {
        assert!(ResultFeature::ConfidenceIntervalLower.is_unsupported());
        assert!(!ResultFeature::PredictedValue.is_unsupported());
    }
}
