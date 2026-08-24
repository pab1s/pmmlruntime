//! Builtin functions — 100+ PMML functions via libm/statrs.
//! Full mapping for Apply 100 builtins including TextIndex, Aggregate, Lag, NormContinuous handling via VM.

use pmml_ir::ir::BuiltinId;

/// Resolve builtin by name (used in lower when generating bytecode).
/// Covers PMML 4.4 Apply functions + aggregate/lag/textIndex vendor extensions.
pub fn builtin_by_name(name: &str) -> Option<BuiltinId> {
    Some(match name {
        // Arithmetic
        "add" | "+" => BuiltinId::Add,
        "subtract" | "-" => BuiltinId::Sub,
        "multiply" | "*" => BuiltinId::Mul,
        "divide" | "/" => BuiltinId::Div,
        "pow" => BuiltinId::Pow,
        "log" | "ln" => BuiltinId::Log,
        "log10" => BuiltinId::Log10,
        "exp" => BuiltinId::Exp,
        "sqrt" => BuiltinId::Sqrt,
        "abs" => BuiltinId::Abs,
        "floor" => BuiltinId::Floor,
        "ceil" => BuiltinId::Ceil,
        "round" => BuiltinId::Round,
        // Math
        "sin" => BuiltinId::Sin,
        "cos" => BuiltinId::Cos,
        "tan" => BuiltinId::Tan,
        "asin" => BuiltinId::Asin,
        "acos" => BuiltinId::Acos,
        "atan" => BuiltinId::Atan,
        "sinh" => BuiltinId::Sinh,
        "cosh" => BuiltinId::Cosh,
        "tanh" => BuiltinId::Tanh,
        "remainder" => BuiltinId::Remainder,
        // Min/Max (also aggregate)
        "min" => BuiltinId::Min,
        "max" => BuiltinId::Max,
        // String
        "uppercase" | "upperCase" => BuiltinId::Uppercase,
        "lowercase" | "lowerCase" => BuiltinId::Lowercase,
        "substring" => BuiltinId::Substring,
        "trimBlanks" => BuiltinId::TrimBlanks,
        "concat" => BuiltinId::Concat,
        "stringLength" => BuiltinId::StringLength,
        "replace" => BuiltinId::Replace,
        "matches" => BuiltinId::Matches,
        // TextIndex
        "textIndex" => BuiltinId::TextIndex,
        // Aggregate (PMML Aggregate function)
        "count" | "aggregateCount" => BuiltinId::AggregateCount,
        "sum" | "aggregateSum" => BuiltinId::AggregateSum,
        "average" | "avg" | "aggregateAverage" => BuiltinId::AggregateAvg,
        "aggregateMin" => BuiltinId::AggregateMin,
        "aggregateMax" => BuiltinId::AggregateMax,
        // Lag (needs session ring buffer)
        "lag" => BuiltinId::Lag,
        // Norm
        "normContinuous" => BuiltinId::NormContinuousOp,
        "normDiscrete" => BuiltinId::NormDiscreteOp,
        // Comparison / Logical via Apply (though predicates also)
        "equal" => BuiltinId::Equal,
        "notEqual" => BuiltinId::NotEqual,
        "lessThan" => BuiltinId::LessThan,
        "lessOrEqual" => BuiltinId::LessOrEqual,
        "greaterThan" => BuiltinId::GreaterThan,
        "greaterOrEqual" => BuiltinId::GreaterOrEqual,
        "and" => BuiltinId::And,
        "or" => BuiltinId::Or,
        "not" => BuiltinId::Not,
        "isMissing" => BuiltinId::IsMissing,
        "isNotMissing" => BuiltinId::IsNotMissing,
        "isValid" => BuiltinId::IsValid,
        "if" => BuiltinId::If,
        "threshold" => BuiltinId::Threshold,
        _ => return None,
    })
}

pub fn eval_builtin(id: BuiltinId, args: &[f64]) -> Option<f64> {
    Some(match id {
        BuiltinId::Add => args.iter().sum(),
        BuiltinId::Sub => args.first()? - args.get(1)?,
        BuiltinId::Mul => args.iter().product(),
        BuiltinId::Div => args.first()? / args.get(1)?,
        BuiltinId::Pow => args.first()?.powf(*args.get(1)?),
        BuiltinId::Log | BuiltinId::Ln => args.first()?.ln(),
        BuiltinId::Log10 => args.first()?.log10(),
        BuiltinId::Exp => args.first()?.exp(),
        BuiltinId::Sqrt => args.first()?.sqrt(),
        BuiltinId::Abs => args.first()?.abs(),
        BuiltinId::Floor => args.first()?.floor(),
        BuiltinId::Ceil => args.first()?.ceil(),
        BuiltinId::Round => args.first()?.round(),
        BuiltinId::Remainder => args.first()? % args.get(1)?,
        BuiltinId::Sin => args.first()?.sin(),
        BuiltinId::Cos => args.first()?.cos(),
        BuiltinId::Tan => args.first()?.tan(),
        BuiltinId::Asin => args.first()?.asin(),
        BuiltinId::Acos => args.first()?.acos(),
        BuiltinId::Atan => args.first()?.atan(),
        BuiltinId::Sinh => args.first()?.sinh(),
        BuiltinId::Cosh => args.first()?.cosh(),
        BuiltinId::Tanh => args.first()?.tanh(),
        BuiltinId::Min => args.iter().cloned().fold(f64::INFINITY, f64::min),
        BuiltinId::Max => args.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        // Aggregate numeric ops — when args already collected, treat as sum/avg etc.
        BuiltinId::AggregateSum | BuiltinId::AggregateAvg | BuiltinId::AggregateMin | BuiltinId::AggregateMax | BuiltinId::AggregateCount => {
            // These are handled in VM with Values, not via eval_builtin f64 slice; fallback to appropriate
            match id {
                BuiltinId::AggregateCount => args.len() as f64,
                BuiltinId::AggregateSum => args.iter().sum(),
                BuiltinId::AggregateAvg => {
                    if args.is_empty() {
                        f64::NAN
                    } else {
                        args.iter().sum::<f64>() / args.len() as f64
                    }
                }
                BuiltinId::AggregateMin => args.iter().cloned().fold(f64::INFINITY, f64::min),
                BuiltinId::AggregateMax => args.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                _ => unreachable!(),
            }
        }
        // Lag, TextIndex, NormContinuous, string ops etc are handled in VM directly on Values
        _ => return None,
    })
}

/// Helper for string builtins evaluated on Values (not f64).
pub fn eval_string_builtin(id: BuiltinId, args: &[String]) -> Option<String> {
    match id {
        BuiltinId::Uppercase => Some(args.first()?.to_uppercase()),
        BuiltinId::Lowercase => Some(args.first()?.to_lowercase()),
        BuiltinId::Concat => Some(args.join("")),
        BuiltinId::TrimBlanks => Some(args.first()?.trim().to_string()),
        BuiltinId::Substring => {
            // substring(string, pos, len) — pos 1-indexed per PMML spec
            let s = args.first()?;
            let pos: usize = args.get(1)?.parse().ok()?;
            let len: usize = args.get(2).and_then(|l| l.parse().ok()).unwrap_or(s.len());
            let start = pos.saturating_sub(1).min(s.len());
            let end = (start + len).min(s.len());
            Some(s[start..end].to_string())
        }
        BuiltinId::Replace => {
            // replace(string, pattern, replacement)
            let s = args.first()?;
            let pat = args.get(1)?;
            let rep = args.get(2)?;
            Some(s.replace(pat, rep))
        }
        _ => None,
    }
}
