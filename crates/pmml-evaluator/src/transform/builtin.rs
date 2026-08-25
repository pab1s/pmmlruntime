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
        "cbrt" => BuiltinId::Cbrt,
        "sign" => BuiltinId::Sign,
        "remainder" => BuiltinId::Remainder,
        "modulo" => BuiltinId::Modulo,
        // Math trig
        "sin" => BuiltinId::Sin,
        "cos" => BuiltinId::Cos,
        "tan" => BuiltinId::Tan,
        "asin" => BuiltinId::Asin,
        "acos" => BuiltinId::Acos,
        "atan" => BuiltinId::Atan,
        "sinh" => BuiltinId::Sinh,
        "cosh" => BuiltinId::Cosh,
        "tanh" => BuiltinId::Tanh,
        "atan2" | "x-atan2" => BuiltinId::Atan2,
        "hypot" => BuiltinId::Hypot,
        // Rounding / exp
        "rint" => BuiltinId::Rint,
        "expm1" => BuiltinId::Expm1,
        "ln1p" | "log1p" => BuiltinId::Ln1p,
        // Statistical aggregates
        "min" => BuiltinId::Min,
        "max" => BuiltinId::Max,
        "median" => BuiltinId::Median,
        "product" => BuiltinId::ProductOp,
        "sum" => BuiltinId::SumOp,
        "avg" | "average" | "mean" => BuiltinId::AvgOp,
        "stddev" => BuiltinId::StdDev,
        "variance" => BuiltinId::Variance,
        // String
        "uppercase" | "upperCase" => BuiltinId::Uppercase,
        "lowercase" | "lowerCase" => BuiltinId::Lowercase,
        "substring" => BuiltinId::Substring,
        "trimBlanks" => BuiltinId::TrimBlanks,
        "normalizeSpace" => BuiltinId::NormalizeSpace,
        "concat" => BuiltinId::Concat,
        "stringLength" => BuiltinId::StringLength,
        "replace" => BuiltinId::Replace,
        "matches" => BuiltinId::Matches,
        "formatNumber" | "format_number" => BuiltinId::FormatNumber,
        "formatDatetime" | "formatDateTime" => BuiltinId::FormatDatetime,
        // Date/time (chrono)
        "dateDaysSinceYear" => BuiltinId::DateDaysSinceYear,
        "dateSecondsSinceYear" => BuiltinId::DateSecondsSinceYear,
        "dateSecondsSinceMidnight" => BuiltinId::DateSecondsSinceMidnight,
        "dateDaysSince1960" => BuiltinId::DateDaysSince1960,
        "dateDaysSince1970" => BuiltinId::DateDaysSince1970,
        "dateDaysSince1980" => BuiltinId::DateDaysSince1980,
        "dateTimeSecondsSince1960" => BuiltinId::DateTimeSecondsSince1960,
        "dateTimeSecondsSince1970" => BuiltinId::DateTimeSecondsSince1970,
        "dateTimeSecondsSince1980" => BuiltinId::DateTimeSecondsSince1980,
        "dateTimeSecondsSince0" => BuiltinId::DateTimeSecondsSince0,
        "timeSeconds" => BuiltinId::TimeSeconds,
        // Distribution (statrs / libm)
        "normalCDF" => BuiltinId::NormalCdf,
        "normalPDF" => BuiltinId::NormalPdf,
        "normalIDF" => BuiltinId::NormalIdf,
        "stdNormalCDF" | "stdNormalCdf" => BuiltinId::StdNormalCdf,
        "stdNormalPDF" | "stdNormalPdf" => BuiltinId::StdNormalPdf,
        "stdNormalIDF" | "stdNormalIdf" => BuiltinId::StdNormalIdf,
        "erf" => BuiltinId::ErfOp,
        // TextIndex
        "textIndex" => BuiltinId::TextIndex,
        // Aggregate (PMML Aggregate function)
        "count" | "aggregateCount" => BuiltinId::AggregateCount,
        "aggregateSum" => BuiltinId::AggregateSum,
        "aggregateAverage" => BuiltinId::AggregateAvg,
        "aggregateMin" => BuiltinId::AggregateMin,
        "aggregateMax" => BuiltinId::AggregateMax,
        "multiset" | "aggregateMultiset" => BuiltinId::AggregateMultiset,
        // Lag (needs session ring buffer)
        "lag" => BuiltinId::Lag,
        // Norm
        "normContinuous" => BuiltinId::NormContinuousOp,
        "normDiscrete" => BuiltinId::NormDiscreteOp,
        // Comparison / Logical
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
        "isNotValid" => BuiltinId::IsNotValid,
        "isIn" => BuiltinId::IsIn,
        "isNotIn" => BuiltinId::IsNotIn,
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
        BuiltinId::Cbrt => args.first()?.cbrt(),
        BuiltinId::Sign => {
            let v = *args.first()?;
            if v > 0.0 {
                1.0
            } else if v < 0.0 {
                -1.0
            } else {
                0.0
            }
        }
        BuiltinId::Remainder => args.first()? % args.get(1)?,
        BuiltinId::Modulo => {
            let a = *args.first()?;
            let b = *args.get(1)?;
            if b == 0.0 {
                f64::NAN
            } else {
                a - (a / b).floor() * b
            }
        }
        BuiltinId::Rint => libm::rint(*args.first()?),
        BuiltinId::Expm1 => libm::expm1(*args.first()?),
        BuiltinId::Ln1p => libm::log1p(*args.first()?),
        BuiltinId::Hypot => libm::hypot(*args.first()?, *args.get(1)?),
        BuiltinId::Atan2 => args.first()?.atan2(*args.get(1)?),
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
        BuiltinId::Median => {
            if args.is_empty() {
                return None;
            }
            let mut v = args.to_vec();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mid = v.len() / 2;
            if v.len() % 2 == 1 {
                v[mid]
            } else {
                (v[mid - 1] + v[mid]) / 2.0
            }
        }
        BuiltinId::ProductOp => args.iter().product(),
        BuiltinId::SumOp => args.iter().sum(),
        BuiltinId::AvgOp | BuiltinId::Mean => {
            if args.is_empty() {
                return None;
            }
            args.iter().sum::<f64>() / args.len() as f64
        }
        BuiltinId::StdDev | BuiltinId::Variance => {
            if args.len() < 2 {
                return None;
            }
            let mean = args.iter().sum::<f64>() / args.len() as f64;
            let var = args.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / args.len() as f64;
            if id == BuiltinId::Variance {
                var
            } else {
                var.sqrt()
            }
        }
        BuiltinId::ErfOp => libm::erf(*args.first()?),
        BuiltinId::NormalCdf
        | BuiltinId::NormalPdf
        | BuiltinId::NormalIdf
        | BuiltinId::StdNormalCdf
        | BuiltinId::StdNormalPdf
        | BuiltinId::StdNormalIdf => return None,
        BuiltinId::DateDaysSinceYear
        | BuiltinId::DateSecondsSinceYear
        | BuiltinId::DateSecondsSinceMidnight
        | BuiltinId::DateDaysSince1960
        | BuiltinId::DateDaysSince1970
        | BuiltinId::DateDaysSince1980
        | BuiltinId::DateTimeSecondsSince1960
        | BuiltinId::DateTimeSecondsSince1970
        | BuiltinId::DateTimeSecondsSince1980
        | BuiltinId::DateTimeSecondsSince0
        | BuiltinId::TimeSeconds => return None,
        BuiltinId::FormatNumber | BuiltinId::FormatDatetime => return None,
        BuiltinId::AggregateSum
        | BuiltinId::AggregateAvg
        | BuiltinId::AggregateMin
        | BuiltinId::AggregateMax
        | BuiltinId::AggregateCount
        | BuiltinId::AggregateMultiset => match id {
            BuiltinId::AggregateCount | BuiltinId::AggregateMultiset => args.len() as f64,
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
        },
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
        BuiltinId::NormalizeSpace => {
            let s = args.first()?;
            let normalized = s.split_whitespace().collect::<Vec<_>>().join(" ");
            Some(normalized)
        }
        BuiltinId::Substring => {
            let s = args.first()?;
            let pos: usize = args.get(1)?.parse().ok()?;
            let len: usize = args.get(2).and_then(|l| l.parse().ok()).unwrap_or(s.len());
            let start = pos.saturating_sub(1).min(s.len());
            let end = (start + len).min(s.len());
            Some(s[start..end].to_string())
        }
        BuiltinId::Replace => {
            let s = args.first()?;
            let pat = args.get(1)?;
            let rep = args.get(2)?;
            let rep_java = rep.replace("$$", "\\$");
            match regex::Regex::new(pat) {
                Ok(re) => Some(re.replace_all(s, rep_java.as_str()).into_owned()),
                Err(_) => Some(s.replace(pat, rep)),
            }
        }
        BuiltinId::FormatNumber => {
            let num_str = args.first()?;
            let pat = args.get(1)?;
            let num: f64 = num_str.parse().ok()?;
            if pat.contains('d') {
                let n = num as i64;
                let width: usize = pat
                    .chars()
                    .filter(|c| c.is_ascii_digit())
                    .collect::<String>()
                    .parse()
                    .unwrap_or(0);
                if width > 0 {
                    Some(format!("{:>width$}", n, width = width))
                } else {
                    Some(format!("{}", n))
                }
            } else if pat.contains('f') {
                Some(format!("{}", num))
            } else {
                Some(num.to_string())
            }
        }
        BuiltinId::FormatDatetime => {
            let dt_str = args.first()?;
            let pat = args.get(1)?;
            let formatted = if let Ok(date) = chrono::NaiveDate::parse_from_str(dt_str, "%Y-%m-%d")
            {
                let chrono_pat = pat
                    .replace("%m", "%m")
                    .replace("%d", "%d")
                    .replace("%y", "%y")
                    .replace("%Y", "%Y");
                Some(date.format(&chrono_pat).to_string())
            } else if let Ok(dt) =
                chrono::NaiveDateTime::parse_from_str(dt_str, "%Y-%m-%d %H:%M:%S")
            {
                Some(dt.format(pat).to_string())
            } else {
                Some(dt_str.clone())
            };
            formatted
        }
        _ => None,
    }
}
