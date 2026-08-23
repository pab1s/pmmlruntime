//! Builtin functions — 100+ PMML functions via libm/statrs.
//! v1 stub: maps BuiltinId -> f64 operation, not yet wired to VM.

use pmml_ir::ir::BuiltinId;

/// Resolve builtin by name (used in lower when generating bytecode).
pub fn builtin_by_name(name: &str) -> Option<BuiltinId> {
    Some(match name {
        "add" | "+" => BuiltinId::Add,
        "subtract" | "-" => BuiltinId::Sub,
        "multiply" | "*" => BuiltinId::Mul,
        "divide" | "/" => BuiltinId::Div,
        "pow" => BuiltinId::Pow,
        "log" | "ln" => BuiltinId::Log,
        "exp" => BuiltinId::Exp,
        "sqrt" => BuiltinId::Sqrt,
        "abs" => BuiltinId::Abs,
        "min" => BuiltinId::Min,
        "max" => BuiltinId::Max,
        _ => return None,
    })
}

pub fn eval_builtin(id: BuiltinId, args: &[f64]) -> Option<f64> {
    Some(match id {
        BuiltinId::Add => args.iter().sum(),
        BuiltinId::Sub => args.get(0)? - args.get(1)?,
        BuiltinId::Mul => args.iter().product(),
        BuiltinId::Div => args.get(0)? / args.get(1)?,
        BuiltinId::Pow => args.get(0)?.powf(*args.get(1)?),
        BuiltinId::Log => args.get(0)?.ln(),
        BuiltinId::Exp => args.get(0)?.exp(),
        BuiltinId::Sqrt => args.get(0)?.sqrt(),
        BuiltinId::Abs => args.get(0)?.abs(),
        BuiltinId::Min => args.iter().cloned().fold(f64::INFINITY, f64::min),
        BuiltinId::Max => args.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        _ => return None,
    })
}
