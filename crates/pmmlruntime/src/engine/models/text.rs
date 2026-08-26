//! TextModel evaluation — TF-IDF / bag-of-words similarity.
//!
//! Implements `TextModel` scoring per `pmml.xsd:4071-4183`.
//! The model holds a `TextDictionary` (terms), `TextCorpus` (documents), and a
//! `DocumentTermMatrix` (rows = documents, cols = terms). Scoring tokenizes the
//! active text input, builds a TF vector, applies `TextModelNormalization`
//! (`localTermWeights`, `globalTermWeights`, `documentNormalization`), then
//! computes similarity (`cosine` or `euclidean`) against each document row.
//! The best-matching document's `id` is returned as `Discrete` (or nearest for
//! euclidean). `Missing` propagates when input missing or dictionary empty.
//!
//! Tokenization is simple: lowercase + split on non-alphanumeric, trim empties.
//! This mirrors a lightweight variant of JPMML's `TextUtil`.
//!
//! # What belongs here
//!
//! - [`evaluate_text`] — public entry point `(&TextIr, &[Value], &HashMap<SymbolId,String>) -> Value`.

use crate::base::{SymbolId, Value};
use crate::ir::TextIr;
use std::collections::HashMap;

/// Evaluate a [`TextIr`] against dense `values`.
///
/// # Parameters
///
/// - `model`: lowered text model
/// - `values`: dense values indexed by `FieldId`
/// - `symbol_names`: optional map to decode `Discrete` input strings (needed when input is `Discrete(SymbolId)`).
///
/// For most PMML, the active text field is `DataType::String` and the input arrives as `Discrete(SymbolId)`.
/// If `symbol_names` is `None`, discrete inputs are treated as missing.
///
/// # Returns
///
/// `Discrete(symbol)` of the best document `id`, or `Missing`.
pub fn evaluate_text(
    model: &TextIr,
    values: &[Value],
    symbol_names: Option<&HashMap<SymbolId, String>>,
) -> Value {
    if model.dictionary.is_empty()
        || model.corpus.is_empty()
        || model.document_term_matrix.is_empty()
    {
        return Value::Missing;
    }
    // Find active text field(s). Typically one. Use first active.
    let text_field = if let Some(fid) = model.mining_schema.active_fields.first() {
        *fid
    } else if let Some(fid) = model.mining_schema.target_field {
        // fallback if no active but target is text? Not typical.
        fid
    } else {
        return Value::Missing;
    };
    let idx = text_field.as_usize();
    let raw_val = if idx < values.len() {
        values[idx]
    } else {
        Value::Missing
    };
    let text = match raw_val {
        Value::Continuous(f) => {
            // numeric text? Not expected; convert to string
            f.to_string()
        }
        Value::Discrete(sid) => {
            if let Some(map) = symbol_names {
                if let Some(s) = map.get(&sid) {
                    s.clone()
                } else {
                    return Value::Missing;
                }
            } else {
                // Without symbol map, we cannot decode discrete; try fallback: value missing -> we can't decode
                // For tests that directly use Value::Discrete with known string via interner, the session layer
                // will pass symbol_names; direct calls without map will fallback to Missing.
                // To make direct unit tests work, treat discrete sid's debug as not allowed, so return Missing.
                return Value::Missing;
            }
        }
        Value::Missing => return Value::Missing,
    };
    // Tokenize input
    let tokens = tokenize(&text);
    if tokens.is_empty() {
        return Value::Missing;
    }
    // Build term vector for input
    let input_vec = build_term_vector(
        &tokens,
        &model.dictionary,
        model.normalization.as_ref(),
        &model.document_term_matrix,
    );
    // Optionally normalize input vector for cosine
    let doc_norm = model
        .normalization
        .as_ref()
        .map(|n| n.document_normalization.as_str())
        .unwrap_or("none");
    let sim_type = model
        .similarity
        .as_ref()
        .map(|s| s.similarity_type.as_str())
        .unwrap_or("cosine");
    let input_normed = if doc_norm == "cosine" {
        normalize_cosine(&input_vec)
    } else {
        input_vec.clone()
    };
    // Precompute IDF if needed for input? Already applied inside build_term_vector.
    // For documents, matrix is assumed already weighted? But we apply same document normalization on the fly.
    // For fair comparison, we should optionally also L2-normalize document rows if cosine.
    let mut best_idx: Option<usize> = None;
    let mut best_score = if sim_type == "euclidean" {
        f64::INFINITY
    } else {
        f64::NEG_INFINITY
    };
    for (doc_idx, doc_row) in model.document_term_matrix.iter().enumerate() {
        if doc_row.len() != model.dictionary.len() {
            // dimension mismatch skip
            continue;
        }
        // Optionally normalize doc row as well if cosine documentNormalization is cosine (PMML says documentNormalization applies to matrix?)
        // We'll treat document rows as already representative; but also normalize if requested to make comparable.
        let doc_vec = if doc_norm == "cosine" {
            normalize_cosine(doc_row)
        } else {
            doc_row.clone()
        };
        let score = if sim_type == "euclidean" {
            euclidean_distance(&input_normed, &doc_vec)
        } else {
            cosine_similarity(&input_normed, &doc_vec)
        };
        if sim_type == "euclidean" {
            if score < best_score {
                best_score = score;
                best_idx = Some(doc_idx);
            }
        } else if score > best_score {
            best_score = score;
            best_idx = Some(doc_idx);
        }
    }
    if let Some(idx) = best_idx {
        if idx < model.corpus.len() {
            let sid = model.corpus[idx].id_symbol;
            return Value::Discrete(sid);
        }
        // fallback
        return Value::Missing;
    }
    Value::Missing
}

/// Return best document index for a given input (helper for Session to map to real SymbolId).
pub fn best_document_index(
    model: &TextIr,
    values: &[Value],
    symbol_names: Option<&HashMap<SymbolId, String>>,
) -> Option<usize> {
    if model.dictionary.is_empty()
        || model.corpus.is_empty()
        || model.document_term_matrix.is_empty()
    {
        return None;
    }
    let text_field = model.mining_schema.active_fields.first().copied()?;
    let idx = text_field.as_usize();
    let raw_val = if idx < values.len() {
        values[idx]
    } else {
        Value::Missing
    };
    let text = match raw_val {
        Value::Continuous(f) => f.to_string(),
        Value::Discrete(sid) => {
            let map = symbol_names?;
            map.get(&sid)?.clone()
        }
        Value::Missing => return None,
    };
    let tokens = tokenize(&text);
    if tokens.is_empty() {
        return None;
    }
    let input_vec = build_term_vector(
        &tokens,
        &model.dictionary,
        model.normalization.as_ref(),
        &model.document_term_matrix,
    );
    let doc_norm = model
        .normalization
        .as_ref()
        .map(|n| n.document_normalization.as_str())
        .unwrap_or("none");
    let sim_type = model
        .similarity
        .as_ref()
        .map(|s| s.similarity_type.as_str())
        .unwrap_or("cosine");
    let input_normed = if doc_norm == "cosine" {
        normalize_cosine(&input_vec)
    } else {
        input_vec.clone()
    };
    let mut best_idx: Option<usize> = None;
    let mut best_score = if sim_type == "euclidean" {
        f64::INFINITY
    } else {
        f64::NEG_INFINITY
    };
    for (doc_idx, doc_row) in model.document_term_matrix.iter().enumerate() {
        if doc_row.len() != model.dictionary.len() {
            continue;
        }
        let doc_vec = if doc_norm == "cosine" {
            normalize_cosine(doc_row)
        } else {
            doc_row.clone()
        };
        let score = if sim_type == "euclidean" {
            euclidean_distance(&input_normed, &doc_vec)
        } else {
            cosine_similarity(&input_normed, &doc_vec)
        };
        if sim_type == "euclidean" {
            if score < best_score {
                best_score = score;
                best_idx = Some(doc_idx);
            }
        } else if score > best_score {
            best_score = score;
            best_idx = Some(doc_idx);
        }
    }
    best_idx
}

fn tokenize(text: &str) -> Vec<String> {
    // Lowercase, split on non-alphanumeric, filter empty, keep as is
    let lower = text.to_lowercase();
    let mut tokens: Vec<String> = Vec::new();
    let mut cur = String::new();
    for ch in lower.chars() {
        if ch.is_alphanumeric() {
            cur.push(ch);
        } else if !cur.is_empty() {
            tokens.push(cur.clone());
            cur.clear();
        }
    }
    if !cur.is_empty() {
        tokens.push(cur);
    }
    tokens
}

fn build_term_vector(
    tokens: &[String],
    dictionary: &[String],
    normalization: Option<&crate::ir::TextNormalizationIr>,
    document_term_matrix: &[Vec<f64>],
) -> Vec<f64> {
    let dict_lower: Vec<String> = dictionary.iter().map(|s| s.to_lowercase()).collect();
    let mut tf: Vec<f64> = vec![0.0; dictionary.len()];
    let mut counts: HashMap<String, usize> = HashMap::new();
    for tok in tokens {
        *counts.entry(tok.clone()).or_insert(0) += 1;
    }
    // Fill tf per dict term
    for (i, term_lower) in dict_lower.iter().enumerate() {
        if let Some(&c) = counts.get(term_lower) {
            tf[i] = c as f64;
        }
    }
    // local term weights
    let local = normalization
        .map(|n| n.local_term_weights.as_str())
        .unwrap_or("termFrequency");
    let mut local_vec = tf.clone();
    match local {
        "binary" => {
            for v in &mut local_vec {
                if *v > 0.0 {
                    *v = 1.0;
                }
            }
        }
        "logarithmic" => {
            for v in &mut local_vec {
                if *v > 0.0 {
                    *v = 1.0 + (*v).ln();
                }
            }
        }
        "augmentedNormalizedTermFrequency" => {
            let max_tf = tf.iter().copied().fold(0.0, f64::max);
            if max_tf > 0.0 {
                for v in &mut local_vec {
                    if *v > 0.0 {
                        *v = 0.5 + 0.5 * *v / max_tf;
                    }
                }
            }
        }
        _ => {} // termFrequency keep as is
    }
    // global term weights
    let global = normalization
        .map(|n| n.global_term_weights.as_str())
        .unwrap_or("inverseDocumentFrequency");
    if global != "none" && !document_term_matrix.is_empty() {
        // compute df per term: number of docs where term >0
        let mut df = vec![0usize; dictionary.len()];
        for row in document_term_matrix {
            for (i, &val) in row.iter().enumerate().take(dictionary.len()) {
                if val > 0.0 {
                    df[i] += 1;
                }
            }
        }
        let n_docs = document_term_matrix.len() as f64;
        for (i, w) in local_vec.iter_mut().enumerate() {
            if *w == 0.0 {
                continue;
            }
            let idf = match global {
                "inverseDocumentFrequency" => {
                    if df[i] == 0 {
                        0.0
                    } else {
                        (n_docs / df[i] as f64).ln()
                    }
                }
                "GFIDF" => {
                    // GFIDF = sum tf / df
                    let sum_tf: f64 = document_term_matrix
                        .iter()
                        .map(|row| row.get(i).copied().unwrap_or(0.0))
                        .sum();
                    if df[i] == 0 || sum_tf == 0.0 {
                        0.0
                    } else {
                        sum_tf / df[i] as f64
                    }
                }
                "normal" => {
                    let sum_sq: f64 = document_term_matrix
                        .iter()
                        .map(|row| {
                            let v = row.get(i).copied().unwrap_or(0.0);
                            v * v
                        })
                        .sum();
                    if sum_sq == 0.0 {
                        0.0
                    } else {
                        1.0 / sum_sq.sqrt()
                    }
                }
                "probabilisticInverse" => {
                    if df[i] == 0 {
                        0.0
                    } else {
                        ((n_docs - df[i] as f64) / df[i] as f64).ln().max(0.0)
                    }
                }
                _ => 1.0,
            };
            *w *= idf;
        }
    }
    local_vec
}

fn normalize_cosine(v: &[f64]) -> Vec<f64> {
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm == 0.0 || !norm.is_finite() {
        return v.to_vec();
    }
    v.iter().map(|x| x / norm).collect()
}

fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return f64::NEG_INFINITY;
    }
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::{FieldId, SymbolId, Value};
    use crate::ir::*;
    use std::collections::HashMap;

    fn sample_text_model() -> (TextIr, HashMap<SymbolId, String>) {
        let f_text = FieldId(0);
        let s_hello = SymbolId(1);
        let s_world = SymbolId(2);
        let s_foo = SymbolId(3);
        let mut sym_map = HashMap::new();
        sym_map.insert(s_hello, "hello world".into());
        sym_map.insert(s_world, "world".into());
        sym_map.insert(s_foo, "foo bar".into());
        let dict = vec!["hello".into(), "world".into(), "foo".into(), "bar".into()];
        let corpus = vec![
            TextDocumentIr {
                id: "doc1".into(),
                id_symbol: SymbolId(10),
                name: None,
            },
            TextDocumentIr {
                id: "doc2".into(),
                id_symbol: SymbolId(11),
                name: None,
            },
        ];
        // doc1 has hello world, doc2 has foo bar
        let dtm = vec![vec![1.0, 1.0, 0.0, 0.0], vec![0.0, 0.0, 1.0, 1.0]];
        let model = TextIr {
            function_name: "classification".into(),
            model_name: Some("text".into()),
            mining_schema: MiningSchemaIr {
                active_fields: vec![f_text],
                target_field: None,
                field_metas: vec![],
                missing_value_replacement: None,
            },
            output: vec![],
            targets: vec![],
            dictionary: dict,
            corpus,
            document_term_matrix: dtm,
            normalization: Some(TextNormalizationIr {
                local_term_weights: "termFrequency".into(),
                global_term_weights: "none".into(),
                document_normalization: "none".into(),
            }),
            similarity: Some(TextSimilarityIr {
                similarity_type: "cosine".into(),
            }),
            number_of_terms: 4,
            number_of_documents: 2,
        };
        (model, sym_map)
    }

    #[test]
    fn text_cosine_hello_world() {
        let (model, sym_map) = sample_text_model();
        let f = FieldId(0);
        // input "hello" -> closest to doc1
        let mut values = vec![Value::Missing; 1];
        // Discrete input requires symbol mapping; we pass symbol id for "hello world" which tokenizes to hello+world -> doc1
        let sid = SymbolId(1);
        values[f.as_usize()] = Value::Discrete(sid);
        let idx = best_document_index(&model, &values, Some(&sym_map)).unwrap();
        assert_eq!(idx, 0);
        let pred = evaluate_text(&model, &values, Some(&sym_map));
        match pred {
            Value::Discrete(sid) => assert_eq!(sid, SymbolId(10)),
            _ => panic!(),
        }
    }

    #[test]
    fn text_euclidean() {
        let (mut model, sym_map) = sample_text_model();
        model.similarity = Some(TextSimilarityIr {
            similarity_type: "euclidean".into(),
        });
        let f = FieldId(0);
        let mut values = vec![Value::Missing; 1];
        values[f.as_usize()] = Value::Discrete(SymbolId(2)); // "world"
        let idx = best_document_index(&model, &values, Some(&sym_map)).unwrap();
        assert_eq!(idx, 0); // world is in doc1
    }

    #[test]
    fn text_missing() {
        let (model, sym_map) = sample_text_model();
        let values = vec![Value::Missing];
        assert_eq!(
            evaluate_text(&model, &values, Some(&sym_map)),
            Value::Missing
        );
        assert!(best_document_index(&model, &values, Some(&sym_map)).is_none());
    }

    #[test]
    fn text_tokenization() {
        let toks = tokenize("Hello, WORLD! foo-bar");
        assert_eq!(toks, vec!["hello", "world", "foo", "bar"]);
    }
}
