# Security & Hardening

> **Info:** This page covers XML hardening for services that score untrusted PMML. For thread safety and buffers, see [Concurrency & Memory](./concurrency.md). For the parse pipeline, see [Architecture Overview](../internals/architecture.md).

pmmlruntime treats every PMML file as untrusted input. The parser is wrapped, the limits are checked before allocation, and `verify_raw` plus `verify_ir` run before any model executes.

## Concepts

| Concept | Description |
| --- | --- |
| **PmmlReader** | Wrapper over `quick-xml 0.37` that checks depth and size before parsing. |
| **MAX_DEPTH** | `512`. Checked on every `Event::Start`; `Event::Empty` does not increase depth. |
| **MAX_FILE_BYTES** | `100 MB`. Rejected before a parser is created, in both `PmmlReader::from_bytes` and `new_reader`. |
| **DTD / XXE blocked** | DTD is ignored and entities are never expanded, so `&xxe;` stays literal. |
| **304 elements** | `RawPmml` mirrors `pmml.xsd`; unsupported markup fails verification instead of executing. |
| **BumpArena** | Per-chunk owned `bumpalo::Bump` that is `Send` and never crosses threads. |

`PmmlReader` and `new_reader` are the only entry points for XML. Both set `trim_text(true)` and `expand_empty_elements(true)`, and neither opts into entity expansion.

```rust
use pmmlruntime::xml::PmmlReader;
use quick_xml::events::Event;

let xml = br#"<PMML version="4.4"><Header/><DataDictionary/></PMML>"#;
let mut r = PmmlReader::from_bytes(xml)?;
loop {
    match r.read_event()? { Event::Eof => break, _ => {} }
}
# Ok::<(), pmmlruntime::base::PmmlError>(())
```

The reader above normalizes whitespace and empty elements. A payload such as `<!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>` returns `&xxe;` literally, because `quick-xml` 0.37 does not expand external entities and this module never enables it.

Depth is one `usize` increment per `Start` and one `saturating_sub` per `End`. A document deeper than 512 levels returns `PmmlError::ValidationError` with `"depth"` in the message, before `RawPmml` is built.

```rust
use pmmlruntime::xml::new_reader;

let bytes = std::fs::read("bench/pmml/DecisionTreeIris.pmml")?;
let reader = new_reader(&bytes)?; // MAX_FILE_BYTES checked here
# Ok::<(), Box<dyn std::error::Error>>(())
```

`new_reader` rejects a file over 100 MB before it allocates a parser buffer. `Session::from_bytes` reaches the same guard through `crate::xml::unmarshal`, then runs `verify_raw`, lowers, runs `verify_ir`, drops `RawPmml`, and publishes `Arc<Ir>`.

> **Note:** `RawPmml` owns `String` and `Vec` values for 304 elements and is dropped after `lower`. Only `Arc<Ir>` reaches the hot path.

The parser is fuzzed, exercised under `miri`, and covered by `cargo test --test hardening`, which drives `MAX_DEPTH`, `MAX_FILE_BYTES`, and the `LAG_BUFFER` cap of 128.

## Boundaries and guards

| Boundary | File | Guard | What it rejects |
| --- | --- | --- | --- |
| **XML depth** | `xml/reader.rs` | `MAX_DEPTH 512` on `Event::Start` | Nesting deeper than 512 levels |
| **XML size** | `xml/reader.rs` | `MAX_FILE_BYTES 100 MB` before `Reader` | Files over 100 MB |
| **Entity expansion** | `xml/reader.rs` | No entity opt-in | `&xxe;` stays literal text |
| **Unsupported markup** | `ir/verify.rs` | `verify_raw` on `unsupported_model` | `ModelComposition`, `CenterFields`, unknown `*Model` tags |
| **IR invariants** | `ir/verify.rs` | `verify_ir` | Dense-id gaps, non-topo `DerivedField` DAG |
| **Memory isolation** | `base/arena.rs` | `BumpArena` is `Send`, not `Sync` | Buffers crossing threads |

```rust
use pmmlruntime::{PmmlEnv, Session, SessionOptions};
let env = PmmlEnv::new();
let bytes = std::fs::read("bench/pmml/DecisionTreeIris.pmml")?;
let sess = Session::from_bytes(&env, &bytes, SessionOptions::default())?;
// Depth, size, XXE, verify_raw, and verify_ir all passed
assert!(sess.num_active_fields() > 0);
# Ok::<(), Box<dyn std::error::Error>>(())
```

One `from_bytes` call enforces all six boundaries. The table above names the file and the guard behind each one, so a reviewer can check a boundary without reading the whole crate. See [Session, Env & Lifecycle](../concepts/session.md).

> **Tip:** A `PmmlError::ValidationError` message contains `"depth"`, `"size"`, or `"unsupported markup"`, which tells you which guard refused the file. Log the full message.

## Pinned dependencies

| Dependency | Version | Security role | How it is tested |
| --- | --- | --- | --- |
| **quick-xml** | `0.37` | Pull parser with DTD blocked and depth 512 | `hardening.rs` depth and XXE cases |
| **bumpalo** | `3.x` | `BumpArena` per chunk | `miri test session_drop_no_leak` |
| **lasso** | `0.7` | `FieldId` and `SymbolId` interning | `proptest` random trees, 128 cases |
| **rayon** | `1.x` | `par_chunks(256)` isolation | `SessionIsSendSync` checks |
| **arrow** | `53.x` | `RecordBatch` columnar input | `csv_str_to_record_batch` overflow |

You pin these versions in `Cargo.toml` and commit `Cargo.lock`. Never write `quick-xml = "*"` or leave `bumpalo` unpinned.

```toml
[dependencies]
quick-xml = { version = "0.37", features = ["serialize"] }
bumpalo = "3"
lasso = "0.7"
```

A `cargo update -p quick-xml` change must pass `cargo test --test hardening` before it merges. That keeps the parser that enforces depth and size, and the arena that enforces memory isolation, under test.

`BumpArena` owns a `bumpalo::Bump` and is `Send`, so `rayon` chunks move it into their threads. `THREAD_ARENA` and `THREAD_VALUES` are `thread_local!` and never cross threads. `Value::Missing` is an explicit variant rather than `Option<Value>`, so `Op::JumpIfMissing` branches without an extra allocation and `miri` finds no leak.

> **Warning:** Never enable a `quick-xml` feature that adds DTD handling or entity expansion. The XXE guarantee depends on keeping entities literal.

> **Note:** `RawPmml` maps 304 elements to `pmml.xsd` 1:1 and is dropped after `lower`. `Extension` payloads always pass `verify_raw`, because a vendor extension cannot be expressed as `unsupported_model`.

## Failure modes

| Symptom | Cause | Fix | Follow-up |
| --- | --- | --- | --- |
| `ValidationError` `"depth"` | Nesting passes `MAX_DEPTH 512` | Split the PMML or raise the cap after review | Add a case to `hardening.rs` |
| `ValidationError` `"size"` | File over 100 MB | Compress the PMML | Update the `MAX_FILE_BYTES` docs |
| `UnsupportedMarkup` | `ModelComposition`, `CenterFields`, or an unknown `*Model` tag | Export without that element | Update `verify_raw` docs |
| `&xxe;` stays literal | DTD payload with `<!ENTITY` | No fix needed, entities stay literal | Keep the XXE test |
| `BumpArena` leak | `with_bump` does not reset | Call `reset` before and after | Add a `miri` test |
| `LAG_BUFFER` growth | Cap above 128, or not `thread_local` | Cap at 128 per batch | Document the cap |

```bash
cargo test --test hardening -- --nocapture
cargo fuzz run unmarshal -- -max_total_time=60
cargo miri test session_drop_no_leak_under_miri
```

These three commands cover depth, size, XXE, 5k-node chains, cycles, `Send + Sync`, and `proptest` byte inputs. Validate an untrusted upload in a short-lived task, then keep the verified `Session` for hot scoring, so you pay the XML cost once.

All 19 supported model elements verify through `pmml.xsd` ordering, including `Extension` vendor payloads that `serde` cannot express without code generation. `ModelComposition`, `CenterFields`, and unrecognized `*Model` tags raise `UnsupportedMarkup` at verification instead of executing partially.

## Next Steps

- [Concurrency & Memory](./concurrency.md): share `Session` across threads with stack and heap buffers.
- [Observability & Troubleshooting](./troubleshooting.md): map `PmmlError` variants to a cause.
- [Session, Env & Lifecycle](../concepts/session.md): cache `PmmlEnv` and pick `SessionOptions`.
- [Architecture Overview](../internals/architecture.md): trace `base → xml → ir → engine → session`.

*Next: [Concurrency & Memory](./concurrency.md) → · Previous: [Architecture Overview](../internals/architecture.md)*
