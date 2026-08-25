# Contributing — pmmlruntime

Gitflow: `main` (protected) ← `development` (protected) ← `feature/*` / `chore/*` / `fix/*`.

- **No direct pushes to `main`/`development`.** Open PR, 1 review required.
- `chore/migration-plan` is the first branch — plan only, no code.
- Follow `docs/PORTING.md` + `docs/OWNERSHIP.tsv` once they land.

## Quick start

```sh
git checkout development && git pull
cargo fmt --check
cargo clippy --workspace -- -W clippy::pedantic -D warnings
cargo clippy --workspace --all-targets -- -W clippy::pedantic || true
cargo check --workspace
cargo test --workspace
cargo test -p pmmlruntime --test all_fixtures -- --nocapture # 45/45
cargo test -p pmmlruntime --doc
cargo doc --workspace --no-deps
```

## Workflow

- Branch per task: `git checkout -b feat/<slug>` off `development` (or `chore/docs-*`, `fix/*`).
- Commit per file, `git commit <file> -m "feat(scope): ..."` — never `git stash` / `reset --hard` / `cargo` in loops (see `docs/PORTING.md` Rules).
- `cargo check` only at queue start; `cargo fmt` before push.
- PR → `development` (draft until gates green) → `main`. `gh pr create --base development --head feat/<slug> --draft`.

## Documentation

- All `pub` items need rustdoc: purpose, params, return, `# Errors`, `# Panics`, `# Safety` for `unsafe`, `# Performance` when material, links, and executable `# Examples` (` ```rust ` not `ignore`).
- Crate/module docs answer: What belongs here? Why does this module exist? How does it relate to neighbors? What should a user import? Keep `docs/ARCHITECTURE.md` for internals, `cargo doc` for API.
- Before stating a guarantee, verify in implementation/tests/spec — never invent complexity/bounds/thread-safety.
- Generated docs must build: `cargo doc --workspace --no-deps` 0 unresolved (17 `redundant_explicit_links` allowed), `cargo test -p pmmlruntime --doc` passes.

## Validation gates (1.0)

| Gate | Command | Threshold |
|---|---|---|
| `fmt` | `cargo fmt --check` | 0 diff |
| `clippy` | `cargo clippy --workspace -- -W clippy::pedantic -D warnings` | 0 warnings |
| `check` | `cargo check --workspace` | green |
| `test` | `cargo test --workspace` + `cargo test -p pmmlruntime --test all_fixtures` | pass, 45/45 `all_fixtures` |
| `doc` | `cargo test -p pmmlruntime --doc` + `cargo doc --workspace --no-deps` | pass, 0 unresolved (17 redundant allowed) |
| `bench` | `cargo bench -p pmml-bench -- --sample-size 30` | `≤800 ns` single, `≤500µs` batched |

## Links

- Architecture: `docs/ARCHITECTURE.md`
- Benchmarks: `docs/BENCHMARK.md`
- Porting: `docs/PORTING.md` + `docs/OWNERSHIP.tsv`
