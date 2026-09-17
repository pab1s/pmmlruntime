# Releasing & Registries

> **Info:** Looking for the build commands instead? See [Docker & CI](./docker.md). This page covers what a tag publishes and what each registry needs before the first release.

One tag publishes every artifact: `git tag v0.1.1 && git push origin v0.1.1`. The `cd.yml` workflow runs the version gate and the four test suites first, so a tag cannot ship a build that fails or disagrees with `VERSION_NUMBER`.

`v0.1.0` is already published on crates.io, so the next release is `v0.1.1`. Bump `VERSION_NUMBER`, run the gate, then tag.

A manual run of the workflow (Actions, cd, Run workflow) is a dry run. Every build, artifact upload, and download runs, and only the steps that write to a registry are skipped. Use it to validate the pipeline before the first tag. A passing dry run means the artifacts are sound; the only thing left is credentials.

## What a tag publishes

| Registry | Artifact | Built by | Skipped until |
| --- | --- | --- | --- |
| crates.io | `pmmlruntime` | `cargo publish -p pmmlruntime --locked` | `CARGO_REGISTRY_TOKEN` exists |
| PyPI | `pmmlruntime` wheels for linux, macOS, and Windows, plus the sdist | `maturin` through `PyO3/maturin-action` | `PUBLISH_PYPI` is `true` |
| npm | `pmmlruntime-node` and one package per platform triple | `@napi-rs/cli` | `PUBLISH_NPM` is `true` |
| npm | `pmmlruntime-web` | `wasm-pack build --target web` | `PUBLISH_NPM` is `true` |
| Maven Central | `com.pmmlruntime:pmmlruntime` with the JNI library inside the jar | `mvn -Prelease deploy` | `PUBLISH_MAVEN` is `true` |
| GHCR | `ghcr.io/<owner>/pmmlruntime:0.1.1`, `:0.1`, `:latest` | `docker build` on distroless | always |
| GitHub release | `pmml_runtime.h`, `pmml_runtime.hpp`, `libpmmlruntime.so`, `libpmmlruntime.a`, `SHA256SUMS` | `cargo build --release` | always |

Registries that need an account are behind a repository variable, so a tag stays green while you set one up. Turn each on under Settings, Variables and Secrets.

## Version rule

`VERSION_NUMBER` is the single source. Every manifest must agree with it:

```bash
python3 scripts/check-versions.py          # all manifests match
python3 scripts/check-versions.py v0.1.1   # and the tag matches
```

The gate covers the Cargo workspace, `python/pyproject.toml`, `python/_native`, `java/pom.xml`, `java/native`, and both JavaScript packages. Bump `VERSION_NUMBER`, run the script, then tag.

## First-time setup per registry

**PyPI.** Configure trusted publishing once at the project settings page with workflow file `cd.yml` and environment `pypi`, then set `PUBLISH_PYPI` to `true`. No secret is stored. To use a token instead, drop the `id-token: write` block in `cd.yml` and pass `password: ${{ secrets.PYPI_API_TOKEN }}` to the publish step.

**npm.** Create an automation token, save it as `NPM_TOKEN`, and set `PUBLISH_NPM` to `true`. The job publishes with `--provenance`, so each release links back to the workflow that built it.

**Maven Central.** Register the `com.pmmlruntime` namespace with the Central Portal, then save four secrets: `MAVEN_CENTRAL_USERNAME`, `MAVEN_CENTRAL_PASSWORD` (portal token), `GPG_PRIVATE_KEY`, and `GPG_PASSPHRASE`. Set `PUBLISH_MAVEN` to `true`. The release profile attaches the sources and javadoc jars that Central requires, signs every file, and waits for the deployment to finish.

**crates.io.** Save `CARGO_REGISTRY_TOKEN`. The job runs `cargo package` before publishing, so a broken package fails before it reaches the registry.

> **Note:** The JNI library ships inside the jar under `native/<os>-<arch>/`. `NativeLoader` extracts the right one at first use, so a Maven user needs no `java.library.path` setup.

## What the workflow refuses to do

- Publish a tag whose name disagrees with `VERSION_NUMBER`.
- Publish any artifact if a test suite failed: Rust fixtures, Python, Java, or Node.
- Push a jar without the natives for linux, macOS, and Windows, because the publish job waits on all five build rows.

## Local check before tagging

Run the same gate the workflow runs:

```bash
python3 scripts/check-versions.py v0.1.1
cargo test --workspace
python3 docs/check-docs.py
mdbook build docs
```

Then tag. The workflow repeats all of it on the runner, which catches a dirty worktree that a local run would hide.

## Next Steps

- [Docker & CI](./docker.md): the container jobs, image tags, and CI steps in full.
- [Bindings Overview](./bindings.md): which binding ships to which registry.
- [C ABI & FFI](./c.md): the header and library the GitHub release carries.
- [Correctness & Fixture Parity](../evaluation/correctness.md): the fixture suite that gates every publish.

*Next: [Correctness & Fixture Parity](../evaluation/correctness.md) → · Previous: [Docker & CI](./docker.md)*
