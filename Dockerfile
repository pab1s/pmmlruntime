# Score PMML from a mounted file or CSV with no JVM in the image.
#
#   docker build -t pmmlruntime:0.1 .
#   docker run --rm -v "$PWD:/data" pmmlruntime:0.1 /data/model.pmml /data/input.csv --output /data/out.csv
#
# The runtime stage carries one binary and no toolchain, no JDK, and no Python.

FROM rust:1.78-bookworm AS builder
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
RUN cargo build -p pmmlruntime --release --example score_file

FROM gcr.io/distroless/cc-debian12
COPY --from=builder /src/target/release/examples/score_file /usr/local/bin/score_file
WORKDIR /data
ENTRYPOINT ["/usr/local/bin/score_file"]
