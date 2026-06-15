FROM rust:slim-bookworm

# System dependencies: libhdf5-dev for the hdf5 crate (links against libhdf5),
# clang/pkg-config in case hdf5-sys needs to generate bindings.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libhdf5-dev pkg-config clang \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy everything needed to resolve and fetch dependencies first, for layer caching.
COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY src ./src

# Install the pinned nightly toolchain (from rust-toolchain.toml) and pre-fetch
# every crate dependency (including the git "rgb" dep) so the entrypoint can
# compile fully offline at container start, on the host's actual CPU.
RUN rustup show && cargo fetch --locked

COPY entrypoint.sh ./
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
