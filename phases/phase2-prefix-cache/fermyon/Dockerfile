# Fermyon Wasm caching proxy — Phase 2
#
# Installs the Spin CLI into debian:bookworm-slim, then packages the
# pre-built Wasm binaries and spin.toml.
#
# The Wasm binaries must be built locally before this image is built:
#
#   cd phases/phase2-prefix-cache/fermyon
#   cargo build --workspace --target wasm32-wasip1 --release
#
# Build (run from phases/phase2-prefix-cache/fermyon/):
#   docker build \
#       -t ghcr.io/jginsj/fermyon-prefix-cache:latest \
#       .
#
# Push:
#   docker push ghcr.io/jginsj/fermyon-prefix-cache:latest
#
# Run locally (supply Spin variables via env):
#   docker run --rm \
#       -e SPIN_VARIABLE_VALKEY_ADDRESS=redis://localhost:6379 \
#       -e SPIN_VARIABLE_VLLM_URL=http://localhost:8000 \
#       -p 8082:8082 \
#       ghcr.io/jginsj/fermyon-prefix-cache:latest
#
# In Kubernetes, variables are provided via a ConfigMap mounted as env vars.
# See fermyon/k8s/fermyon-deployment.yaml.

FROM debian:bookworm-slim

# Download the spin binary directly from GitHub releases.
# Using the tarball avoids the install.sh script, which would try to clone
# the templates repo via git (not installed) and fail with exit code 1.
# Templates are only needed for `spin new`; `spin up` works without them.
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates && \
    curl -fsSL https://github.com/spinframework/spin/releases/download/v3.6.3/spin-v3.6.3-linux-amd64.tar.gz \
        | tar -xz spin && \
    mv spin /usr/local/bin/ && \
    apt-get purge -y curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the Spin application manifest.
COPY spin.toml .

# Copy the compiled Wasm binaries from the workspace-level target directory.
# These are produced by: cargo build --workspace --target wasm32-wasip1 --release
# Cargo outputs both binaries to the workspace target/, not the crate-level target/.
# Paths must match the `source =` values in spin.toml exactly.
COPY target/wasm32-wasip1/release/proxy.wasm  target/wasm32-wasip1/release/proxy.wasm
COPY target/wasm32-wasip1/release/health.wasm target/wasm32-wasip1/release/health.wasm

# Start the Spin HTTP server on port 8082.
# The listen address must match [application.trigger.http] listen in spin.toml
# and the containerPort declared in fermyon/k8s/fermyon-deployment.yaml.
# Spin variables (valkey_address, vllm_url, cache_ttl) are injected at
# runtime via SPIN_VARIABLE_* environment variables — never baked into the image.
ENTRYPOINT ["spin", "up", "--listen", "0.0.0.0:8082"]
