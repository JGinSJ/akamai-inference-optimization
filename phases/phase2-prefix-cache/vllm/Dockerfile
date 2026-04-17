# vLLM + LMCache — Phase 2 external KV cache offload to Valkey
#
# Base:    vllm/vllm-openai:v0.18.1  (latest stable, released 2026-03-31)
# LMCache: 0.4.3                      (latest stable, released 2026-04-07)
#
# Compatibility: LMCache 0.4.3 is confirmed compatible with vLLM 0.18.x.
# See the LMCache compatibility matrix: https://docs.lmcache.ai/getting_started/installation.html
#
# What this image adds
# --------------------
# The official vllm/vllm-openai image provides the vLLM OpenAI-compatible
# server.  This layer adds LMCache, which offloads KV cache blocks that
# overflow GPU VRAM to an external store.  In this project that store is
# Valkey — see phases/phase2-prefix-cache/valkey/valkey.yaml.
#
# Protocol note: LMCache communicates with Valkey using the RESP wire
# protocol (the same protocol used by Redis).  Valkey is an open-source,
# Redis-compatible fork — not Redis itself.  The lmcache package requires
# no Redis-specific extras; RESP compatibility is sufficient.
#
# Build
# -----
#   docker build \
#       -t ghcr.io/<owner>/vllm-lmcache:v0.18.1 \
#       -f phases/phase2-prefix-cache/vllm/Dockerfile \
#       phases/phase2-prefix-cache/vllm/
#
# Push
# ----
#   docker push ghcr.io/<owner>/vllm-lmcache:v0.18.1
#
# The deployment manifest (vllm.yaml) references this image and passes:
#   --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
#   --no-enable-prefix-caching
# with LMCACHE_CONFIG_FILE pointing to the mounted LMCache ConfigMap.
# See vllm.yaml for the full argument list.

FROM vllm/vllm-openai:v0.18.1

# Install LMCache 0.4.3.
#
# No extras flag (e.g. [redis]) is needed.  LMCache uses the RESP protocol
# to talk to Valkey (open-source Redis-compatible fork) — not a Redis
# client library.  Plain `pip install lmcache` includes all required
# dependencies for the RESP remote backend.
RUN pip install --no-cache-dir lmcache==0.4.3

# Entrypoint and CMD are inherited unchanged from the base image.
# The deployment manifest overrides the command at runtime:
#   vllm serve <model> [args...]
# See vllm.yaml for the full runtime command.
