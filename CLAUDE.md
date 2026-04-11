# CLAUDE.md — AI Session Context

This file is loaded automatically by Claude Code at the start of every session.
It defines working rules, technical constraints, and project context.

## Working rules

These rules apply to every session in this repo. Follow them exactly.

1. Before writing any code, read any existing files in the repo.
2. Produce a step-by-step plan and wait for approval before starting.
3. Work on ONE phase or ONE clearly scoped task at a time.
4. After completing a task, stop. Summarize what changed, list files
   created or modified, and explain how to run/test it. Do not
   continue to the next task unless explicitly told to proceed.
5. Never fabricate benchmark numbers, GPU specs, or performance claims.
6. If unsure about something, ask — do not assume and proceed.
7. Keep all placeholder content clearly labeled with TODO or PLACEHOLDER.

## Technical constraints

Never deviate from these.

| Constraint | Value |
|---|---|
| Platform | Akamai Cloud / Akamai LKE |
| GPU targets | RTX 4000 Ada and RTX PRO 6000 Blackwell **only** |
| Phase 2 cache | Valkey (not Redis) |
| Phase 2 front door | Fermyon Wasm Functions (not EdgeWorkers) |
| Phase 3 image model | Qwen-Image (not FLUX or other image models) |
| Package manager | Plain pip (no uv, no poetry) |
| Docs format | Pure Markdown (no MkDocs, no Sphinx) |
| CLAUDE.md | Repo root only (not duplicated into phases/) |
| Infrastructure | Open-source wherever possible |

## Project phases

| Phase | Name | Status |
|-------|------|--------|
| 1 | KV Cache from Scratch (PyTorch) | Complete |
| 2 | Fermyon + Valkey + vLLM Prefix Caching | Not started |
| 3 | Qwen-Image Inference on Akamai Cloud | Not started |
| 4 | Multi-GPU Benchmarking and Cost Model | Not started |

Update the Status column as phases are completed.

## Repo layout

```
akamai-inference-optimization/
├── phases/
│   ├── phase1-kv-cache/
│   ├── phase2-prefix-cache/
│   ├── phase3-qwen-image/
│   └── phase4-benchmarks/
├── docs/
│   ├── architecture.md
│   ├── hardware.md
│   └── phases/
│       ├── phase1-kv-cache.md
│       ├── phase2-fermyon-valkey.md
│       ├── phase3-qwen-image.md
│       └── phase4-benchmarks.md
├── CLAUDE.md               ← this file
├── LICENSE
├── README.md
└── pyproject.toml
```

## Open questions log

Record unresolved decisions here. Remove entries when resolved.

| # | Question | Raised | Resolved |
|---|----------|--------|----------|
| 1 | CI/CD pipeline design (GitHub Actions vs other) | Phase 1 scaffold | — |
| 2 | Akamai LKE cluster sizing for Phase 2 | Phase 1 scaffold | — |
| 3 | Valkey version and deployment mode (standalone vs cluster) | Phase 1 scaffold | — |
| 4 | Qwen-Image model variant and quantization level for Phase 3 | Phase 1 scaffold | — |
| 5 | Cost model methodology for Phase 4 (per-token, per-request, or per-hour) | Phase 1 scaffold | — |

## Session notes

Use this section to capture decisions made mid-session that do not fit
neatly into the open questions log.

- 2026-04-10: Package manager confirmed as plain pip.
- 2026-04-10: CLAUDE.md at repo root only.
- 2026-04-10: Pure Markdown for all docs.
- 2026-04-10: Apache 2.0 license selected.
- 2026-04-10: hardware.md GPU specs stubbed as PLACEHOLDER pending real data.
