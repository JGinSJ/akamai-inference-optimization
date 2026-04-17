// This file is superseded by the Cargo workspace layout.
//
// The codebase has been split into two separate wasm binaries so that each
// Spin component has its own dedicated entry point with no path inspection:
//
//   proxy/src/lib.rs   — POST /v1/chat/completions handler
//   health/src/lib.rs  — GET /health handler
//
// This file is no longer compiled.  The workspace root Cargo.toml
// (fermyon/Cargo.toml) lists members = ["proxy", "health"] and does not
// include a top-level [package] or [lib] section.
