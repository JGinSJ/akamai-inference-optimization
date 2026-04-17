//! Health check component — Phase 2
//!
//! Single-purpose entry point for the `health` component in spin.toml.
//! Responds to every request with 200 OK.  No Spin variables are read,
//! no outbound connections are made, and no path inspection is needed —
//! Spin's route-based dispatch ensures this handler only receives
//! requests matched by the `/health` trigger route.

use spin_sdk::http::{IntoResponse, Request, Response};
use spin_sdk::http_component;

#[http_component]
async fn handle(_req: Request) -> impl IntoResponse {
    Response::builder()
        .status(200)
        .header("content-type", "text/plain")
        .body("ok")
        .build()
}
