use axum::Json;
use axum::extract::State;
use serde::Deserialize;

use crate::server::app_state::AppState;

pub async fn handle_hello(State(state): State<AppState>) -> String {
    format!("Hello from {}", state.get_name())
}

#[derive(Deserialize)]
pub struct Generate {
    prompt: String,
}

pub async fn handle_generate(
    State(state): State<AppState>,
    Json(payload): Json<Generate>,
) -> String {
    state.generate(&payload.prompt)
}
