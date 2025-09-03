use axum::extract::State;

use crate::server::app_state::AppState;

pub async fn handle_hello(State(state): State<AppState>) -> String {
    format!("Hello from {}", state.get_name())
}
