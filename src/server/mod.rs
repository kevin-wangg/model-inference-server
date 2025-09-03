mod app_state;
mod handlers;

use std::net::SocketAddr;

use axum::routing::{Router, get, post};

use crate::server::app_state::AppState;
use crate::server::handlers::{handle_generate, handle_hello};

pub struct Server {}

impl Server {
    pub async fn run(&self) {
        let state = AppState::new("Inference server");
        let app = Router::new()
            .route("/hello", get(handle_hello))
            .route("/generate", post(handle_generate))
            .with_state(state);
        let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
        let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
        println!("Server running on port 3000");
        axum::serve(listener, app).await.unwrap();
    }
}
