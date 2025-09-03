mod inference;
mod server;

use llama_cpp_2::llama_backend::LlamaBackend;

use crate::inference::engine::QwenInferenceEngine;
use crate::server::Server;

#[tokio::main]
async fn main() {

    let backend = LlamaBackend::init().unwrap();
    let engine =
        QwenInferenceEngine::new("./models/Qwen3-4B-Thinking-2507-Q2_K_L.gguf", backend);

    let server = Server {};
    server.run().await;
}
