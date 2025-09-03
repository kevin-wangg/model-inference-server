use std::sync::Arc;

use llama_cpp_2::llama_backend::LlamaBackend;

use crate::inference::engine::QwenInferenceEngine;

#[derive(Clone)]
pub struct AppState {
    app_name: String,
    engine: Arc<QwenInferenceEngine>,
}

impl AppState {
    pub fn new(app_name: &str) -> Self {
        let backend = LlamaBackend::init().unwrap();
        let engine =
            QwenInferenceEngine::new("./models/Qwen3-4B-Thinking-2507-Q2_K_L.gguf", backend);
        Self {
            app_name: app_name.into(),
            engine: Arc::new(engine),
        }
    }

    pub fn get_name(&self) -> String {
        self.app_name.clone()
    }

    pub fn generate(&self, prompt: &str) -> String {
        self.engine.generate(prompt).expect("Failed to generate")
    } 
}
