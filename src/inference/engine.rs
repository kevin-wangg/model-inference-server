use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::params::LlamaModelParams;

pub struct QwenInferenceEngine {
    model: LlamaModel,
    backend: LlamaBackend,
}

impl QwenInferenceEngine {
    pub fn new(model_path: &str, backend: LlamaBackend) -> Self {
        let model_params = LlamaModelParams::default();
        let model = LlamaModel::load_from_file(&backend, model_path, &model_params).unwrap();
        Self { model, backend }
    }
}
