use std::cmp;
use std::error;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel, Special};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;

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

    pub fn generate(&self, prompt: &str) -> Result<String, Box<dyn error::Error>> {
        println!("prompt {}", prompt);
        let context_params = LlamaContextParams::default();
        let mut context = self.model.new_context(&self.backend, context_params)?;

        let tokens = self.model.str_to_token(prompt, AddBos::Always)?;

        println!("tokens {:?}", tokens);

        let mut batch = LlamaBatch::new(tokens.len() + 256, 1);
        batch.add_sequence(&tokens, 0, true)?;
        context.decode(&mut batch)?;

        let mut generated_tokens = Vec::new();
        let max_tokens = 500;

        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::top_k(20),      // Keep top 40 tokens
            LlamaSampler::top_p(0.8, 1), // Nucleus sampling
            LlamaSampler::min_p(0.00, 1), // Min-p filtering
            LlamaSampler::temp(0.7),      // Apply temperature
            LlamaSampler::dist(0),    // Final random selection with seed
        ]);

        for _ in 0..max_tokens {
            // let candidates = context.candidates();
            // let next_token = candidates
            //     .into_iter()
            //     .max_by(|a, b| a.p().partial_cmp(&b.p()).unwrap_or(cmp::Ordering::Equal))
            //     .map(|candidate| candidate.id())
            //     .unwrap_or(LlamaToken::new(0));
            let next_token = sampler.sample(&context, -1);

            if self.model.is_eog_token(next_token) {
                break;
            }

            generated_tokens.push(next_token);
            batch.clear();
            batch.add(
                next_token,
                (tokens.len() + generated_tokens.len() - 1) as i32,
                &[0],
                true,
            )?;
            // batch.add_sequence(&generated_tokens, 0, true)?;
            context.decode(&mut batch)?;
        }

        println!("generated tokens {:?}", generated_tokens);

        let generated_text = self
            .model
            .tokens_to_str(&generated_tokens, Special::Plaintext)?;

        println!("generated text: {}", generated_text);

        Ok(generated_text)
    }
}
