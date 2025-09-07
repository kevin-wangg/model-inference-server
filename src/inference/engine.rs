use std::error;
use std::num::NonZero;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel, Special};
use llama_cpp_2::sampling::LlamaSampler;

const MAX_TOKENS: i32 = 10000;

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
        let formatted_prompt = format!(
            "
            <|im_start|>user
            {}<|im_end|>
            <im_start|>assistant
            ",
            prompt
        );
        let context_params = LlamaContextParams::default().with_n_ctx(NonZero::new(4096));
        let mut context = self.model.new_context(&self.backend, context_params)?;

        let tokens = self.model.str_to_token(&formatted_prompt, AddBos::Always)?;

        let mut batch = LlamaBatch::new(tokens.len() + 256, 1);
        batch.add_sequence(&tokens, 0, true)?;
        context.decode(&mut batch)?;

        let mut generated_tokens = Vec::new();

        // Follows the recommended settings from https://docs.unsloth.ai/basics/qwen3-how-to-run-and-fine-tune/qwen3-2507#best-practices
        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::top_k(20),      // Keep top 40 tokens
            LlamaSampler::top_p(0.8, 1),  // Nucleus sampling
            LlamaSampler::min_p(0.00, 1), // Min-p filtering
            LlamaSampler::temp(0.7),      // Apply temperature
            LlamaSampler::dist(0),        // Final random selection with seed
        ]);

        for _ in 0..MAX_TOKENS {
            let next_token = sampler.sample(&context, -1);

            if self.model.is_eog_token(next_token) {
                println!("EOG token encountered");
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
            context.decode(&mut batch)?;
        }

        let generated_text = self
            .model
            .tokens_to_str(&generated_tokens, Special::Plaintext)?;

        println!("generated text: {}", generated_text);

        Ok(generated_text)
    }
}
