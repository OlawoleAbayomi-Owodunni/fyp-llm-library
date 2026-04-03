#pragma once
#include <string>
#include <memory>
#include "llama.h"

// Custom deleters for unique_ptr to ensure proper cleanup of llama resources
struct LLamaModelDeleter final {
	void operator()(llama_model* p) const noexcept {
		if (p) llama_model_free(p);
	}
};

struct LlamaContextDeleter final {
	void operator()(llama_context* p) const noexcept {
		if (p) llama_free(p);
	}
};

struct LlamaSamplerDeleter final {
	void operator()(llama_sampler* p) const noexcept {
		if (p) llama_sampler_free(p);
	}
};


class LLMWrapper {
public:
	LLMWrapper();
	~LLMWrapper();

    bool LoadModel(const std::string& model_path);
	void UnloadModel();
    std::string Generate(const std::string& prompt);

private:
	std::unique_ptr<llama_model, LLamaModelDeleter> llm_model;
	std::unique_ptr<llama_context, LlamaContextDeleter> llm_context;
	std::shared_ptr<const llama_vocab> llm_vocab;

	int llm_maxNewTokens;
};