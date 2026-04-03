#include "LLMWrapper.h"

#include <algorithm>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <vector>

#include <iostream>
#include <filesystem>

#pragma region Helper Functions
namespace {
	/// <summary>
	/// Ensures the llama backend is initialized exactly once in a thread-safe manner.
	/// </summary>
	void EnsureLlamaBackendInit() {
		static std::once_flag s_once;
		std::call_once(s_once, []() {
			llama_backend_init();
			std::atexit([]() { llama_backend_free(); });
			});
	}

	/// <summary>
	/// Determines the optimal number of threads to use based on hardware capabilities.
	/// </summary>
	/// <returns>The recommended thread count, which is one less than the available hardware threads (minimum of 1), or 4 if hardware concurrency cannot be determined.</returns>
	int PickThreadCount() {
		unsigned int threadCount = std::thread::hardware_concurrency();
		if (threadCount == 0) {
			threadCount = 4; // Default to 4 threads if hardware_concurrency can't determine
		}
		return std::max(1u, threadCount - 1); // Use one less than the total to avoid saturating the system
	}

	/// <summary>
	/// Decodes a vector of tokens in batches using the provided llama context.
	/// </summary>
	/// <param name="context">A reference to a unique pointer containing the llama context used for decoding.</param>
	/// <param name="tokens">The vector of tokens to decode.</param>
	/// <returns>True if all token chunks were successfully decoded; false if the context is invalid or decoding fails.</returns>
	bool DecodeTokensInChunks(std::unique_ptr<llama_context, LlamaContextDeleter>& context, const std::vector<llama_token>& tokens) {
		if (!context) return false;
		const int batchSize = llama_n_batch(context.get());
		int i = 0;

		while (i < tokens.size())
		{
			const int chunkSize = std::min<int>(batchSize, tokens.size() - i);

			llama_batch batch = llama_batch_get_one(const_cast<llama_token*>(tokens.data()) + i, chunkSize);

			const int result = llama_decode(context.get(), batch);

			if (result != 0) return false;
			i += chunkSize;
		}

		return true;
	}

	/// <summary>
	/// Decodes a single token using the provided llama context.
	/// </summary>
	/// <param name="context">A reference to a unique pointer containing the llama context used for decoding.</param>
	/// <param name="token">The token to decode.</param>
	/// <returns>Returns true if the token was successfully decoded, false otherwise.</returns>
	bool DecodeSingleToken(std::unique_ptr<llama_context, LlamaContextDeleter>& context, llama_token token) {
		llama_batch batch = llama_batch_get_one(&token, 1);
		bool result = llama_decode(context.get(), batch) == 0;
		return result;
	}

	/// <summary>
	/// Converts a token to its string representation and returns the detokinsed string.
	/// </summary>
	/// <param name="vocab">A reference to the unique pointer containing the vocabulary object used for token conversion.</param>
	/// <param name="token">The token to convert to its string representation.</param>
	/// <returns>The string representation of the token, or an empty string if conversion fails.</returns>
	std::string TokenToString(std::shared_ptr<const llama_vocab>& vocab, llama_token token) {
		std::vector<char> buffer(64);

		int stringLength = llama_token_to_piece(vocab.get(), token, buffer.data(), buffer.size(), 0, false);
		if (stringLength < 0)
		{
			buffer.resize(-stringLength);
			stringLength = llama_token_to_piece(vocab.get(), token, buffer.data(), buffer.size(), 0, false);
		}
		else if (stringLength > buffer.size())
		{
			buffer.resize(stringLength);
			stringLength = llama_token_to_piece(vocab.get(), token, buffer.data(), buffer.size(), 0, false);
		}

		if (stringLength <= 0) return std::string();

		std::string detokenisedString(buffer.data(), buffer.data() + stringLength);
		return detokenisedString;
	}
}

#pragma	endregion

LLMWrapper::LLMWrapper() : llm_maxNewTokens(128)
{
	EnsureLlamaBackendInit();
}

LLMWrapper::~LLMWrapper()
{
	UnloadModel();
}


void LLMWrapper::UnloadModel()
{
	llm_model.reset();
	llm_context.reset();
	llm_vocab.reset();
}


bool LLMWrapper::LoadModel(const std::string& model_path)
{
	// 1) Init the llama backend and free any existing model/context if already loaded
	EnsureLlamaBackendInit();
	UnloadModel();


	// 2) Load the model from the specified path with appropriate parameters
	llama_model_params modelParams = llama_model_default_params();

	/*mmap sometimes causes issues, so we default to no mmap to avoid problems*/
	modelParams.use_mmap = true;
	modelParams.use_mlock = false;
	modelParams.n_gpu_layers = 0; // -1 to use all GPU Layers, 0 to use CPU, >0 to use that many GPU layers

	// Log information about the model file to make sure the path is correct and the file is accessible
	std::cout << "sizeof(void*): " << sizeof(void*) << "\n";
	std::cout << "file_size: " << std::filesystem::file_size(model_path) << " bytes\n";

	llm_model.reset(llama_model_load_from_file(model_path.c_str(), modelParams));

	if (!llm_model) {
		return false;
	}
	

	// 3) Get the vocab from the model and store it in a shared_ptr with a no-op deleter since it's owned by the model
	llm_vocab = std::shared_ptr<const llama_vocab>(
		llama_model_get_vocab(llm_model.get()),
		[](const llama_vocab*) { /* no-op: owned by llm_model */ }
	);

	if (!llm_vocab) {
		UnloadModel();
		return false;
	}


	// 4) Create a context for the model with appropriate parameters
	llama_context_params contextParams = llama_context_default_params();

	contextParams.n_ctx = 1024; // Set context size to 1024 tokens (can be adjusted based on model capabilities and requirements)
	contextParams.n_batch = 256; // Set logical batch size to 256 tokens for decoding the prompt in chunks
	contextParams.n_ubatch = contextParams.n_batch; // Set physical batch size to match logical batch size
	contextParams.n_seq_max = 1; // Set max sequences to 1 for single sequence generation

	contextParams.n_threads = PickThreadCount(); // Set number of threads for generation to the optimal thread count
	contextParams.n_threads_batch = contextParams.n_threads; // Set number of threads for batch processing to match generation threads

	contextParams.embeddings = false; // Disable embedding extraction for generation
	contextParams.no_perf = true; // Disable performance timing to reduce overhead during generation

	contextParams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED; // Disable Flash Attention for compatibility

	llm_context.reset(llama_init_from_model(llm_model.get(), contextParams));

	if (!llm_context) {
		UnloadModel();
		return false;
	}

	return true;
}


std::string LLMWrapper::Generate(const std::string& prompt)
{
	if (!llm_context || !llm_model || !llm_vocab)
		return std::string();

	// 0) Reset the model "memory" (KV Cache) so each generate call is fresh
	llama_memory_clear(llama_get_memory(llm_context.get()), true);


	// 1) Tokenize the prompt and decode the tokens to update the model state with the prompt
	std::vector<llama_token> promptTokens(prompt.size() + 8); // Allocate more than enough space for tokenization
	int tokenCount = llama_tokenize(
		llm_vocab.get(),
		prompt.c_str(),
		prompt.size(),
		promptTokens.data(),
		promptTokens.size(),
		true,
		false);

	if (tokenCount < 0) {
		// Need more space for tokenization, resize and try again
		promptTokens.resize(-tokenCount);
		tokenCount = llama_tokenize(
			llm_vocab.get(),
			prompt.c_str(),
			prompt.size(),
			promptTokens.data(),
			promptTokens.size(),
			true,
			false);
	}

	if (tokenCount < 0) {
		// Tokenization failed even after resizing
		return std::string();
	}

	promptTokens.resize(tokenCount); // Resize to actual token count


	// 2) Decode the prompt tokens in batches to update the model state
	if (!DecodeTokensInChunks(llm_context, promptTokens)) {
		// Failed to decode the end-of-sequence token after the prompt, return empty response
		return std::string();
	}


	// 3) Initialize the sampler chain with a greedy sampler for generation
	llama_sampler_chain_params samplerParams = llama_sampler_chain_default_params();
	samplerParams.no_perf = true; // Disable performance timing for sampling to reduce overhead

	std::unique_ptr<llama_sampler, LlamaSamplerDeleter> sampler(llama_sampler_chain_init(samplerParams));
	llama_sampler_chain_add(sampler.get(), llama_sampler_init_greedy());

	// 4) Generate tokens one at a time until we reach the max new tokens or an end-of-sequence token
	std::string response;
	response.reserve(256); // Reserve some initial space for the response string to reduce reallocations

	const int n_ctx = llama_n_ctx(llm_context.get());
	int maxNewTokens = llm_maxNewTokens;
	maxNewTokens = std::max<int>(0, std::min<int>(maxNewTokens, n_ctx - promptTokens.size() - 1));

	for (int i = 0; i < maxNewTokens; ++i)
	{
		// Sample using logits from the last decoded token
		const llama_token next = llama_sampler_sample(sampler.get(), llm_context.get(), -1);

		// Tell the sampler what token we accepted so it can update its internal state
		llama_sampler_accept(sampler.get(), next);

		// If we hit the end-of-sequence token, stop generation
		if (llama_vocab_is_eog(llm_vocab.get(), next)) {
			break;
		}

		response += TokenToString(llm_vocab, next);

		// Decode the new token to update the model state for the next sampling step
		if (!DecodeSingleToken(llm_context, next)) {
			// Failed to decode the new token, stop generation and return what we have so far
			break;
		}
	}

	// 5) Return the generated response string
	llama_sampler_free(sampler.release()); // Free the sampler explicitly since we're using a unique_ptr
	
	return response;
}
