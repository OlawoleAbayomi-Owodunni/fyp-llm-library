# Software Requirements Specification

## FYP LLM Library — Local Inference Wrapper

**Project Title:** FYP LLM Library (Local Inference Wrapper)

**Author:** Olawole Abayomi-Owodunni

**Date:** April 2026

**Version:** 1.0

---

## Contents

1. [Acknowledgements](#1-acknowledgements)
2. [Functional Specification](#2-functional-specification)
   - 2.1 [Introduction and Background](#21-introduction-and-background)
   - 2.2 [Problem Domain](#22-problem-domain)
   - 2.3 [Feasibility Study](#23-feasibility-study)
   - 2.4 [Literature Survey and Research](#24-literature-survey-and-research)
   - 2.5 [System Overview](#25-system-overview)
   - 2.6 [Functional Requirements](#26-functional-requirements)
   - 2.7 [Non-Functional Requirements](#27-non-functional-requirements)
   - 2.8 [User Interface Specification](#28-user-interface-specification)
   - 2.9 [Navigation and Usage Flow](#29-navigation-and-usage-flow)
   - 2.10 [System Constraints and Limitations](#210-system-constraints-and-limitations)
   - 2.11 [Critical Analysis and Conclusions](#211-critical-analysis-and-conclusions)
3. [References](#3-references)

---

## 1. Acknowledgements

We would like to thank the following people and organisations who assisted in completing this project:

- **Georgi Gerganov** and the open-source contributors to the `llama.cpp` project, whose high-performance C/C++ inference engine forms the foundational dependency of this library.
- **Meta AI** for releasing the Llama family of large language models under permissive research licences, making local inference research accessible.
- **Hugging Face** for hosting quantised GGUF model files and providing the `huggingface-hub` Python package used by this project's model download tooling.
- **Unsloth** for providing the pre-quantised `Llama-3.2-1B-Instruct-Q4_K_M.gguf` model file used as the default model in this project.
- The faculty and supervisors at the university who provided guidance and feedback throughout the development of this final-year project.

---

## 2. Functional Specification

### 2.1 Introduction and Background

Modern video games increasingly seek to deliver dynamic, emergent narrative experiences. Non-player character (NPC) dialogue, lore entries, and in-game text are traditionally hand-authored, which limits variety and replayability. Large Language Models (LLMs) offer an opportunity to generate contextual text at runtime, producing unique dialogue every playthrough.

However, integrating LLMs into games presents significant challenges. Cloud-based inference APIs introduce network latency, require an internet connection, incur per-request costs, and raise data-privacy concerns — all of which conflict with the requirements of offline, single-player titles. Running inference locally inside the game process eliminates these issues but demands careful engineering to manage model loading, memory, threading, and the complexity of third-party inference engines.

This project, the **FYP LLM Library**, addresses the gap by providing a small, modular C++17 static library that wraps the `llama.cpp` C API behind a clean, game-friendly `LLMWrapper` class. The library is designed to be consumed as a git submodule by a parent game project (e.g., an SFML-based game), keeping LLM inference concerns cleanly separated from game logic.

### 2.2 Problem Domain

The problem domain sits at the intersection of **game development** and **natural language processing (NLP) / large language model inference**.

Key domain concepts include:

- **GGUF model files** — a binary format for storing quantised LLM weights, designed for efficient loading and inference on consumer hardware [1].
- **Tokenisation** — the process of converting a text prompt into a sequence of integer token IDs that the model understands, and the reverse process (detokenisation) of converting generated token IDs back into human-readable text.
- **Autoregressive generation** — the inference loop in which the model predicts one token at a time, feeding each predicted token back as input for the next step.
- **KV cache** — an internal data structure that stores intermediate attention computations; must be managed (cleared or retained) between generation calls.
- **Quantisation** — a technique for reducing model precision (e.g., from 16-bit floats to 4-bit integers) to decrease memory usage and increase throughput on CPUs, making local inference viable on consumer hardware.
- **Sampling strategies** — algorithms that decide which token to select from the model's predicted probability distribution (e.g., greedy, top-k, top-p, temperature scaling).

The intended users of this library are **game developers** integrating local text generation into C++ game projects. They should be able to load a model file and generate text with minimal API surface, without needing deep knowledge of LLM internals.

### 2.3 Feasibility Study

#### Technical Feasibility

| Factor | Assessment |
|---|---|
| **Language and toolchain** | C++17 is widely supported by modern compilers (MSVC, GCC, Clang). CMake is the de-facto cross-platform build system for C++ projects. Both are mature and stable. |
| **Third-party dependency** | `llama.cpp` is an actively maintained, well-documented open-source project with thousands of contributors. It is designed to be embedded as a library and exposes a stable C API. |
| **Model availability** | Quantised GGUF models (e.g., Llama 3.2 1B Q4_K_M at ~0.75 GB) are freely available on Hugging Face and fit comfortably in consumer RAM. |
| **Performance** | CPU-only inference of a 1-billion-parameter Q4 model can generate tokens at several tokens per second on modern consumer hardware, which is acceptable for non-real-time text generation (e.g., dialogue that appears in a text box). |
| **Integration model** | Delivering the library as a git submodule with a CMake static-library target is a standard, well-understood pattern in C++ game development. |

#### Operational Feasibility

The library is designed to be self-contained: the game developer clones the submodule, links the CMake target, and calls two functions (`LoadModel`, `Generate`). No external servers, accounts, or API keys are required at runtime. A Python helper script is provided for the one-time model download step.

#### Economic Feasibility

All dependencies (compiler, CMake, `llama.cpp`, model files) are freely available. There are no runtime costs since inference runs locally. The quantised model fits in standard consumer RAM (8 GB+), so no specialised hardware is required.

### 2.4 Literature Survey and Research

#### 2.4.1 Local LLM Inference

The rise of quantised model formats has made running billion-parameter language models on consumer CPUs practical. Gerganov's `llama.cpp` project [1] demonstrated that 4-bit quantised Llama models can run inference on commodity hardware with no GPU, achieving usable token generation rates. The GGUF format [2] succeeded earlier formats (GGML, GGJT) to provide a single-file, self-describing container for quantised weights and metadata.

#### 2.4.2 LLMs in Games

Academic and industry research has explored using LLMs for NPC dialogue. Generative Agents (Park et al., 2023) [3] demonstrated that LLM-driven agents can produce believable social behaviour in a sandbox game environment. While that work used cloud APIs, it established the viability of LLM-generated dialogue for games. The key insight for this project is that even small models (1B parameters) can produce acceptable short-form text (single dialogue lines, lore snippets) when given well-crafted prompts.

#### 2.4.3 Quantisation Techniques

Quantisation reduces model memory footprint and increases throughput. Common approaches include GPTQ [4] (post-training quantisation to 4-bit) and the `Q4_K_M` quantisation scheme used in `llama.cpp`, which applies mixed-precision 4-bit quantisation with per-block scaling factors [1]. The Q4_K_M variant balances size, speed, and output quality, making it suitable for real-time game applications.

#### 2.4.4 RAII and Safe C++ Wrappers

Modern C++ best practice recommends RAII (Resource Acquisition Is Initialisation) for managing resources with manual lifetimes [5]. The C++ Core Guidelines [6] recommend wrapping owning raw pointers in `std::unique_ptr` with custom deleters when interfacing with C APIs. This project follows that guidance by wrapping `llama_model*`, `llama_context*`, and `llama_sampler*` in RAII smart pointers.

#### 2.4.5 CMake and Modular C++ Libraries

The CMake build system [7] is the standard for cross-platform C++ builds. Delivering a library as a static CMake target that consumers include via `add_subdirectory()` is a widely adopted pattern (used by projects such as SDL, GLFW, and `llama.cpp` itself). This approach handles transitive include paths and link dependencies automatically.

### 2.5 System Overview

The FYP LLM Library is a **static C++ library** (`LLMLib`) that provides local LLM inference through a minimal API. The major components are:

```
┌─────────────────────────────────────────────────────────┐
│                    Game Application                      │
│         (e.g., SFML game — separate repository)          │
└───────────────────────┬─────────────────────────────────┘
                        │ #include "LLMWrapper.h"
                        │ target_link_libraries(... LLMLib)
                        ▼
┌─────────────────────────────────────────────────────────┐
│                      LLMLib                              │
│  ┌───────────────────────────────────────────────────┐  │
│  │ LLMWrapper                                        │  │
│  │  • LoadModel(path) → bool                         │  │
│  │  • UnloadModel()                                  │  │
│  │  • Generate(prompt) → string                      │  │
│  │                                                   │  │
│  │  Internal helpers:                                │  │
│  │  • EnsureLlamaBackendInit()                       │  │
│  │  • PickThreadCount()                              │  │
│  │  • DecodeTokensInChunks()                         │  │
│  │  • DecodeSingleToken()                            │  │
│  │  • TokenToString()                                │  │
│  └───────────────────────────────────────────────────┘  │
└───────────────────────┬─────────────────────────────────┘
                        │ links against
                        ▼
┌─────────────────────────────────────────────────────────┐
│               external/llama.cpp                         │
│  (Third-party inference engine — git submodule)          │
└─────────────────────────────────────────────────────────┘
                        │ loads at runtime
                        ▼
┌─────────────────────────────────────────────────────────┐
│        resources/downloaded_resources/*.gguf              │
│     (Quantised model file — downloaded separately)       │
└─────────────────────────────────────────────────────────┘
```

Additionally, the repository includes:

- **`LLMTest`** — a console smoke-test executable that loads the default model and generates a single line, verifying that the library functions correctly.
- **`resources/scripts/download_model.py`** — a Python utility that downloads the default GGUF model from Hugging Face.

### 2.6 Functional Requirements

| ID | Requirement | Priority | Description |
|---|---|---|---|
| **FR-01** | Load a GGUF model | Must Have | The system shall load a GGUF model file from a given filesystem path into memory, initialising the model, vocabulary, and inference context. On failure, the function shall return `false` and leave the wrapper in an unloaded state. |
| **FR-02** | Generate text from a prompt | Must Have | The system shall accept a text prompt string and return a generated text response string. Generation shall use the currently loaded model. If no model is loaded, the function shall return an empty string. |
| **FR-03** | Unload model resources | Must Have | The system shall release all model, context, and vocabulary resources on demand via `UnloadModel()`, and automatically on wrapper destruction (RAII). |
| **FR-04** | Automatic backend initialisation | Must Have | The system shall ensure the `llama.cpp` backend is initialised exactly once per process, regardless of how many `LLMWrapper` instances are created, and shall register backend cleanup at process exit. |
| **FR-05** | Stateless generation calls | Must Have | Each call to `Generate()` shall clear the KV cache before processing the prompt, ensuring that successive calls are independent and do not carry over state from previous generations. |
| **FR-06** | Chunked prompt decoding | Must Have | The system shall decode prompt tokens in batches no larger than the context's configured batch size, to avoid exceeding internal buffer limits for long prompts. |
| **FR-07** | End-of-sequence detection | Must Have | The generation loop shall terminate when the model produces an end-of-generation token, preventing runaway output. |
| **FR-08** | Configurable maximum token limit | Should Have | The system shall enforce a configurable maximum number of newly generated tokens (default: 128), capped by the remaining context window. |
| **FR-09** | Console smoke test | Should Have | The repository shall include a test executable (`LLMTest`) that loads the default model and generates a response, printing the prompt and output to the console. |
| **FR-10** | Model download utility | Should Have | The repository shall include a Python script that downloads the default GGUF model file from Hugging Face into the expected directory. |

### 2.7 Non-Functional Requirements

| ID | Requirement | Category | Description |
|---|---|---|---|
| **NFR-01** | Performance | Efficiency | Text generation shall complete within a time frame acceptable for non-real-time game dialogue (order of seconds, not milliseconds). CPU-only inference of the default 1B Q4 model should achieve multiple tokens per second on a modern consumer CPU. |
| **NFR-02** | Memory usage | Efficiency | The loaded model and inference context shall fit within 2 GB of RAM for the default Q4_K_M 1B model, making it suitable for machines with 8 GB+ total RAM. |
| **NFR-03** | Thread safety (backend init) | Reliability | Backend initialisation shall be thread-safe via `std::once_flag`, supporting scenarios where multiple threads may attempt to create `LLMWrapper` instances. |
| **NFR-04** | Resource safety | Reliability | All dynamically allocated `llama.cpp` resources shall be managed via RAII (smart pointers with custom deleters), preventing memory and resource leaks on normal and exceptional code paths. |
| **NFR-05** | Portability | Portability | The library shall build on Windows (MSVC / Visual Studio 2026) using the provided CMake presets. The codebase uses standard C++17 and CMake, making it portable to other platforms with minimal changes. |
| **NFR-06** | Modularity | Maintainability | The library shall be consumable as a git submodule. A consuming project needs only `add_subdirectory()` and `target_link_libraries()` to integrate it. |
| **NFR-07** | Minimal API surface | Usability | The public API shall expose no more than three primary functions (`LoadModel`, `UnloadModel`, `Generate`) to minimise the learning curve for game developers. |
| **NFR-08** | Deterministic output (greedy) | Correctness | When using the default greedy sampler, the same prompt on the same model shall produce the same output, aiding testing and debugging. |
| **NFR-09** | Offline operation | Availability | The library shall not require any network connection at runtime. Model files are downloaded once during setup. |

### 2.8 User Interface Specification

This library is a **programmatic API**, not an end-user-facing application. The "user interface" is the C++ API consumed by game developers. The interaction points are:

#### 2.8.1 C++ API Interface

```cpp
#include "LLMWrapper.h"

// 1. Create a wrapper instance
LLMWrapper llm;

// 2. Load a GGUF model file
bool success = llm.LoadModel("path/to/model.gguf");

// 3. Generate text from a prompt
std::string response = llm.Generate("Say hello in a cyberpunk style");

// 4. (Optional) Unload explicitly, or let the destructor handle it
llm.UnloadModel();
```

#### 2.8.2 Console Smoke Test (`LLMTest`)

The `LLMTest` executable demonstrates end-to-end usage. Below is a representation of its expected console output:

```
sizeof(void*): 4
file_size: 789789696 bytes
llama_model_loader: loaded meta data with 33 key-value pairs ...
...
Prompt: Generate one short Sci-Fi NPC dialogue line
Response: "The stars are whispering again, traveller. You'd best listen."
```

#### 2.8.3 Model Download Script

The Python download script is run once during project setup:

```
> cd resources
> python scripts/download_model.py
Download complete.
File saved to: ./downloaded_resources/Llama-3.2-1B-Instruct-Q4_K_M.gguf
```

### 2.9 Navigation and Usage Flow

The following sequence describes the typical workflow for a developer integrating and using the library:

```
┌──────────────────────────────────────────┐
│  1. Clone repo with submodules           │
│     git clone --recurse-submodules ...   │
└─────────────────┬────────────────────────┘
                  ▼
┌──────────────────────────────────────────┐
│  2. Download model (one-time setup)      │
│     cd resources                         │
│     python scripts/download_model.py     │
└─────────────────┬────────────────────────┘
                  ▼
┌──────────────────────────────────────────┐
│  3. Configure and build with CMake       │
│     cmake --preset x86_debug             │
│     cmake --build --preset x86_debug     │
└─────────────────┬────────────────────────┘
                  ▼
┌──────────────────────────────────────────┐
│  4. Run LLMTest to verify                │
│     ./out/build/x86_debug/bin/LLMTest    │
└─────────────────┬────────────────────────┘
                  ▼
┌──────────────────────────────────────────┐
│  5. Integrate into game project          │
│     add_subdirectory(external/fyp-llm-library) │
│     target_link_libraries(Game LLMLib)   │
└─────────────────┬────────────────────────┘
                  ▼
┌──────────────────────────────────────────┐
│  6. Use in game code                     │
│     LLMWrapper llm;                      │
│     llm.LoadModel("model.gguf");         │
│     auto text = llm.Generate(prompt);    │
└──────────────────────────────────────────┘
```

#### Generation Call Flow (Internal)

When `Generate()` is called, the following internal sequence executes:

1. **Validate state** — ensure model, context, and vocab are loaded.
2. **Clear KV cache** — reset model memory for a fresh generation.
3. **Tokenise prompt** — convert the input string to token IDs.
4. **Decode prompt** — process prompt tokens in batches to prime the model.
5. **Sampling loop** — repeatedly sample the next token, check for end-of-sequence, detokenise, and decode the new token into the context.
6. **Return response** — concatenate all detokenised pieces into the final output string.

### 2.10 System Constraints and Limitations

| Constraint | Details |
|---|---|
| **Single-threaded generation** | Only one generation can run at a time per `LLMWrapper` instance. The `Generate()` call blocks until completion. |
| **Stateless calls** | The KV cache is cleared on every `Generate()` call; there is no multi-turn conversation support. |
| **Greedy sampling only** | The current implementation uses greedy (argmax) sampling. Temperature, top-k, and top-p are not yet exposed. |
| **CPU-only by default** | GPU layer offloading is disabled (`n_gpu_layers = 0`) to maximise compatibility. Users must modify source to enable GPU. |
| **No mobile/web support** | The library targets desktop platforms (Windows primarily, portable to Linux/macOS). |
| **Model size** | The default context window is 512 tokens with 128 max new tokens. Prompts exceeding the context window will be truncated. |
| **No streaming** | Generated text is returned as a complete string; there is no token-by-token callback mechanism. |

### 2.11 Critical Analysis and Conclusions

#### Strengths

1. **Clean separation of concerns** — By isolating LLM inference into a standalone library consumed via git submodule, the game project remains uncluttered by inference engine complexity. Changes to the LLM layer do not require modifications to game code, and vice versa.

2. **Minimal API surface** — The three-function public API (`LoadModel`, `UnloadModel`, `Generate`) significantly lowers the barrier to entry for game developers who may not have NLP expertise. This follows the principle of least astonishment and makes the library easy to learn, use, and test.

3. **Robust resource management** — The use of `std::unique_ptr` with custom deleters and `std::shared_ptr` with a no-op deleter for the vocabulary pointer ensures that `llama.cpp`'s manually-managed C resources are freed correctly in all code paths, including early returns and exceptions. This is a well-established modern C++ pattern that prevents resource leaks.

4. **Thread-safe initialisation** — The `std::once_flag` / `std::call_once` pattern for backend initialisation is a proven, standards-compliant approach that avoids double-initialisation bugs in multi-threaded game engines.

5. **Offline operation** — Running inference locally eliminates runtime network dependencies, per-request costs, and privacy concerns — all critical requirements for single-player games.

6. **Practical model choice** — Using a 1B-parameter Q4_K_M quantised model strikes a balance between output quality and resource consumption. The model fits comfortably in consumer RAM and generates tokens at acceptable speeds on CPU.

#### Limitations and Areas for Improvement

1. **Synchronous blocking calls** — The `Generate()` function blocks the calling thread. In a game with a 60 FPS render loop, this would cause visible frame drops. A future version should provide an asynchronous interface (e.g., a background thread with a callback or future-based API).

2. **No conversation context** — Clearing the KV cache on every call prevents multi-turn dialogue. For richer NPC interactions, the library would benefit from an optional "session" mode that retains context between calls.

3. **Limited sampling options** — Greedy sampling produces deterministic but often repetitive output. Exposing temperature, top-k, and top-p parameters would allow game designers to tune the creativity and variability of generated text.

4. **Hardcoded configuration** — Parameters such as context size (512), batch size (64), max tokens (128), and thread count are set in code. These should be configurable via the public API or a configuration struct.

5. **No streaming support** — Some game UIs display dialogue character-by-character. A streaming callback would enable this pattern without waiting for the full response.

6. **Platform-specific presets** — The CMake presets currently target Visual Studio 2026 on Windows. While the underlying code is portable C++17, extending the presets to cover GCC/Clang and Linux/macOS would broaden usability.

#### Conclusion

The FYP LLM Library successfully meets its core objective: providing a minimal, safe, and modular C++ wrapper for local LLM inference that can be integrated into a game project with minimal effort. The design decisions — RAII resource management, thread-safe initialisation, stateless generation, and submodule-based distribution — are well-grounded in modern C++ best practices and game development conventions.

The library is well-suited for its intended use case of generating short-form NPC dialogue and lore text in offline, single-player games. The identified limitations (synchronous API, greedy-only sampling, no streaming) represent clear, scoped extension points for future development rather than fundamental design flaws. The architecture is intentionally simple and extensible, providing a solid foundation upon which more sophisticated features can be built.

---

## 3. References

[1] Gerganov, G. (2023). llama.cpp — LLM inference in C/C++. [Online]. (https://github.com/ggerganov/llama.cpp). (Accessed 10 April 2026).

[2] Gerganov, G. (2023, August 21). GGUF specification. [Online]. (https://github.com/ggerganov/ggml/blob/master/docs/gguf.md). (Accessed 10 April 2026).

[3] Park, J.S., et al. (2023). Generative Agents: Interactive Simulacra of Human Behavior. In *Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology (UIST '23)*. Association for Computing Machinery.

[4] Frantar, E., et al. (2023). GPTQ: Accurate Post-Training Quantization for Generative Pre-Trained Transformers. In *Proceedings of the International Conference on Learning Representations (ICLR 2023)*.

[5] Stroustrup, B. (2013). The C++ Programming Language. 4th ed. Upper Saddle River: Addison-Wesley.

[6] Stroustrup, B. and Sutter, H. (2015). C++ Core Guidelines. [Online]. (https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines). (Accessed 10 April 2026).

[7] Kitware, Inc. (2024). CMake Documentation. [Online]. (https://cmake.org/documentation/). (Accessed 10 April 2026).

[8] Hugging Face. (2024). Hugging Face Hub Documentation. [Online]. (https://huggingface.co/docs/hub/). (Accessed 10 April 2026).

[9] Touvron, H., et al. (2023). LLaMA: Open and Efficient Foundation Language Models. *arXiv preprint arXiv:2302.13971*.

[10] Meta AI. (2024). Llama 3.2 Collection. [Online]. (https://huggingface.co/collections/meta-llama/llama-32-collection). (Accessed 10 April 2026).
