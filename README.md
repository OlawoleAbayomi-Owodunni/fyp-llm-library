# FYP LLM Library (Local Inference Wrapper)

A small C++17 static library that wraps the `llama.cpp` C API behind a simple `LLMWrapper` class so a game (e.g., an SFML project) can run **local LLM inference** for NPC dialogue / lore generation without calling a remote service.

This repository is intended to be modular (usable as a submodule) and keep LLM inference concerns separate from the main game repository.

## What this repo is

- A CMake-built static library target: `LLMLib`
- A minimal wrapper API: `LLMWrapper::LoadModel()` and `LLMWrapper::Generate()`
- A small console test executable: `LLMTest`
- A Python script to download a GGUF model from Hugging Face
- A vendored dependency as a git submodule: `external/llama.cpp`

## What this repo is not

- Not an online service / REST API
- Not a full chat system (no conversation history management; `Generate()` clears KV cache each call)
- Not a token streaming UI (generation is returned as one `std::string`)
- Not an “engine plugin” (Unity/Unreal/etc.) out of the box

## Typical use-cases

Good fit:
- Single-player games that want local text generation for NPC barks, ambient lore, item descriptions
- Prototyping offline generation where determinism and networking are not required
- Tooling apps that want an embedded LLM without external dependencies at runtime (besides the model file)

Poor fit / not supported by design:
- Mobile / web builds
- High-throughput server inference
- Very large models that exceed typical consumer RAM/VRAM
- Multiple concurrent generations from one wrapper instance (this wrapper uses a single `llama_context`)

## Repository layout

- `llm/`
  - `LLMWrapper.h`, `LLMWrapper.cpp`: the public wrapper API and implementation
- `resources/`
  - `scripts/requirements.txt`: Python dependency list
  - `scripts/download_model.py`: downloads a GGUF model
  - `downloaded_resources/` *(created locally)*: downloaded models (ignored by git)
- `external/llama.cpp/`: `llama.cpp` git submodule (third-party)
- `vs/LLMLib/`: Visual Studio solution/projects for convenience
  - `LLMLib`: static library project
  - `LLMTest`: console test app project

## Dependencies

Build-time:
- C++17 compiler
- CMake (3.21+) and Ninja (if using CMake presets)
- Python 3.x (only for model download)

Third-party:
- `external/llama.cpp` (git submodule)

Model:
- By default this repo uses the GGUF file `Llama-3.2-1B-Instruct-Q4_K_M.gguf` (downloaded from the Hugging Face repo configured in `resources/scripts/download_model.py`).

## Quick start (Windows)

### 1) Clone with submodules

If you are cloning fresh:

```powershell
git clone --recurse-submodules <THIS_REPO_URL>
cd fyp-llm-library
```

If you already cloned:

```powershell
git submodule update --init --recursive
```

### 2) Create and activate the Python virtual environment

From the repository root:

```powershell
py -m venv .venv
.\.venv\Scripts\activate
pip install -r resources\scripts\requirements.txt
```

### 3) Download the model into the expected folder

`LLMTest` expects the model at `resources/downloaded_resources/Llama-3.2-1B-Instruct-Q4_K_M.gguf`.

The provided script downloads to `./downloaded_resources` relative to your *current working directory*, so run it from `resources/`:

```powershell
cd resources
python .\scripts\download_model.py
cd ..
```

If Hugging Face requires authentication for the model repo, log in first (one-time) with `huggingface-cli login` or set `HUGGINGFACE_HUB_TOKEN`.

### 4.1) Build and run (CMake presets)

This repo ships a root `CMakePresets.json` with `Ninja` presets.

Configure:

```powershell
cmake --preset x86_debug
```

Build:

```powershell
cmake --build --preset x86_debug
```

Run the test executable:

```powershell
.\out\build\x86_debug\bin\LLMTest.exe
```

Notes:
- The root `CMakePresets.json` currently contains **hard-coded MSVC tool paths** as I haven't yet made that part virtual. On another machine you will likely need to edit those, or create `CMakeUserPresets.json` (which is git-ignored) and override the compiler paths there.

### 4.2) Build and run (Visual Studio solution)

Open `vs/LLMLib/LLMLib.slnx` in Visual Studio.

- Build the solution (`LLMLib` + `LLMTest`)
- Set `LLMTest` as the startup project
- Run (Ctrl+F5)

Notes:
- `LLMTest` uses a relative model path (`./resources/downloaded_resources/...`). When running from Visual Studio, ensure the **working directory is the repository root** (Project Properties → Debugging → Working Directory), or define `FYP_SOURCE_DIR` located at the top of the Wrapper.cpp file to point at the repo root.

## Using `LLMLib` in your own project

### Option A: Add as a git submodule (recommended)

```powershell
git submodule add <REPO_URL> external/fyp-llm-library
```

In your game’s `CMakeLists.txt`:

```cmake
add_subdirectory(external/fyp-llm-library)

target_link_libraries(YourGameTarget PRIVATE LLMLib)
```

In code:

```cpp
#include "LLMWrapper.h"

LLMWrapper llm;
llm.LoadModel("path/to/model.gguf");
auto text = llm.Generate("Say hello in a cyberpunk style");
```

### Option B: Build once and link the static library

- Build `LLMLib`
- Link the produced `.lib` into your project
- Add `llm/` and `external/llama.cpp/include` include paths

(Option A is simpler because it handles the `llama.cpp` dependency automatically.)

## How `LLMWrapper` works (high-level)

At a high level, a generation call follows this pipeline:

1. Ensure `llama.cpp` backend is initialized once per process
2. Load model (`llama_model`) from a GGUF file
3. Create an inference context (`llama_context`) with settings (context size, batch size, thread count)
4. When generating:
   - clear the KV cache so each call starts from a clean state
   - tokenize the prompt using the model’s vocab
   - decode the prompt tokens to prime the model state
   - repeatedly sample the next token (greedy sampler in this implementation)
   - decode each sampled token back into the context
   - detokenize sampled tokens into a final response string

For deeper implementation details and UML diagrams, see `docs/TECHNICAL_DESIGN.md`.

## Configuration knobs (current defaults)

Inside `llm/LLMWrapper.cpp`:

- Max generated tokens: `llm_maxNewTokens = 128`
- Context length: `contextParams.n_ctx = 1024`
- Prompt decode batch size: `contextParams.n_batch = 256`
- Threads: `PickThreadCount()` chooses `max(1, hardware_concurrency - 1)`
- Sampling: greedy (`llama_sampler_init_greedy()`)
- GPU offload: disabled (`modelParams.n_gpu_layers = 0`)

## Common issues / troubleshooting

- **Model file not found**: confirm `resources/downloaded_resources/<model>.gguf` exists.
- **Wrong working directory when downloading**: run the download script from `resources/` as shown above.
- **Preset compiler paths don’t match your machine**: create `CMakeUserPresets.json` and set `CMAKE_CXX_COMPILER` / friends, or delete the hard-coded entries in `CMakePresets.json`. If you're struggling to define your own presets, you can copy this readme as well as the cmakelist into chat gpt and ask it generate you one for your machine.
- **Slow generation**: this wrapper is CPU-only by default; consider enabling GPU layers via `modelParams.n_gpu_layers`. Note that if you built using MSVC, you MIGHT encounter issues with trying to use the GPU cores of your system. I did and that's why I have it use CPU cores by default.

## License / attribution

- `external/llama.cpp` is a third-party dependency with its own license.
- GGUF model files downloaded from Hugging Face have their own license/terms.
