cmake_minimum_required(VERSION 3.21)

project(fyp_llm_library LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build Llama.cpp as a subproject (creates target: 'llama')
add_subdirectory(external/llama.cpp)

add_library(LLMLib STATIC
	llm/LLMWrapper.cpp
	llm/LLMWrapper.h
)
target_include_directories(LLMLib PUBLIC llm)
target_link_libraries(LLMLib PUBLIC llama)

# Create the LLMTest executable
add_executable(LLMTest "${CMAKE_CURRENT_SOURCE_DIR}/llm/tests/LLMTest.cpp")
target_link_libraries(LLMTest PRIVATE LLMLib)
target_compile_definitions(LLMTest PRIVATE FYP_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

# Put the EXE next to llama.dll / ggml*.dll (which are under bin/)
set_target_properties(LLMTest PROPERTIES
	RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
)