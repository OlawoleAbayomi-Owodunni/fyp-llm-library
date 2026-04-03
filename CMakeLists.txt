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

# Locate LLMTest.cpp
set(_LLMTEST_SRC "")

set(_LLMTEST_CANDIDATES
	"${CMAKE_SOURCE_DIR}/vs/LLMLib/LLMTest/LLMTest.cpp"
	"${CMAKE_SOURCE_DIR}/LLMTest/LLMTest.cpp"
	"${CMAKE_SOURCE_DIR}/LLMTest.cpp"
	"${CMAKE_SOURCE_DIR}/vs/LLMTest/LLMTest.cpp"
)

# Check candidates in order
foreach(_cand IN LISTS _LLMTEST_CANDIDATES)
    if (EXISTS "${_cand}")
        set(LLMTEST_SRC "${_cand}")
        break()
    endif()
endforeach()

# If not found, search recursively
if (LLMTEST_SRC STREQUAL "")
    file(GLOB_RECURSE _FOUND_LLMTEST CONFIGURE_DEPENDS
        "${CMAKE_SOURCE_DIR}/**/LLMTest.cpp"
    )
    list(LENGTH _FOUND_LLMTEST _FOUND_LLMTEST_COUNT)

    if (_FOUND_LLMTEST_COUNT EQUAL 1)
        list(GET _FOUND_LLMTEST 0 LLMTEST_SRC)
    else()
        message(FATAL_ERROR
            "Could not uniquely locate LLMTest.cpp.\n"
            "Candidates checked:\n  - ${_LLMTEST_CANDIDATES}\n"
            "Recursive matches found (${_FOUND_LLMTEST_COUNT}):\n  - ${_FOUND_LLMTEST}\n"
            "Fix by setting the correct path in CMakeLists.txt."
        )
    endif()
endif()

# Create the LLMTest executable
add_executable(LLMTest "${LLMTEST_SRC}")
target_link_libraries(LLMTest PRIVATE LLMLib)
target_compile_definitions(LLMTest PRIVATE FYP_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

# Put the EXE next to llama.dll / ggml*.dll (which are under bin/)
set_target_properties(LLMTest PROPERTIES
	RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)