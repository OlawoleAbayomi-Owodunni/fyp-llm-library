// LLMTest.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <filesystem>
#include "LLMWrapper.h"

#ifndef FYP_SOURCE_DIR
#define FYP_SOURCE_DIR "."
#endif


int main()
{
	const std::filesystem::path modelPath = std::filesystem::path(FYP_SOURCE_DIR) / "resources" / "downloaded_resources" / "Llama-3.2-1B-Instruct-Q4_K_M.gguf";

	LLMWrapper llm;
	
	if (!llm.LoadModel(modelPath.string())) {
		std::cerr << "Failed to load model from path: " << modelPath << std::endl;
		return 1;
	}

	const std::string prompt = "Generate one short Sci-Fi NPC dialogue line";
	std::string response = llm.Generate(prompt);
	std::cout << "Prompt: " << prompt << "\nResponse: " << response << std::endl;
	return 0;
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
