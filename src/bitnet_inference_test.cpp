#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>

// Minimal BitNet function declarations for testing
extern "C" {
void ggml_bitnet_init(void);
void ggml_bitnet_free(void);
}
// Shared BitNet model and GGUF structures
#include "bitnet_inference.h"


// BitNet inference function
int bitnet_inference_test(const char* model_path, const char* prompt) {
    printf("[bitnet_inference_test] Starting BitNet inference test\n");
    printf("[bitnet_inference_test] Model path: %s\n", model_path);
    printf("[bitnet_inference_test] Prompt: %s\n", prompt);
    
    // Initialize BitNet
    ggml_bitnet_init();
    
    // Parse GGUF header (skip, handled by real loader below)
    
    // Tokenize input
    std::vector<int32_t> tokens = tokenize(prompt);
    printf("[bitnet_inference_test] Input tokens (%zu): ", tokens.size());
    for (size_t i = 0; i < std::min(tokens.size(), (size_t)10); ++i) {
        printf("%d ", tokens[i]);
    }
    if (tokens.size() > 10) printf("...");
    printf("\n");

    // Load model using parse_gguf_file from bitnet_inference.cpp
    std::ifstream model_file(model_path, std::ios::binary | std::ios::ate);
    if (!model_file.is_open()) {
        printf("[bitnet_inference_test] Failed to open model file for real loading\n");
        ggml_bitnet_free();
        return -1;
    }
    std::streamsize model_size = model_file.tellg();
    model_file.seekg(0, std::ios::beg);
    std::vector<uint8_t> model_data(model_size);
    if (!model_file.read(reinterpret_cast<char*>(model_data.data()), model_size)) {
        printf("[bitnet_inference_test] Failed to read model file\n");
        ggml_bitnet_free();
        return -1;
    }
    if (!parse_gguf_file(model_data.data(), model_data.size(), g_model)) {
        printf("[bitnet_inference_test] Failed to load model with real loader\n");
        ggml_bitnet_free();
        return -1;
    }

    // Real inference using bitnet_inference from bitnet_inference.cpp
    std::vector<int32_t> input_tokens(tokens.begin(), tokens.end());
    std::vector<int32_t> output_tokens = bitnet_inference(input_tokens, 32);
    std::string output = detokenize(output_tokens);
    printf("[bitnet_inference_test] Generated output: %s\n", output.c_str());
    
    // Clean up
    ggml_bitnet_free();
    
    printf("[bitnet_inference_test] Test completed successfully\n");
    return 0;
}

// Export for WASM
int bitnet_wasm_infer(const void* model_data, size_t model_size, const char* prompt, char* output, size_t output_size) {
    // For WASM, we need to handle the model data in memory
    // This is a simplified version that would need proper implementation
    printf("[bitnet_wasm_infer] WASM inference called\n");
    printf("[bitnet_wasm_infer] Model size: %zu bytes\n", model_size);
    printf("[bitnet_wasm_infer] Prompt: %s\n", prompt);
    // Simulate inference
    std::string result = std::string(prompt) + " [BitNet generated text]";
    if (result.length() >= output_size) {
        return -1; // Output buffer too small
    }
    strcpy(output, result.c_str());
    return result.length();
}

// Main function for native testing
int main(int argc, char** argv) {
    if (argc != 3) {
        printf("Usage: %s <model.gguf> <prompt>\n", argv[0]);
        printf("Example: %s models/bitnet_b1_58-3B/ggml-model-i2_s.gguf \"Hello\"\n", argv[0]);
        return 1;
    }
    
    const char* model_path = argv[1];
    const char* prompt = argv[2];
    
    return bitnet_inference_test(model_path, prompt);
}