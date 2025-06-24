// Build information for BitNet WASM

extern "C" {
    int LLAMA_BUILD_NUMBER = 1;
    const char* LLAMA_COMMIT = "bitnet-wasm";
    const char* LLAMA_COMPILER = "emcc";
    const char* LLAMA_BUILD_TARGET = "wasm32";
}
