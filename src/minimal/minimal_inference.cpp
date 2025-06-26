#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <memory>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

// Use only standard llama.cpp headers (no BitNet)
#include "llama.h"
#include "common.h"
#include "sampling.h"

static struct llama_model* g_model = nullptr;
static struct llama_context* g_context = nullptr;
static struct common_sampler* g_sampler = nullptr;

extern "C" {
    __attribute__((visibility("default"))) EMSCRIPTEN_KEEPALIVE void minimal_init() {
        llama_backend_init();
        llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);
        std::cout << "Minimal llama.cpp initialized" << std::endl;
    }
    
    __attribute__((visibility("default"))) EMSCRIPTEN_KEEPALIVE int minimal_load_model(const uint8_t* data, size_t size) {
        const char* temp_path = "/tmp/model.gguf";
        std::ofstream file(temp_path, std::ios::binary);
        file.write(reinterpret_cast<const char*>(data), size);
        file.close();
        
        // Standard llama.cpp model params without BitNet
        llama_model_params model_params = llama_model_default_params();
        model_params.use_mmap = false;
        model_params.use_mlock = false;
        model_params.n_gpu_layers = 0;
        
        g_model = llama_load_model_from_file(temp_path, model_params);
        if (!g_model) {
            std::cerr << "Failed to load model" << std::endl;
            return 0;
        }
        
        // Standard context params
        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = 512;
        ctx_params.n_batch = 32;
        
        g_context = llama_new_context_with_model(g_model, ctx_params);
        if (!g_context) {
            std::cerr << "Failed to create context" << std::endl;
            return 0;
        }
        
        std::cout << "Model loaded successfully (standard llama.cpp)" << std::endl;
        return 1;
    }
    
    __attribute__((visibility("default"))) EMSCRIPTEN_KEEPALIVE int minimal_load_model_from_memory(uintptr_t data_ptr, size_t size) {
        return minimal_load_model(reinterpret_cast<const uint8_t*>(data_ptr), size);
    }
    
    __attribute__((visibility("default"))) EMSCRIPTEN_KEEPALIVE int minimal_is_loaded() {
        return (g_model && g_context) ? 1 : 0;
    }
}
