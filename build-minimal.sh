#!/bin/bash

# Test with standard llama.cpp WASM (no BitNet extensions)
echo "🔧 Building minimal WASM without BitNet extensions for testing..."

# Source emsdk if available
if [ -d "emsdk" ]; then
    source emsdk/emsdk_env.sh
    echo "✅ Emscripten environment loaded"
fi

# Create minimal test without BitNet
mkdir -p src/minimal

cat > src/minimal/minimal_inference.cpp << 'EOF'
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
EOF

# Build minimal version
echo "🏗️ Building minimal WASM..."

emcc -O3 \
    -I3rdparty/BitNet/3rdparty/llama.cpp \
    -I3rdparty/BitNet/3rdparty/llama.cpp/common \
    -I3rdparty/BitNet/3rdparty/llama.cpp/include \
    -I3rdparty/BitNet/include \
    src/minimal/minimal_inference.cpp \
    3rdparty/BitNet/3rdparty/llama.cpp/src/llama.cpp \
    3rdparty/BitNet/3rdparty/llama.cpp/src/llama-vocab.cpp \
    3rdparty/BitNet/3rdparty/llama.cpp/src/llama-grammar.cpp \
    3rdparty/BitNet/3rdparty/llama.cpp/src/llama-sampling.cpp \
    3rdparty/BitNet/3rdparty/llama.cpp/common/common.cpp \
    3rdparty/BitNet/3rdparty/llama.cpp/common/sampling.cpp \
    3rdparty/BitNet/3rdparty/llama.cpp/ggml/src/ggml.c \
    3rdparty/BitNet/3rdparty/llama.cpp/ggml/src/ggml-alloc.c \
    3rdparty/BitNet/3rdparty/llama.cpp/ggml/src/ggml-backend.c \
    3rdparty/BitNet/3rdparty/llama.cpp/ggml/src/ggml-opt.c \
    3rdparty/BitNet/3rdparty/llama.cpp/ggml/src/ggml-quants.c \
    3rdparty/BitNet/3rdparty/llama.cpp/ggml/src/ggml-threading.c \
    3rdparty/BitNet/3rdparty/llama.cpp/ggml/src/llamafile/sgemm.cpp \
    -s WASM=1 \
    -s EXPORTED_FUNCTIONS='["_malloc","_free"]' \
    -s EXPORTED_RUNTIME_METHODS='["ccall","cwrap","setValue","getValue"]' \
    -s ALLOW_MEMORY_GROWTH=1 \
    -s INITIAL_MEMORY=256MB \
    -s STACK_SIZE=8MB \
    -s ENVIRONMENT=node \
    -s MODULARIZE=1 \
    -s EXPORT_ES6=0 \
    -s EXPORT_NAME=MinimalModule \
    -s SAFE_HEAP=1 \
    -DGGML_USE_LLAMAFILE \
    -o minimal.js

if [ $? -eq 0 ]; then
    echo "✅ Minimal WASM build successful"
    echo "📁 Generated: minimal.js, minimal.wasm"
else
    echo "❌ Build failed"
    exit 1
fi
