#!/bin/bash

# Navigate to the script's directory to ensure relative paths work
cd "$(dirname "$0")"

echo "Building BitNet-WASM..."

# Define source files and include directories
BITNET_SOURCES="src/bitnet_inference.cpp src/ggml-bitnet-lut.cpp src/ggml-bitnet-mad.cpp src/ggml_stub.c"
INCLUDE_DIRS="-Iinclude -I3rdparty/BitNet/include"

# Define compilation flags for BitNet
COMPILATION_DEFINES="-DGGML_USE_BITNET=1 -DNDEBUG=1"

# Output WASM file name
OUTPUT_FILE="bitnet.wasm"
OUTPUT_JS_FILE="bitnet.js" # Emscripten generates a JS loader

# Emscripten compiler flags with assertions enabled for debugging
EMCC_FLAGS="-O1 -s WASM=1 -s MODULARIZE=1 -s EXPORT_ES6=1 -s EXPORTED_RUNTIME_METHODS=['ccall','cwrap','HEAPU8','HEAPF32','HEAP8','HEAP32','lengthBytesUTF8','stringToUTF8'] -s EXPORTED_FUNCTIONS=['_bitnet_init','_bitnet_load_model','_bitnet_inference_run','_bitnet_get_model_info','_bitnet_is_model_loaded','_bitnet_free_model','_ggml_init','_ggml_bitnet_init','_ggml_bitnet_free','_ggml_bitnet_mul_mat_task_compute','_ggml_bitnet_transform_tensor','_malloc','_free'] -s ALLOW_MEMORY_GROWTH=1 -s INITIAL_MEMORY=16MB -s MAXIMUM_MEMORY=4GB -s STACK_SIZE=1MB -s DISABLE_EXCEPTION_CATCHING=0 -s ASSERTIONS=1 -s SAFE_HEAP=0 -s USE_PTHREADS=0 -s PTHREAD_POOL_SIZE=0 --bind"

# Prepare bitnet-lut-kernels.h by copying a preset one (using 3B preset for 2B model)
PRESET_KERNEL_HEADER="preset_kernels/bitnet_b1_58-3B/bitnet-lut-kernels-tl1.h"
TARGET_KERNEL_HEADER="include/bitnet-lut-kernels.h"

if [ ! -f "$PRESET_KERNEL_HEADER" ]; then
    echo "Error: Preset kernel header $PRESET_KERNEL_HEADER not found."
    exit 1
fi

echo "Copying $PRESET_KERNEL_HEADER to $TARGET_KERNEL_HEADER"
cp "$PRESET_KERNEL_HEADER" "$TARGET_KERNEL_HEADER"

# Create comprehensive ggml stubs for BitNet
echo "Creating comprehensive ggml.h for BitNet..."
cat > include/ggml.h << 'EOF'
#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_API

// GGML tensor structure
struct ggml_tensor {
    int64_t ne[4]; // dimensions
    void * data;   // data pointer
    void * extra;  // extra data
    size_t nb[4];  // strides
    int type;      // tensor type
};

// GGML type enum including BitNet types
enum ggml_type {
    GGML_TYPE_F32 = 0,
    GGML_TYPE_F16 = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    GGML_TYPE_Q5_0 = 6,
    GGML_TYPE_Q5_1 = 7,
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q8_1 = 9,
    GGML_TYPE_Q2_K = 10,
    GGML_TYPE_Q3_K = 11,
    GGML_TYPE_Q4_K = 12,
    GGML_TYPE_Q5_K = 13,
    GGML_TYPE_Q6_K = 14,
    GGML_TYPE_Q8_K = 15,
    GGML_TYPE_I8 = 16,
    GGML_TYPE_I16 = 17,
    GGML_TYPE_I32 = 18,
    GGML_TYPE_COUNT = 19,
    GGML_TYPE_TL1 = 20,
    GGML_TYPE_TL2 = 21,
    GGML_TYPE_I2_S = 22,
};

// GGML API functions
GGML_API void ggml_init(void);
GGML_API int64_t ggml_nelements(const struct ggml_tensor * tensor);
GGML_API size_t ggml_row_size(enum ggml_type type, int64_t n);
GGML_API size_t ggml_type_size(enum ggml_type type);

#ifdef __cplusplus
}
#endif
EOF

# Create ggml-backend.h stub
echo "Creating ggml-backend.h stub..."
cat > include/ggml-backend.h << 'EOF'
#pragma once

#define GGML_API

#ifdef __cplusplus
extern "C" {
#endif

// Empty stub for ggml-backend.h

#ifdef __cplusplus
}
#endif
EOF

# Create ggml-alloc.h stub
echo "Creating ggml-alloc.h stub..."
cat > include/ggml-alloc.h << 'EOF'
#pragma once

// Empty stub for ggml-alloc.h
EOF

# Create ggml-quants.h stub
echo "Creating ggml-quants.h stub..."
cat > include/ggml-quants.h << 'EOF'
#pragma once

// Empty stub for ggml-quants.h
EOF

# Create comprehensive ggml.c implementation
echo "Creating ggml.c implementation..."
cat > src/ggml_stub.c << 'EOF'
#include "ggml.h"

void ggml_init(void) {
    // Stub implementation
}

int64_t ggml_nelements(const struct ggml_tensor * tensor) {
    if (!tensor) return 0;
    return tensor->ne[0] * tensor->ne[1] * tensor->ne[2] * tensor->ne[3];
}

size_t ggml_type_size(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32: return 4;
        case GGML_TYPE_F16: return 2;
        case GGML_TYPE_I8:  return 1;
        case GGML_TYPE_I16: return 2;
        case GGML_TYPE_I32: return 4;
        case GGML_TYPE_TL1: return 1;
        case GGML_TYPE_TL2: return 1;
        case GGML_TYPE_I2_S: return 1;
        default:            return 4; // Default to 4 bytes
    }
}

size_t ggml_row_size(enum ggml_type type, int64_t n) {
    return ggml_type_size(type) * n;
}

// BitNet stub functions 
void ggml_bitnet_init(void) {
    // Stub implementation for BitNet initialization
}

void ggml_bitnet_free(void) {
    // Stub implementation for BitNet cleanup
}

void ggml_bitnet_mul_mat_task_compute(void * src0, void * scales, void * qlut, void * lut_scales, void * lut_biases, void * dst, int n, int k, int m, int bits) {
    // Stub implementation for BitNet matrix multiplication
}

void ggml_bitnet_transform_tensor(struct ggml_tensor * tensor) {
    // Stub implementation for BitNet tensor transformation
}
EOF

echo "Compiling with Emscripten..."
echo "Executing: emcc $EMCC_FLAGS $COMPILATION_DEFINES $INCLUDE_DIRS $BITNET_SOURCES -o $OUTPUT_JS_FILE"
emcc $EMCC_FLAGS $COMPILATION_DEFINES $INCLUDE_DIRS $BITNET_SOURCES -o $OUTPUT_JS_FILE > emcc_stdout.log 2> emcc_stderr.log
EMCC_EXIT_CODE=$?
echo "emcc exit code: $EMCC_EXIT_CODE"
echo "--- emcc stderr ---"
cat emcc_stderr.log
echo "--- end emcc stderr ---"
echo "--- emcc stdout ---"
cat emcc_stdout.log
echo "--- end emcc stdout ---"

if [ $EMCC_EXIT_CODE -eq 0 ] && [ -f "$OUTPUT_FILE" ] && [ -f "$OUTPUT_JS_FILE" ]; then
    echo "Build successful!"
    echo "Output files: $OUTPUT_JS_FILE, $OUTPUT_FILE"
else
    echo "Build failed. Please check the output from emcc."
    exit 1
fi

echo "Build script finished."
