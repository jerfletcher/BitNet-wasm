#!/bin/bash

# Navigate to the script's directory to ensure relative paths work
cd "$(dirname "$0")"

echo "Building BitNet-WASM..."

# Activate Emscripten SDK environment
echo "Activating Emscripten SDK..."
source emsdk/emsdk_env.sh

# Define source files and include directories using BitNet's llama.cpp fork
BITNET_SOURCES="src/bitnet_inference.cpp src/build-info.cpp 3rdparty/BitNet/src/ggml-bitnet-lut.cpp 3rdparty/BitNet/src/ggml-bitnet-mad.cpp 3rdparty/BitNet/3rdparty/llama.cpp/ggml/src/ggml.c 3rdparty/BitNet/3rdparty/llama.cpp/ggml/src/ggml-quants.c 3rdparty/BitNet/3rdparty/llama.cpp/ggml/src/ggml-backend.cpp 3rdparty/BitNet/3rdparty/llama.cpp/ggml/src/ggml-alloc.c 3rdparty/BitNet/3rdparty/llama.cpp/src/llama.cpp 3rdparty/BitNet/3rdparty/llama.cpp/src/llama-vocab.cpp 3rdparty/BitNet/3rdparty/llama.cpp/src/llama-sampling.cpp 3rdparty/BitNet/3rdparty/llama.cpp/src/llama-grammar.cpp 3rdparty/BitNet/3rdparty/llama.cpp/src/unicode.cpp 3rdparty/BitNet/3rdparty/llama.cpp/src/unicode-data.cpp 3rdparty/BitNet/3rdparty/llama.cpp/common/common.cpp 3rdparty/BitNet/3rdparty/llama.cpp/common/sampling.cpp 3rdparty/BitNet/3rdparty/llama.cpp/common/arg.cpp 3rdparty/BitNet/3rdparty/llama.cpp/common/log.cpp"
INCLUDE_DIRS="-Iinclude -I3rdparty/BitNet/include -I3rdparty/BitNet/3rdparty/llama.cpp/ggml/include -I3rdparty/BitNet/3rdparty/llama.cpp/include -I3rdparty/BitNet/3rdparty/llama.cpp/ggml/src -I3rdparty/BitNet/3rdparty/llama.cpp/common"

# Define compilation flags for BitNet - try x86 TL2 instead of ARM TL1 for WASM compatibility
COMPILATION_DEFINES="-DGGML_USE_BITNET=1 -DNDEBUG=1 -DGGML_BITNET_X86_TL2=1 -DGGML_NO_ACCELERATE=1 -DGGML_NO_OPENMP=1 -DGGML_WASM_SINGLE_THREAD=1"

# Add warning suppressions for deprecated C++17 features in upstream code
WARNING_FLAGS="-Wno-deprecated-declarations -Wno-incompatible-pointer-types -Wno-incompatible-pointer-types-discards-qualifiers"

# Output WASM file name
OUTPUT_FILE="bitnet.wasm"
OUTPUT_JS_FILE="bitnet.js" # Emscripten generates a JS loader

# Emscripten compiler flags with assertions enabled for debugging
EMCC_FLAGS="-O0 -s WASM=1 -s MODULARIZE=1 -s EXPORT_ES6=0 -s EXPORTED_RUNTIME_METHODS=['ccall','cwrap','HEAPU8','HEAPU32','HEAPF32','HEAP8','HEAP32','lengthBytesUTF8','stringToUTF8','UTF8ToString'] -s EXPORTED_FUNCTIONS=['_malloc','_free','_bitnet_init','_bitnet_load_model_from_memory','_bitnet_run_inference_simple','_bitnet_is_model_loaded','_bitnet_get_vocab_size','_bitnet_get_embedding_dim','_bitnet_get_num_layers','_bitnet_free_model','_bitnet_cleanup'] -s ALLOW_MEMORY_GROWTH=1 -s INITIAL_MEMORY=32MB -s MAXIMUM_MEMORY=4GB -s STACK_SIZE=2MB -s DISABLE_EXCEPTION_CATCHING=0 -s ASSERTIONS=1 -s SAFE_HEAP=0 -s USE_PTHREADS=0 -s PTHREAD_POOL_SIZE=0 --bind -s ERROR_ON_UNDEFINED_SYMBOLS=0"

# Prepare bitnet-lut-kernels.h by copying a preset one to local include (using x86 TL2 for WASM)
PRESET_KERNEL_HEADER="3rdparty/BitNet/preset_kernels/bitnet_b1_58-3B/bitnet-lut-kernels-tl2.h"
LOCAL_INCLUDE_DIR="include"
TARGET_KERNEL_HEADER="$LOCAL_INCLUDE_DIR/bitnet-lut-kernels.h"

if [ ! -f "$PRESET_KERNEL_HEADER" ]; then
    echo "Error: Preset kernel header $PRESET_KERNEL_HEADER not found."
    exit 1
fi

# Create local include directory if it doesn't exist
mkdir -p "$LOCAL_INCLUDE_DIR"

# Only copy if target doesn't exist or is different
if [ ! -f "$TARGET_KERNEL_HEADER" ] || ! cmp -s "$PRESET_KERNEL_HEADER" "$TARGET_KERNEL_HEADER"; then
    echo "Copying $PRESET_KERNEL_HEADER to $TARGET_KERNEL_HEADER"
    cp "$PRESET_KERNEL_HEADER" "$TARGET_KERNEL_HEADER"
else
    echo "Kernel header already up to date"
fi

echo "Using real GGML and BitNet sources from 3rdparty folders..."

echo "Compiling with Emscripten..."
echo "Executing: emcc $EMCC_FLAGS $WARNING_FLAGS $COMPILATION_DEFINES $INCLUDE_DIRS $BITNET_SOURCES -o $OUTPUT_JS_FILE"
emcc $EMCC_FLAGS $WARNING_FLAGS $COMPILATION_DEFINES $INCLUDE_DIRS $BITNET_SOURCES -o $OUTPUT_JS_FILE > emcc_stdout.log 2> emcc_stderr.log
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
