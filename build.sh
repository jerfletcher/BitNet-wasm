#!/bin/bash

# Navigate to the script's directory to ensure relative paths work
cd "$(dirname "$0")"

echo "Building BitNet-WASM..."

# Define source files and include directories using 3rdparty sources
BITNET_SOURCES="src/bitnet_inference.cpp 3rdparty/BitNet/src/ggml-bitnet-lut.cpp 3rdparty/BitNet/src/ggml-bitnet-mad.cpp 3rdparty/llama.cpp/ggml/src/ggml.c 3rdparty/llama.cpp/ggml/src/ggml-quants.c 3rdparty/llama.cpp/ggml/src/ggml-backend.cpp 3rdparty/llama.cpp/ggml/src/ggml-alloc.c 3rdparty/llama.cpp/src/llama.cpp 3rdparty/llama.cpp/src/llama-vocab.cpp 3rdparty/llama.cpp/src/llama-sampling.cpp 3rdparty/llama.cpp/src/llama-grammar.cpp 3rdparty/llama.cpp/src/unicode.cpp 3rdparty/llama.cpp/src/unicode-data.cpp 3rdparty/llama.cpp/common/common.cpp 3rdparty/llama.cpp/common/sampling.cpp 3rdparty/llama.cpp/common/arg.cpp 3rdparty/llama.cpp/common/log.cpp"
INCLUDE_DIRS="-I3rdparty/BitNet/include -I3rdparty/llama.cpp/ggml/include -I3rdparty/llama.cpp/include -I3rdparty/llama.cpp/ggml/src -I3rdparty/llama.cpp/common"

# Define compilation flags for BitNet - enable ARM TL1 for now since that seems to be the most complete
COMPILATION_DEFINES="-DGGML_USE_BITNET=1 -DNDEBUG=1 -DGGML_BITNET_ARM_TL1=1"

# Output WASM file name
OUTPUT_FILE="bitnet.wasm"
OUTPUT_JS_FILE="bitnet.js" # Emscripten generates a JS loader

# Emscripten compiler flags with assertions enabled for debugging
EMCC_FLAGS="-O1 -s WASM=1 -s MODULARIZE=1 -s EXPORT_ES6=1 -s EXPORTED_RUNTIME_METHODS=['ccall','cwrap','HEAPU8','HEAPU32','HEAPF32','HEAP8','HEAP32','lengthBytesUTF8','stringToUTF8','UTF8ToString'] -s EXPORTED_FUNCTIONS=['_malloc','_free'] -s ALLOW_MEMORY_GROWTH=1 -s INITIAL_MEMORY=16MB -s MAXIMUM_MEMORY=4GB -s STACK_SIZE=1MB -s DISABLE_EXCEPTION_CATCHING=0 -s ASSERTIONS=1 -s SAFE_HEAP=0 -s USE_PTHREADS=0 -s PTHREAD_POOL_SIZE=0"

# Prepare bitnet-lut-kernels.h by copying a preset one (using 3B preset for 2B model)
PRESET_KERNEL_HEADER="3rdparty/BitNet/preset_kernels/bitnet_b1_58-3B/bitnet-lut-kernels-tl1.h"
TARGET_KERNEL_HEADER="3rdparty/BitNet/include/bitnet-lut-kernels.h"

if [ ! -f "$PRESET_KERNEL_HEADER" ]; then
    echo "Error: Preset kernel header $PRESET_KERNEL_HEADER not found."
    exit 1
fi

echo "Copying $PRESET_KERNEL_HEADER to $TARGET_KERNEL_HEADER"
cp "$PRESET_KERNEL_HEADER" "$TARGET_KERNEL_HEADER"

echo "Using real GGML and BitNet sources from 3rdparty folders..."

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
