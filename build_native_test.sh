#!/bin/bash

# Build native inference test for BitNet-WASM
# This builds the inference test as a native executable for testing

cd "$(dirname "$0")"

echo "Building native BitNet inference test..."

# Define source files and include directories
BITNET_SOURCES="src/bitnet_minimal.cpp src/bitnet_inference_test.cpp src/bitnet_inference.cpp"
INCLUDE_DIRS="-Iinclude -I3rdparty/BitNet/include"

# Output executable name
OUTPUT_FILE="bitnet_inference_test"

# Compiler flags
CXX_FLAGS="-O2 -std=c++11 -pthread"

# Prepare bitnet-lut-kernels.h by copying a preset one (using 3B preset for 2B model)
PRESET_KERNEL_HEADER="preset_kernels/bitnet_b1_58-3B/bitnet-lut-kernels-tl1.h"
TARGET_KERNEL_HEADER="include/bitnet-lut-kernels.h"

if [ ! -f "$PRESET_KERNEL_HEADER" ]; then
    echo "Error: Preset kernel header $PRESET_KERNEL_HEADER not found."
    exit 1
fi

echo "Copying $PRESET_KERNEL_HEADER to $TARGET_KERNEL_HEADER"
cp "$PRESET_KERNEL_HEADER" "$TARGET_KERNEL_HEADER"

echo "Compiling native test..."
echo "Executing: g++ $CXX_FLAGS $INCLUDE_DIRS $BITNET_SOURCES -o $OUTPUT_FILE"
g++ $CXX_FLAGS $INCLUDE_DIRS $BITNET_SOURCES -o $OUTPUT_FILE

if [ $? -eq 0 ] && [ -f "$OUTPUT_FILE" ]; then
    echo "Native build successful!"
    echo "Output file: $OUTPUT_FILE"
    echo ""
    echo "To test the inference, run:"
    echo "./$OUTPUT_FILE models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \"Hello\""
else
    echo "Native build failed. Please check the output above."
    exit 1
fi

echo "Native build script finished."