#!/bin/bash

# BitNet-WASM Setup and Build Script
# This script installs the necessary dependencies and builds the BitNet-WASM module

set -e  # Exit on error

# Navigate to the script's directory to ensure relative paths work
cd "$(dirname "$0")"

echo "=== BitNet-WASM Setup and Build ==="
echo "This script will install Emscripten SDK and build the BitNet-WASM module."

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "Git is not installed. Installing git..."
    sudo apt-get update
    sudo apt-get install -y git
fi

# Initialize and update submodules if not already done
if [ ! -d "3rdparty/llama.cpp/.git" ] || [ ! -d "3rdparty/llama-cpp-wasm/.git" ]; then
    echo "Initializing git submodules..."
    git submodule update --init --recursive
fi

# Install Emscripten SDK if not already installed
if [ ! -d "emsdk" ]; then
    echo "Installing Emscripten SDK..."
    git clone https://github.com/emscripten-core/emsdk.git
    cd emsdk
    ./emsdk install 4.0.8
    ./emsdk activate 4.0.8
    cd ..
fi

# Source Emscripten environment
echo "Activating Emscripten environment..."
source ./emsdk/emsdk_env.sh

# Verify Emscripten is working
echo "Emscripten version:"
emcc --version

# Create models directory if it doesn't exist
if [ ! -d "models" ]; then
    echo "Creating models directory..."
    mkdir -p models
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Installing Python 3..."
    sudo apt-get update
    sudo apt-get install -y python3 python3-pip
fi

# Install Python dependencies for model downloading (optional)
echo "Installing Python dependencies for model downloading..."
pip3 install --break-system-packages huggingface_hub

# Download the base BitNet model using the official Microsoft model
MODEL_DIR="models/BitNet-b1.58-2B-4T"
MODEL_PATH="$MODEL_DIR/ggml-model-i2_s.gguf"
if [ ! -f "$MODEL_PATH" ]; then
    echo "Downloading base BitNet model (BitNet-b1.58-2B-4T) from Hugging Face..."
    # Use the official Microsoft BitNet model
    python3 -c "from huggingface_hub import hf_hub_download; import os; os.makedirs('$MODEL_DIR', exist_ok=True); hf_hub_download(repo_id='microsoft/BitNet-b1.58-2B-4T-gguf', filename='ggml-model-i2_s.gguf', local_dir='$MODEL_DIR')"
    
    # Also download the tokenizer if available
    if ! python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='microsoft/BitNet-b1.58-2B-4T-gguf', filename='tokenizer.model', local_dir='$MODEL_DIR')" 2>/dev/null; then
        echo "Note: tokenizer.model not found, will use fallback tokenization"
    fi
fi

# Now run the actual build script
echo "Running BitNet-WASM build script..."
./build.sh

# Start a simple HTTP server for testing (optional)
echo ""
echo "=== Build Complete ==="
echo "To test the example, run:"
echo "python3 -m http.server 8000"
echo "Then open http://localhost:8000/example/index.html in your browser"
echo ""
echo "To run the tests, open http://localhost:8000/tests/index.html in your browser"