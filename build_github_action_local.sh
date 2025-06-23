#!/bin/bash
# Simulate the main steps of the GitHub Actions build locally
set -e

# 1. Setup Node.js (skip if already installed)
# 2. Setup Python (skip if already installed)
# 3. Install Python dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install huggingface_hub

# 4. Install Emscripten SDK if not present
if [ ! -d "emsdk" ]; then
  git clone https://github.com/emscripten-core/emsdk.git
  cd emsdk
  ./emsdk install 4.0.8
  ./emsdk activate 4.0.8
  cd ..
else
  cd emsdk
  ./emsdk activate 4.0.8
  cd ..
fi

# 5. Build BitNet-WASM
source ./emsdk/emsdk_env.sh
./build.sh

echo "Local GitHub Actions build completed."
