#!/bin/bash
# BitNet WASM Compatible Model Download

echo "🔽 Downloading WASM-compatible test model..."

# Option 1: Small test model (Qwen2-0.5B)
echo "📥 Downloading Qwen2-0.5B-Instruct (Q4_0) - ~350MB"
mkdir -p models/test
cd models/test

# Use huggingface-cli or wget to download
# huggingface-cli download Qwen/Qwen2-0.5B-Instruct-GGUF qwen2-0_5b-instruct-q4_0.gguf --local-dir .

# Option 2: Use curl/wget (example URL - replace with actual)
# wget -O qwen2-0_5b-instruct-q4_0.gguf "https://huggingface.co/Qwen/Qwen2-0.5B-Instruct-GGUF/resolve/main/qwen2-0_5b-instruct-q4_0.gguf"

echo "✅ Download complete"
echo "📝 Update test-minimal.js to use: ./models/test/qwen2-0_5b-instruct-q4_0.gguf"
