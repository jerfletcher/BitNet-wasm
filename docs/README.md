# BitNet-WASM Demo

This is a demonstration of the BitNet-WASM library, which provides efficient 2-bit quantized neural network inference directly in your web browser using WebAssembly.

## Features

- **Model Inference**: Load and run inference with BitNet models
- **IndexedDB Caching**: Models are cached in your browser's IndexedDB for faster subsequent loads
- **Matrix Multiplication**: Demonstrate BitNet's 2-bit quantized matrix multiplication
- **Tensor Transformation**: Show how tensors are transformed using BitNet quantization

## How to Use

1. Click the "Load Model" button to download and cache the default model
2. Enter your input text in the text area
3. Click "Run Inference" to see the model's output

## Technical Details

- The demo uses [Dexie.js](https://dexie.org/) for IndexedDB access
- Models are automatically cached and managed to prevent excessive storage use
- GGUF model files are automatically converted to the flat binary format required by BitNet-WASM

## Source Code

The source code for this demo and the BitNet-WASM library is available on [GitHub](https://github.com/jerfletcher/BitNet-wasm).