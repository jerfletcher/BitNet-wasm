# BitNet-WASM Demo

This is a demonstration of the BitNet-WASM library, which provides efficient 2-bit quantized neural network inference directly in your web browser using WebAssembly.

## Features

- **Model Inference**: Load and run inference with BitNet models
- **Matrix Multiplication**: Demonstrate BitNet's 2-bit quantized matrix multiplication
- **Integration Example**: Shows how to integrate BitNet-WASM in your own projects

## How to Use

1. Click the "Load Model" button to download the default model (note: models are downloaded fresh each time)
2. Enter your input text in the text area
3. Click "Run Inference" to see the model's output

## Integration Guide

This demo shows how to integrate BitNet-WASM in your own projects. You can use it in three ways:

### CDN (Recommended for quick testing)

```javascript
import BitNetModule from 'https://cdn.jsdelivr.net/gh/jerfletcher/BitNet-wasm@latest/bitnet.js';

// Initialize the module
const bitnet = await BitNetModule();
bitnet._bitnet_init();
```

### Download and include locally

1. Download `bitnet.js` and `bitnet.wasm` from [GitHub Releases](https://github.com/jerfletcher/BitNet-wasm/releases)
2. Place them in your project directory
3. Import in your JavaScript:

```javascript
import BitNetModule from './bitnet.js';
```

### Build from source

```bash
git clone --recursive https://github.com/jerfletcher/BitNet-wasm.git
cd BitNet-wasm
./setup_and_build.sh
```

## Technical Details

- The demo downloads models fresh each time for simplicity
- The implementation follows the integration guide from the BitNet-WASM documentation
- Uses a Node.js Express server for local development

## Source Code

The source code for this demo and the BitNet-WASM library is available on [GitHub](https://github.com/jerfletcher/BitNet-wasm).

## License

MIT License - see the [LICENSE](https://github.com/jerfletcher/BitNet-wasm/blob/main/LICENSE) file for details.