# BitNet-WASM Integration Guide

This guide explains how to integrate BitNet-WASM into your own web applications and projects.

## Installation Methods

### Method 1: Direct Download (Recommended)

Download the latest release files from [GitHub Releases](https://github.com/jerfletcher/BitNet-wasm/releases):
- `bitnet.js` - JavaScript loader and interface
- `bitnet.wasm` - WebAssembly module
- `bitnet.d.ts` - TypeScript definitions

### Method 2: CDN (jsDelivr)

Use the CDN version directly in your HTML:

```html
<script type="module">
  import BitNetModule from 'https://cdn.jsdelivr.net/gh/jerfletcher/BitNet-wasm@latest/bitnet.js';
  // Your code here
</script>
```

### Method 3: Clone Repository

Clone the repository and build from source:

```bash
git clone --recursive https://github.com/jerfletcher/BitNet-wasm.git
cd BitNet-wasm
./setup_and_build.sh
```

## Basic Integration

### HTML Setup

```html
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>My BitNet App</title>
</head>
<body>
    <div id="output"></div>
    <script type="module" src="app.js"></script>
</body>
</html>
```

### JavaScript Integration

```javascript
// app.js
import BitNetModule from './bitnet.js'; // or from CDN

async function initBitNet() {
    try {
        // Initialize the WASM module
        const bitnet = await BitNetModule();
        
        // Initialize BitNet engine
        bitnet._bitnet_init();
        
        console.log('BitNet-WASM initialized successfully!');
        return bitnet;
    } catch (error) {
        console.error('Failed to initialize BitNet-WASM:', error);
        throw error;
    }
}

// Example usage
async function main() {
    const bitnet = await initBitNet();
    
    // Check if model is loaded
    const isLoaded = bitnet._bitnet_is_model_loaded();
    console.log('Model loaded:', isLoaded === 1);
    
    // Your BitNet operations here...
}

main().catch(console.error);
```

## Loading Models

### From URL

```javascript
async function loadModelFromURL(bitnet, modelUrl) {
    try {
        // Fetch model data
        const response = await fetch(modelUrl);
        if (!response.ok) {
            throw new Error(`Failed to fetch model: ${response.statusText}`);
        }
        
        const modelData = await response.arrayBuffer();
        
        // Allocate memory in WASM
        const modelSize = modelData.byteLength;
        const modelPtr = bitnet._malloc(modelSize);
        
        // Copy model data to WASM memory
        const heapBytes = new Uint8Array(bitnet.HEAPU8.buffer, modelPtr, modelSize);
        heapBytes.set(new Uint8Array(modelData));
        
        // Load model
        const success = bitnet._bitnet_load_model(modelPtr, modelSize);
        
        // Free temporary memory
        bitnet._free(modelPtr);
        
        if (success === 1) {
            console.log('Model loaded successfully');
            return true;
        } else {
            throw new Error('Failed to load model');
        }
    } catch (error) {
        console.error('Error loading model:', error);
        return false;
    }
}

// Usage
const modelLoaded = await loadModelFromURL(bitnet, '/path/to/your/model.gguf');
```

### From File Input

```javascript
function setupFileInput(bitnet) {
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.accept = '.gguf';
    
    fileInput.addEventListener('change', async (event) => {
        const file = event.target.files[0];
        if (!file) return;
        
        try {
            const arrayBuffer = await file.arrayBuffer();
            
            // Allocate memory and load model
            const modelSize = arrayBuffer.byteLength;
            const modelPtr = bitnet._malloc(modelSize);
            
            const heapBytes = new Uint8Array(bitnet.HEAPU8.buffer, modelPtr, modelSize);
            heapBytes.set(new Uint8Array(arrayBuffer));
            
            const success = bitnet._bitnet_load_model(modelPtr, modelSize);
            bitnet._free(modelPtr);
            
            if (success === 1) {
                console.log('Model loaded from file successfully');
            } else {
                console.error('Failed to load model from file');
            }
        } catch (error) {
            console.error('Error loading model from file:', error);
        }
    });
    
    return fileInput;
}
```

## Running Inference

```javascript
async function runInference(bitnet, inputText, maxLength = 100) {
    try {
        // Check if model is loaded
        if (bitnet._bitnet_is_model_loaded() !== 1) {
            throw new Error('No model loaded');
        }
        
        // Allocate memory for input and output
        const inputBytes = bitnet.lengthBytesUTF8(inputText) + 1;
        const inputPtr = bitnet._malloc(inputBytes);
        const outputPtr = bitnet._malloc(maxLength);
        
        // Copy input text to WASM memory
        bitnet.stringToUTF8(inputText, inputPtr, inputBytes);
        
        // Run inference
        const outputLength = bitnet._bitnet_inference_run(inputPtr, outputPtr, maxLength);
        
        let result = '';
        if (outputLength > 0) {
            // Read output from WASM memory
            const outputBytes = new Uint8Array(bitnet.HEAPU8.buffer, outputPtr, outputLength);
            result = new TextDecoder().decode(outputBytes);
        }
        
        // Free memory
        bitnet._free(inputPtr);
        bitnet._free(outputPtr);
        
        return result;
    } catch (error) {
        console.error('Error running inference:', error);
        return '';
    }
}

// Usage
const result = await runInference(bitnet, "Hello, how are you?", 200);
console.log('Generated text:', result);
```

## Matrix Operations

```javascript
function performMatrixMultiplication(bitnet, matrixA, matrixB, rows, cols) {
    try {
        // This is a simplified example - you'll need to implement
        // the actual matrix multiplication using BitNet's quantization
        
        // Allocate memory for matrices
        const sizeA = matrixA.length * 4; // float32
        const sizeB = matrixB.length * 4;
        const sizeResult = rows * cols * 4;
        
        const ptrA = bitnet._malloc(sizeA);
        const ptrB = bitnet._malloc(sizeB);
        const ptrResult = bitnet._malloc(sizeResult);
        
        // Copy data to WASM memory
        const heapA = new Float32Array(bitnet.HEAPF32.buffer, ptrA, matrixA.length);
        const heapB = new Float32Array(bitnet.HEAPF32.buffer, ptrB, matrixB.length);
        
        heapA.set(matrixA);
        heapB.set(matrixB);
        
        // Perform matrix multiplication (you'll need to implement this function)
        // const success = bitnet._bitnet_matrix_multiply(ptrA, ptrB, ptrResult, rows, cols);
        
        // Read result
        const resultArray = new Float32Array(bitnet.HEAPF32.buffer, ptrResult, rows * cols);
        const result = Array.from(resultArray);
        
        // Free memory
        bitnet._free(ptrA);
        bitnet._free(ptrB);
        bitnet._free(ptrResult);
        
        return result;
    } catch (error) {
        console.error('Error in matrix multiplication:', error);
        return [];
    }
}
```

## TypeScript Support

If you're using TypeScript, download the `bitnet.d.ts` file and reference it:

```typescript
/// <reference path="./bitnet.d.ts" />
import BitNetModule from './bitnet.js';

class BitNetWrapper {
    private module: BitNetModule | null = null;
    
    async initialize(): Promise<void> {
        this.module = await BitNetModule();
        this.module._bitnet_init();
    }
    
    async loadModel(modelData: ArrayBuffer): Promise<boolean> {
        if (!this.module) throw new Error('Module not initialized');
        
        const modelSize = modelData.byteLength;
        const modelPtr = this.module._malloc(modelSize);
        
        const heapBytes = new Uint8Array(this.module.HEAPU8.buffer, modelPtr, modelSize);
        heapBytes.set(new Uint8Array(modelData));
        
        const success = this.module._bitnet_load_model(modelPtr, modelSize);
        this.module._free(modelPtr);
        
        return success === 1;
    }
    
    async runInference(input: string, maxLength: number = 100): Promise<string> {
        if (!this.module) throw new Error('Module not initialized');
        
        const inputBytes = this.module.lengthBytesUTF8(input) + 1;
        const inputPtr = this.module._malloc(inputBytes);
        const outputPtr = this.module._malloc(maxLength);
        
        this.module.stringToUTF8(input, inputPtr, inputBytes);
        
        const outputLength = this.module._bitnet_inference_run(inputPtr, outputPtr, maxLength);
        
        let result = '';
        if (outputLength > 0) {
            const outputBytes = new Uint8Array(this.module.HEAPU8.buffer, outputPtr, outputLength);
            result = new TextDecoder().decode(outputBytes);
        }
        
        this.module._free(inputPtr);
        this.module._free(outputPtr);
        
        return result;
    }
    
    cleanup(): void {
        if (this.module) {
            this.module._bitnet_free_model();
        }
    }
}
```

## React Integration

```jsx
import React, { useState, useEffect } from 'react';
import BitNetModule from './bitnet.js';

function BitNetComponent() {
    const [bitnet, setBitnet] = useState(null);
    const [isLoaded, setIsLoaded] = useState(false);
    const [output, setOutput] = useState('');
    
    useEffect(() => {
        async function initBitNet() {
            try {
                const module = await BitNetModule();
                module._bitnet_init();
                setBitnet(module);
                setIsLoaded(true);
            } catch (error) {
                console.error('Failed to initialize BitNet:', error);
            }
        }
        
        initBitNet();
    }, []);
    
    const runInference = async (input) => {
        if (!bitnet || bitnet._bitnet_is_model_loaded() !== 1) {
            setOutput('No model loaded');
            return;
        }
        
        // Implementation similar to previous examples
        // ...
    };
    
    return (
        <div>
            <h2>BitNet-WASM React Component</h2>
            <p>Status: {isLoaded ? 'Loaded' : 'Loading...'}</p>
            {/* Your UI components here */}
        </div>
    );
}

export default BitNetComponent;
```

## Vue.js Integration

```vue
<template>
  <div>
    <h2>BitNet-WASM Vue Component</h2>
    <p>Status: {{ status }}</p>
    <button @click="runInference" :disabled="!isReady">Run Inference</button>
    <div v-if="output">{{ output }}</div>
  </div>
</template>

<script>
import BitNetModule from './bitnet.js';

export default {
  name: 'BitNetComponent',
  data() {
    return {
      bitnet: null,
      isReady: false,
      status: 'Loading...',
      output: ''
    };
  },
  async mounted() {
    try {
      this.bitnet = await BitNetModule();
      this.bitnet._bitnet_init();
      this.isReady = true;
      this.status = 'Ready';
    } catch (error) {
      console.error('Failed to initialize BitNet:', error);
      this.status = 'Error';
    }
  },
  methods: {
    async runInference() {
      // Implementation here
    }
  }
};
</script>
```

## Performance Considerations

### Memory Management

- Always free allocated memory using `_free()` to prevent memory leaks
- Consider implementing a memory pool for frequent allocations
- Monitor memory usage in browser dev tools

### Loading Optimization

```javascript
// Preload the WASM module
const bitnetPromise = BitNetModule();

// Use it when needed
async function useBitNet() {
    const bitnet = await bitnetPromise;
    // Use bitnet...
}
```

### Worker Thread Integration

```javascript
// worker.js
import BitNetModule from './bitnet.js';

let bitnet = null;

self.onmessage = async function(e) {
    const { type, data } = e.data;
    
    switch (type) {
        case 'init':
            try {
                bitnet = await BitNetModule();
                bitnet._bitnet_init();
                self.postMessage({ type: 'init', success: true });
            } catch (error) {
                self.postMessage({ type: 'init', success: false, error: error.message });
            }
            break;
            
        case 'inference':
            if (!bitnet) {
                self.postMessage({ type: 'inference', error: 'Not initialized' });
                return;
            }
            
            // Run inference in worker thread
            const result = await runInference(bitnet, data.input, data.maxLength);
            self.postMessage({ type: 'inference', result });
            break;
    }
};
```

## Error Handling

```javascript
class BitNetError extends Error {
    constructor(message, code) {
        super(message);
        this.name = 'BitNetError';
        this.code = code;
    }
}

async function safeBitNetOperation(bitnet, operation) {
    try {
        if (!bitnet) {
            throw new BitNetError('BitNet module not initialized', 'NOT_INITIALIZED');
        }
        
        return await operation(bitnet);
    } catch (error) {
        if (error instanceof BitNetError) {
            throw error;
        }
        throw new BitNetError(`BitNet operation failed: ${error.message}`, 'OPERATION_FAILED');
    }
}
```

## Browser Compatibility

BitNet-WASM requires:
- WebAssembly support (all modern browsers)
- ES6 modules support
- Sufficient memory for model loading

### Feature Detection

```javascript
function checkBrowserSupport() {
    const checks = {
        webassembly: typeof WebAssembly === 'object',
        modules: 'noModule' in HTMLScriptElement.prototype,
        memory: navigator.deviceMemory ? navigator.deviceMemory >= 2 : true // 2GB minimum recommended
    };
    
    const supported = Object.values(checks).every(Boolean);
    
    if (!supported) {
        console.warn('Browser compatibility issues detected:', checks);
    }
    
    return supported;
}
```

## Deployment Considerations

### CORS Headers

Ensure your server serves the WASM file with proper CORS headers:

```
Access-Control-Allow-Origin: *
Access-Control-Allow-Methods: GET
Access-Control-Allow-Headers: Content-Type
```

### MIME Types

Configure your server to serve `.wasm` files with the correct MIME type:

```
application/wasm
```

### CDN Configuration

When using a CDN, ensure:
- Proper cache headers for WASM files
- Compression (gzip/brotli) for JavaScript files
- Geographic distribution for better performance

## Examples Repository

For complete working examples, see the [examples directory](./example/) in this repository, which includes:
- Basic HTML/JavaScript integration
- Model loading and inference
- Matrix operations
- Error handling patterns

## Support

For issues and questions:
- [GitHub Issues](https://github.com/jerfletcher/BitNet-wasm/issues)
- [Discussions](https://github.com/jerfletcher/BitNet-wasm/discussions)

## License

MIT License - see [LICENSE](./LICENSE) file for details.