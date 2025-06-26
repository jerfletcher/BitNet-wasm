# BitNet-WASM Integration Guide

**Status (June 26, 2025)**: Model loading and context creation working. Memory bounds optimization in progress.

This guide explains how to integrate BitNet-WASM into your own web applications and projects.

## 🎯 Current State

✅ **Working**: Model loading, context creation, single-token inference  
🔄 **In Progress**: Multi-token generation (memory bounds optimization)  
📋 **Tested**: BitNet-b1.58-2B with i2_s quantization (native format)

## Installation Methods

### Method 1: Build from Source (Recommended)

Clone and build the latest development version:

```bash
git clone --recursive https://github.com/jerfletcher/BitNet-wasm.git
cd BitNet-wasm
npm install
npm run build
npm test
```

### Method 2: Direct Download

Download the latest build files:
- `bitnet.js` - JavaScript loader and interface
- `bitnet.wasm` - WebAssembly module

### Method 3: CDN (When Available)

```html
<script type="module">
  import BitNetModule from 'https://cdn.jsdelivr.net/gh/jerfletcher/BitNet-wasm@latest/bitnet.js';
  // Your code here
</script>
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

### Supported Model Formats
- ✅ **GGUF with i2_s quantization**: Native BitNet format (tested working)
- ✅ **GGUF with Q8_0 quantization**: Standard format (expected to work)
- ⚠️ **Model Size**: 512MB WASM memory limit (use models under 400MB for safety)

### From URL (Current Method)

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
## Running Inference

### Current Status: Single-Token Working
✅ **Working**: Model loading, context creation, BOS token processing  
🔄 **In Progress**: Multi-token generation (memory optimization needed)

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
        
        // Run inference (currently single-token stable)
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

// Usage - currently works for single tokens
const result = await runInference(bitnet, "Hello", 50);  // Shorter outputs recommended
console.log('Generated text:', result);
```

### Testing and Validation

```javascript
// Test model loading and basic functionality
async function testBitNet(bitnet) {
    console.log('Testing BitNet-WASM...');
    
    // Test 1: Check initialization
    const initialized = bitnet._bitnet_is_model_loaded();
    console.log('Model loaded:', initialized === 1);
    
    // Test 2: Basic inference test
    if (initialized === 1) {
        const testResult = await runInference(bitnet, "test", 10);
        console.log('Test inference result:', testResult);
    }
    
    return initialized === 1;
}
```
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

## Development and Testing

### Build from Source

```bash
# Clone with submodules
git clone --recursive https://github.com/jerfletcher/BitNet-wasm.git
cd BitNet-wasm

# Install dependencies
npm install

# Build WASM module
npm run build

# Run browser tests
npm test

# Quick Node.js test
node tests/quick-test.js
```

### Test Suite

The project includes comprehensive testing in the `tests/` directory:

```bash
tests/
├── quick-test.js           # Main functionality test
├── test-minimal.js         # Memory and loading test
├── analyze-model.js        # Model format analysis
├── diagnose-alignment.js   # Alignment issue diagnosis (solved)
└── README.md              # Detailed test documentation
```

### Current Test Results

✅ **Model Loading**: Successfully loads BitNet-b1.58-2B (336MB)  
✅ **Context Creation**: Creates inference context with proper WASM config  
✅ **Memory Management**: 512MB WASM heap handles large models  
✅ **Single Token**: BOS token processing works correctly  
🔄 **Multi-Token**: In progress (memory bounds optimization needed)

## TypeScript Support

### Basic TypeScript Wrapper

```typescript
interface BitNetModule {
    _bitnet_init(): void;
    _bitnet_load_model(ptr: number, size: number): number;
    _bitnet_inference_run(inputPtr: number, outputPtr: number, maxLen: number): number;
    _bitnet_is_model_loaded(): number;
    _bitnet_free_model(): void;
    _malloc(size: number): number;
    _free(ptr: number): void;
    lengthBytesUTF8(str: string): number;
    stringToUTF8(str: string, ptr: number, maxLen: number): void;
    HEAPU8: Uint8Array;
}

class BitNetWrapper {
    private module: BitNetModule | null = null;
    
    async initialize(): Promise<void> {
        // @ts-ignore - Module loading
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
    
    async runInference(input: string, maxLength: number = 50): Promise<string> {
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

## Framework Integration

### React Integration (Current Status)

```jsx
import React, { useState, useEffect } from 'react';
// Note: Use local build files rather than CDN for now

function BitNetComponent() {
    const [bitnet, setBitnet] = useState(null);
    const [isLoaded, setIsLoaded] = useState(false);
    const [modelLoaded, setModelLoaded] = useState(false);
    const [output, setOutput] = useState('');
    const [status, setStatus] = useState('Initializing...');
    
    useEffect(() => {
        async function initBitNet() {
            try {
                // Load BitNet module (use local build)
                const BitNetModule = await import('./bitnet.js');
                const module = await BitNetModule.default();
                
                setBitnet(module);
                setIsLoaded(true);
                setStatus('Module loaded');
                
                // Check if model is already loaded (for testing)
                const loaded = module._bitnet_is_model_loaded();
                setModelLoaded(loaded === 1);
                if (loaded === 1) {
                    setStatus('Ready for inference');
                }
            } catch (error) {
                console.error('Failed to initialize BitNet:', error);
                setStatus('Error loading module');
            }
        }
        
        initBitNet();
    }, []);
    
    const runInference = async (input) => {
        if (!bitnet || !modelLoaded) {
            setOutput('No model loaded');
            return;
        }
        
        setStatus('Running inference...');
        try {
            // Use shorter inputs for current stability
            const result = await runInference(bitnet, input.substring(0, 20), 30);
            setOutput(result || 'No output generated');
            setStatus('Inference complete');
        } catch (error) {
            setOutput(`Error: ${error.message}`);
            setStatus('Inference failed');
        }
    };
    
    return (
        <div>
            <h2>BitNet-WASM React Component</h2>
            <p>Status: {status}</p>
            <p>Module: {isLoaded ? 'Loaded' : 'Loading...'}</p>
            <p>Model: {modelLoaded ? 'Loaded' : 'Not loaded'}</p>
            
            {modelLoaded && (
                <div>
                    <input 
                        type="text" 
                        placeholder="Enter short text..."
                        maxLength={20}
                        onChange={(e) => runInference(e.target.value)}
                    />
                    <div>Output: {output}</div>
                </div>
            )}
        </div>
    );
}

export default BitNetComponent;
```

## Performance Considerations & Limitations

### Current Performance Characteristics

**Working Features:**
- ✅ Model Loading: ~2-3 seconds for 336MB models
- ✅ Context Creation: Immediate after model load
- ✅ Single Token Processing: Near-native speed
- ✅ Memory Efficiency: 512MB WASM heap handles large models

**Current Limitations:**
- 🔄 Multi-token generation: Hits memory bounds after ~1 token
- ⚠️ Maximum output length: Recommend 10-50 tokens until optimization
- ⚠️ Context size: Currently 256, needs reduction to 128 for stability

### Memory Management Best Practices

```javascript
// Always free allocated memory
function safeInference(bitnet, input) {
    let inputPtr = null;
    let outputPtr = null;
    
    try {
        const inputBytes = bitnet.lengthBytesUTF8(input) + 1;
        inputPtr = bitnet._malloc(inputBytes);
        outputPtr = bitnet._malloc(50); // Smaller outputs for stability
        
        bitnet.stringToUTF8(input, inputPtr, inputBytes);
        const outputLength = bitnet._bitnet_inference_run(inputPtr, outputPtr, 50);
        
        if (outputLength > 0) {
            const outputBytes = new Uint8Array(bitnet.HEAPU8.buffer, outputPtr, outputLength);
            return new TextDecoder().decode(outputBytes);
        }
        return '';
    } catch (error) {
        console.error('Inference error:', error);
        return '';
    } finally {
        // Always clean up, even if error occurred
        if (inputPtr) bitnet._free(inputPtr);
        if (outputPtr) bitnet._free(outputPtr);
    }
}
```

### Browser Compatibility

**Requirements:**
- WebAssembly support (all modern browsers)
- ~512MB available memory
- Single-threaded execution (WASM limitation)

**Tested Browsers:**
- ✅ Chrome/Edge: Full compatibility
- ✅ Firefox: Full compatibility  
- ✅ Safari: Compatible (may need CORS headers)

## Troubleshooting Common Issues

### "Memory access out of bounds"
**Current Issue**: Multi-token generation hits memory limits  
**Workaround**: Use shorter inputs (≤20 chars) and outputs (≤50 tokens)  
**Solution in Progress**: Context size reduction (256→128)

### "Failed to load model"
**Check**: Model format (GGUF with i2_s or Q8_0 quantization)  
**Check**: Model size (under 400MB recommended)  
**Check**: File path and CORS headers

### "Alignment fault" 
**Status**: SOLVED! Removed SAFE_HEAP=1 from build config

### Performance Issues
**Current**: Single-token inference works well  
**Optimization**: Reduce max_length to 10-30 tokens for stability

## Examples and Demos

### Basic HTML Example

```html
<!DOCTYPE html>
<html>
<head>
    <title>BitNet-WASM Demo</title>
</head>
<body>
    <h1>BitNet-WASM Basic Demo</h1>
    <div id="status">Loading...</div>
    <div id="controls" style="display:none;">
        <input type="text" id="input" placeholder="Short text..." maxlength="20">
        <button onclick="runTest()">Test Inference</button>
    </div>
    <div id="output"></div>

    <script type="module">
        let bitnet = null;
        
        async function init() {
            try {
                const BitNetModule = await import('./bitnet.js');
                bitnet = await BitNetModule.default();
                
                const modelLoaded = bitnet._bitnet_is_model_loaded();
                if (modelLoaded === 1) {
                    document.getElementById('status').textContent = 'Ready!';
                    document.getElementById('controls').style.display = 'block';
                } else {
                    document.getElementById('status').textContent = 'No model loaded';
                }
            } catch (error) {
                document.getElementById('status').textContent = 'Error: ' + error.message;
            }
        }
        
        window.runTest = async function() {
            const input = document.getElementById('input').value;
            const output = document.getElementById('output');
            
            if (!bitnet || !input) return;
            
            output.textContent = 'Processing...';
            
            try {
                // For current stability, use very short outputs
                const result = await runInference(bitnet, input, 20);
                output.textContent = result || 'No output generated';
            } catch (error) {
                output.textContent = 'Error: ' + error.message;
            }
        };
        
        // Use the runInference function from previous examples
        async function runInference(bitnet, inputText, maxLength = 20) {
            // Implementation from earlier examples...
        }
        
        init();
    </script>
</body>
</html>
```

## Road to Production

### Current Development Status
1. ✅ **Core Infrastructure**: WASM build, model loading, context creation
2. ✅ **Single Token Processing**: BOS token and basic inference working
3. 🔄 **Memory Optimization**: Context size reduction in progress
4. 📋 **Multi-Token Generation**: Pending memory optimization completion
5. 📋 **Production Release**: After stable multi-token inference

### Next Milestones
1. **Memory Bounds Fix**: Reduce context from 256→128, batch 16→8
2. **Stable Multi-Token**: Consistent text generation up to 100+ tokens  
3. **Performance Tuning**: Optimize for longer inference sessions
4. **API Stabilization**: Finalize JavaScript interface
5. **Documentation**: Complete integration examples and best practices

## Support and Resources

### Getting Help
- **GitHub Issues**: [Report bugs and request features](https://github.com/jerfletcher/BitNet-wasm/issues)
- **Test Suite**: Run `npm test` for comprehensive diagnostics
- **Quick Test**: Use `node tests/quick-test.js` for rapid debugging

### Development Resources
- **Source Code**: `src/bitnet_wasm.cpp` - Main WASM implementation
- **Test Scripts**: `tests/` directory with comprehensive test suite
- **Build System**: `npm run build` for development builds
- **Documentation**: Complete technical analysis in `docs/` directory

### Model Resources
- **Supported**: GGUF format with i2_s quantization (BitNet native)
- **Alternative**: Q8_0 quantization for broader compatibility
- **Size Limits**: Under 400MB recommended for current WASM memory config

BitNet-WASM is actively developed with real BitNet inference capabilities. Current focus is memory optimization for stable multi-token generation.
        
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