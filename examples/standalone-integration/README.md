# BitNet-WASM Standalone Integration Example

This example demonstrates how to integrate BitNet-WASM into a separate website or application.

## Overview

This standalone example shows:
- How to load BitNet-WASM from different sources (NPM, CDN, GitHub releases)
- Basic integration patterns
- Error handling
- Performance monitoring
- User interface design

## Files

- `index.html` - Complete standalone example with simulated BitNet operations
- `README.md` - This documentation

## How to Use

### Option 1: View the Example

Simply open `index.html` in a web browser to see the integration example in action.

### Option 2: Integrate into Your Project

1. **Choose your installation method:**

   **NPM Package:**
   ```bash
   npm install bitnet-wasm
   ```

   **CDN:**
   ```html
   <script type="module">
     import BitNetModule from 'https://cdn.jsdelivr.net/npm/bitnet-wasm@latest/bitnet.js';
   </script>
   ```

   **GitHub Releases:**
   Download `bitnet.js` and `bitnet.wasm` from the [releases page](https://github.com/jerfletcher/BitNet-wasm/releases).

2. **Copy the integration code:**

   Use the JavaScript code from `index.html` as a starting point for your integration.

3. **Customize for your needs:**

   - Modify the UI to match your application
   - Add your own model loading logic
   - Implement your specific use cases

## Key Integration Points

### Initialization

```javascript
import BitNetModule from 'bitnet-wasm';

async function initBitNet() {
    const bitnet = await BitNetModule();
    bitnet._bitnet_init();
    return bitnet;
}
```

### Model Loading

```javascript
async function loadModel(bitnet, modelUrl) {
    const response = await fetch(modelUrl);
    const modelData = await response.arrayBuffer();
    
    const modelSize = modelData.byteLength;
    const modelPtr = bitnet._malloc(modelSize);
    
    const heapBytes = new Uint8Array(bitnet.HEAPU8.buffer, modelPtr, modelSize);
    heapBytes.set(new Uint8Array(modelData));
    
    const success = bitnet._bitnet_load_model(modelPtr, modelSize);
    bitnet._free(modelPtr);
    
    return success === 1;
}
```

### Running Inference

```javascript
async function runInference(bitnet, inputText, maxLength = 100) {
    const inputBytes = bitnet.lengthBytesUTF8(inputText) + 1;
    const inputPtr = bitnet._malloc(inputBytes);
    const outputPtr = bitnet._malloc(maxLength);
    
    bitnet.stringToUTF8(inputText, inputPtr, inputBytes);
    
    const outputLength = bitnet._bitnet_inference_run(inputPtr, outputPtr, maxLength);
    
    let result = '';
    if (outputLength > 0) {
        const outputBytes = new Uint8Array(bitnet.HEAPU8.buffer, outputPtr, outputLength);
        result = new TextDecoder().decode(outputBytes);
    }
    
    bitnet._free(inputPtr);
    bitnet._free(outputPtr);
    
    return result;
}
```

## Features Demonstrated

### 🧠 Model Information
- Getting model metadata
- Displaying model statistics
- Memory usage monitoring

### 💬 Text Inference
- Loading input text
- Running BitNet inference
- Displaying generated output
- Performance metrics

### 🔢 Matrix Operations
- BitNet quantized matrix multiplication
- Input validation
- Result visualization

### ⚡ Tensor Quantization
- 1.58-bit quantization demonstration
- Scale factor calculation
- Compression ratio display

### 📊 Performance Benchmarking
- Timing various operations
- Memory usage tracking
- Browser compatibility testing

## Error Handling

The example includes comprehensive error handling:

```javascript
try {
    const result = await runInference(bitnet, inputText);
    displayResult(result);
} catch (error) {
    console.error('Inference failed:', error);
    displayError(error.message);
}
```

## Browser Compatibility

This example works in all modern browsers that support:
- WebAssembly
- ES6 modules
- Fetch API
- Async/await

## Performance Considerations

- **Memory Management**: Always free allocated memory
- **Async Operations**: Use proper async/await patterns
- **Error Boundaries**: Implement comprehensive error handling
- **Loading States**: Provide user feedback during operations

## Customization

You can customize this example by:

1. **Styling**: Modify the CSS to match your application's design
2. **Functionality**: Add or remove features based on your needs
3. **Integration**: Adapt the code for your framework (React, Vue, Angular, etc.)
4. **Models**: Use different BitNet models for your specific use case

## Next Steps

1. Review the [Integration Guide](../../INTEGRATION.md) for more detailed instructions
2. Check out the [main example](../index.html) for a complete implementation
3. Visit the [GitHub repository](https://github.com/jerfletcher/BitNet-wasm) for the latest updates
4. Join the [discussions](https://github.com/jerfletcher/BitNet-wasm/discussions) for community support

## Support

For questions and issues:
- [GitHub Issues](https://github.com/jerfletcher/BitNet-wasm/issues)
- [GitHub Discussions](https://github.com/jerfletcher/BitNet-wasm/discussions)
- [Integration Guide](../../INTEGRATION.md)