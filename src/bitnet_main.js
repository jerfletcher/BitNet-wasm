// BitNet WASM inference main.js
import ModuleFactory from '../bitnet.js';

// Global variable to store the WASM module instance
let wasmModule = null;
let modelLoaded = false;

// Helper function to allocate string in WASM memory
function allocateString(str) {
    const len = wasmModule.lengthBytesUTF8(str) + 1;
    const ptr = wasmModule._malloc(len);
    wasmModule.stringToUTF8(str, ptr, len);
    return { ptr, len, free: () => wasmModule._free(ptr) };
}

// Helper function to read string from WASM memory
function readString(ptr, maxLen = 1024) {
    const buffer = new Uint8Array(wasmModule.HEAPU8.buffer, ptr, maxLen);
    let str = '';
    for (let i = 0; i < maxLen; i++) {
        if (buffer[i] === 0) break;
        str += String.fromCharCode(buffer[i]);
    }
    return str;
}

// Load model from URL
async function loadModelFromURL(modelPath) {
    const outputElement = document.getElementById('output');
    const loadStatusElement = document.getElementById('load-status');
    
    try {
        outputElement.innerHTML += `Loading model from ${modelPath}...<br>`;
        loadStatusElement.innerHTML = 'Loading model...';
        
        const response = await fetch(modelPath);
        if (!response.ok) {
            throw new Error(`Failed to fetch model: ${response.status} ${response.statusText}`);
        }
        
        const arrayBuffer = await response.arrayBuffer();
        const modelSize = arrayBuffer.byteLength;
        
        outputElement.innerHTML += `Model downloaded: ${(modelSize / (1024 * 1024)).toFixed(2)} MB<br>`;
        loadStatusElement.innerHTML = 'Processing model...';
        
        // Allocate memory in WASM for the model
        const modelPtr = wasmModule._malloc(modelSize);
        if (!modelPtr) {
            throw new Error('Failed to allocate memory for model');
        }
        
        // Copy model data to WASM memory
        const modelData = new Uint8Array(wasmModule.HEAPU8.buffer, modelPtr, modelSize);
        modelData.set(new Uint8Array(arrayBuffer));
        
        outputElement.innerHTML += `Model data copied to WASM memory<br>`;
        loadStatusElement.innerHTML = 'Loading model into BitNet...';
        
        // Load model using BitNet
        const result = wasmModule._bitnet_load_model(modelPtr, modelSize);
        
        // Free the temporary model data
        wasmModule._free(modelPtr);
        
        if (result === 1) {
            modelLoaded = true;
            outputElement.innerHTML += `<span class="success">Model loaded successfully!</span><br>`;
            loadStatusElement.innerHTML = '<span class="success">Model loaded successfully!</span>';
            
            // Get model info
            const vocabSizePtr = wasmModule._malloc(4);
            const nEmbdPtr = wasmModule._malloc(4);
            const nLayerPtr = wasmModule._malloc(4);
            
            wasmModule._bitnet_get_model_info(vocabSizePtr, nEmbdPtr, nLayerPtr);
            
            const vocabSize = new Uint32Array(wasmModule.HEAPU8.buffer, vocabSizePtr, 1)[0];
            const nEmbd = new Uint32Array(wasmModule.HEAPU8.buffer, nEmbdPtr, 1)[0];
            const nLayer = new Uint32Array(wasmModule.HEAPU8.buffer, nLayerPtr, 1)[0];
            
            wasmModule._free(vocabSizePtr);
            wasmModule._free(nEmbdPtr);
            wasmModule._free(nLayerPtr);
            
            outputElement.innerHTML += `Model info: vocab=${vocabSize}, embd=${nEmbd}, layers=${nLayer}<br>`;
            
            // Enable inference button
            const inferenceButton = document.getElementById('run-inference');
            if (inferenceButton) {
                inferenceButton.disabled = false;
            }
            
            return true;
        } else {
            throw new Error('BitNet failed to load model');
        }
        
    } catch (error) {
        outputElement.innerHTML += `<span class="error">Error loading model: ${error.message}</span><br>`;
        loadStatusElement.innerHTML = `<span class="error">Error: ${error.message}</span>`;
        console.error('Error loading model:', error);
        return false;
    }
}

// Run BitNet inference
function runBitNetInference(inputText) {
    const outputElement = document.getElementById('output');
    const resultElement = document.getElementById('inference-result');
    
    if (!modelLoaded) {
        const error = 'Model not loaded. Please load a model first.';
        outputElement.innerHTML += `<span class="error">${error}</span><br>`;
        resultElement.innerHTML = `<span class="error">${error}</span>`;
        return;
    }
    
    try {
        outputElement.innerHTML += `Running BitNet inference on: "${inputText}"<br>`;
        resultElement.innerHTML = 'Running inference...';
        
        // Allocate input string
        const inputAlloc = allocateString(inputText);
        
        // Allocate output buffer
        const maxOutputLen = 512;
        const outputPtr = wasmModule._malloc(maxOutputLen);
        
        // Run inference
        const outputLen = wasmModule._bitnet_inference_run(inputAlloc.ptr, outputPtr, maxOutputLen);
        
        if (outputLen > 0) {
            const outputText = readString(outputPtr, outputLen);
            outputElement.innerHTML += `<span class="success">Inference completed!</span><br>`;
            resultElement.innerHTML = `<div class="model-output"><strong>BitNet Output:</strong><br>${outputText}</div>`;
        } else {
            throw new Error('Inference failed or returned empty result');
        }
        
        // Clean up
        inputAlloc.free();
        wasmModule._free(outputPtr);
        
    } catch (error) {
        const errorMsg = `Error during inference: ${error.message}`;
        outputElement.innerHTML += `<span class="error">${errorMsg}</span><br>`;
        resultElement.innerHTML = `<span class="error">${errorMsg}</span>`;
        console.error('Inference error:', error);
    }
}

// Matrix multiplication demo (simplified for BitNet)
function performMatrixMultiplication(matrixA, matrixB) {
    const outputElement = document.getElementById('output');
    
    try {
        outputElement.innerHTML += `Performing BitNet matrix multiplication...<br>`;
        
        // Simple matrix multiplication with BitNet-style quantization
        const result = new Float32Array(matrixA.rows * matrixB.cols);
        
        // Quantize weights to {-1, 0, 1}
        const quantizedB = new Int8Array(matrixB.data.length);
        for (let i = 0; i < matrixB.data.length; i++) {
            const val = matrixB.data[i];
            if (Math.abs(val) < 0.1) {
                quantizedB[i] = 0;
            } else if (val > 0) {
                quantizedB[i] = 1;
            } else {
                quantizedB[i] = -1;
            }
        }
        
        // Perform quantized matrix multiplication
        for (let i = 0; i < matrixA.rows; i++) {
            for (let j = 0; j < matrixB.cols; j++) {
                let sum = 0;
                for (let k = 0; k < matrixA.cols; k++) {
                    const a_val = matrixA.data[i * matrixA.cols + k];
                    const b_val = quantizedB[k * matrixB.cols + j] * 0.5; // Scale factor
                    sum += a_val * b_val;
                }
                result[i * matrixB.cols + j] = sum;
            }
        }
        
        outputElement.innerHTML += `Matrix multiplication completed (BitNet quantized)<br>`;
        
        return {
            data: result,
            rows: matrixA.rows,
            cols: matrixB.cols
        };
    } catch (e) {
        console.error("Error in matrix multiplication:", e);
        throw e;
    }
}

// Tensor transformation with BitNet quantization
function transformTensor(tensorData) {
    const outputElement = document.getElementById('output');
    
    try {
        outputElement.innerHTML += `Performing BitNet tensor quantization...<br>`;
        
        // BitNet quantization: quantize to {-1, 0, 1}
        const quantized = new Int8Array(tensorData.length);
        const dequantized = new Float32Array(tensorData.length);
        
        // Find scale (max absolute value)
        let scale = 0;
        for (let i = 0; i < tensorData.length; i++) {
            scale = Math.max(scale, Math.abs(tensorData[i]));
        }
        
        if (scale === 0) scale = 1; // Avoid division by zero
        
        // Quantize
        for (let i = 0; i < tensorData.length; i++) {
            const normalized = tensorData[i] / scale;
            if (Math.abs(normalized) < 0.33) {
                quantized[i] = 0;
            } else if (normalized > 0) {
                quantized[i] = 1;
            } else {
                quantized[i] = -1;
            }
            dequantized[i] = quantized[i] * scale;
        }
        
        outputElement.innerHTML += `BitNet quantization completed (scale: ${scale.toFixed(4)})<br>`;
        
        return {
            original: Array.from(tensorData),
            quantized: Array.from(quantized),
            dequantized: Array.from(dequantized),
            scale: scale,
            message: "Tensor quantized using BitNet 1.58-bit quantization."
        };
    } catch (e) {
        console.error("Error in tensor transformation:", e);
        return {
            original: Array.from(tensorData),
            error: e.toString(),
            message: "Error transforming tensor."
        };
    }
}

// Helper function to parse comma-separated values into a Float32Array
function parseFloatArray(text) {
    return new Float32Array(
        text.split(',')
            .map(s => s.trim())
            .filter(s => s.length > 0)
            .map(s => parseFloat(s))
    );
}

// Helper function to create a matrix from a Float32Array
function createMatrix(data, rows, cols) {
    if (data.length !== rows * cols) {
        throw new Error(`Data length ${data.length} does not match dimensions ${rows}x${cols}`);
    }
    
    return {
        data: data,
        rows: rows,
        cols: cols
    };
}

// Initialize WASM module
function onWasmInitialized(wasmModuleInstance) {
    wasmModule = wasmModuleInstance;
    
    const outputElement = document.getElementById('output');
    const statusElement = document.getElementById('status');
    
    outputElement.innerHTML = 'BitNet WASM Module Initialized and Ready.<br>';
    if (statusElement) {
        statusElement.textContent = 'WASM module initialized. Ready to use BitNet functions.';
        statusElement.classList.add('success');
    }
    
    // Initialize BitNet
    try {
        wasmModule._bitnet_init();
        outputElement.innerHTML += 'BitNet inference engine initialized successfully.<br>';
        
        // Check available functions
        const availableFunctions = [];
        if (typeof wasmModule._bitnet_init === 'function') availableFunctions.push('bitnet_init');
        if (typeof wasmModule._bitnet_load_model === 'function') availableFunctions.push('bitnet_load_model');
        if (typeof wasmModule._bitnet_inference_run === 'function') availableFunctions.push('bitnet_inference_run');
        if (typeof wasmModule._bitnet_get_model_info === 'function') availableFunctions.push('bitnet_get_model_info');
        if (typeof wasmModule._bitnet_is_model_loaded === 'function') availableFunctions.push('bitnet_is_model_loaded');
        if (typeof wasmModule._bitnet_free_model === 'function') availableFunctions.push('bitnet_free_model');
        
        outputElement.innerHTML += `Available BitNet functions: ${availableFunctions.join(', ')}<br>`;
        
    } catch (e) {
        console.error('Error initializing BitNet:', e);
        outputElement.innerHTML += `<span class="error">Error initializing BitNet: ${e}</span><br>`;
        if (statusElement) {
            statusElement.textContent = 'Error initializing BitNet.';
            statusElement.classList.add('error');
        }
        return;
    }
    
    setupModelInferenceDemo();
    setupMatrixMultiplicationDemo();
    setupTensorTransformationDemo();
    
    // Enable load model button
    const loadButton = document.getElementById('load-model');
    if (loadButton) loadButton.disabled = false;
}

// Set up the model inference demo
function setupModelInferenceDemo() {
    const loadButton = document.getElementById('load-model');
    const inferenceButton = document.getElementById('run-inference');
    
    if (loadButton) {
        loadButton.addEventListener('click', async () => {
            const modelPath = document.getElementById('model-path').value;
            loadButton.disabled = true;
            loadButton.textContent = 'Loading...';
            
            const success = await loadModelFromURL(modelPath);
            
            loadButton.disabled = false;
            loadButton.textContent = 'Load Model';
            
            if (!success) {
                inferenceButton.disabled = true;
            }
        });
    }
    
    if (inferenceButton) {
        inferenceButton.addEventListener('click', () => {
            const inputText = document.getElementById('inference-input').value;
            runBitNetInference(inputText);
        });
    }
}

// Set up the matrix multiplication demo
function setupMatrixMultiplicationDemo() {
    const runButton = document.getElementById('run-matmul');
    if (!runButton) return;
    
    runButton.addEventListener('click', () => {
        const matrixAInput = document.getElementById('matrix-a').value;
        const matrixBInput = document.getElementById('matrix-b').value;
        const resultElement = document.getElementById('matmul-result');
        
        try {
            // Parse input matrices
            const matrixAData = parseFloatArray(matrixAInput);
            const matrixBData = parseFloatArray(matrixBInput);
            
            // Determine matrix dimensions (assuming square matrices for simplicity)
            const size = Math.sqrt(matrixAData.length);
            if (!Number.isInteger(size) || !Number.isInteger(Math.sqrt(matrixBData.length))) {
                throw new Error('Input matrices must be square for this demo');
            }
            
            const matrixA = createMatrix(matrixAData, size, size);
            const matrixB = createMatrix(matrixBData, size, size);
            
            // Perform matrix multiplication
            const result = performMatrixMultiplication(matrixA, matrixB);
            
            // Display the result
            let resultHTML = '<h4>BitNet Quantized Result Matrix:</h4><pre>';
            for (let i = 0; i < result.rows; i++) {
                for (let j = 0; j < result.cols; j++) {
                    resultHTML += result.data[i * result.cols + j].toFixed(4) + '\t';
                }
                resultHTML += '\n';
            }
            resultHTML += '</pre>';
            
            resultElement.innerHTML = resultHTML;
        } catch (e) {
            console.error('Error in matrix multiplication demo:', e);
            resultElement.innerHTML = `<span class="error">Error: ${e.message}</span>`;
        }
    });
}

// Set up the tensor transformation demo
function setupTensorTransformationDemo() {
    const runButton = document.getElementById('run-transform');
    if (!runButton) return;
    
    runButton.addEventListener('click', () => {
        const tensorInput = document.getElementById('tensor-data').value;
        const resultElement = document.getElementById('transform-result');
        
        try {
            // Parse input tensor
            const tensorData = parseFloatArray(tensorInput);
            
            // Perform tensor transformation
            const result = transformTensor(tensorData);
            
            // Display the result
            let resultHTML = '<h4>Original Tensor:</h4><pre>';
            resultHTML += result.original.map(x => x.toFixed(4)).join(', ');
            resultHTML += '</pre>';
            
            if (result.quantized) {
                resultHTML += '<h4>Quantized Tensor (1.58-bit):</h4><pre>';
                resultHTML += result.quantized.join(', ');
                resultHTML += '</pre>';
                
                resultHTML += '<h4>Dequantized Tensor:</h4><pre>';
                resultHTML += result.dequantized.map(x => x.toFixed(4)).join(', ');
                resultHTML += '</pre>';
                
                resultHTML += `<p><strong>Scale factor:</strong> ${result.scale.toFixed(4)}</p>`;
            }
            
            resultHTML += `<p>${result.message}</p>`;
            
            resultElement.innerHTML = resultHTML;
        } catch (e) {
            console.error('Error in tensor transformation demo:', e);
            resultElement.innerHTML = `<span class="error">Error: ${e.message}</span>`;
        }
    });
}

// Initialize the module when the page loads
document.addEventListener('DOMContentLoaded', async () => {
    try {
        const wasmModuleInstance = await ModuleFactory();
        onWasmInitialized(wasmModuleInstance);
    } catch (error) {
        console.error('Failed to initialize WASM module:', error);
        const statusElement = document.getElementById('status');
        if (statusElement) {
            statusElement.textContent = 'Failed to initialize WASM module.';
            statusElement.classList.add('error');
        }
    }
});