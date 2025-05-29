// example/main.js
import ModuleFactory from '../bitnet.js';

// Global variable to store the WASM module instance
let wasmModule = null;

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

// Helper function to allocate memory in the WASM heap
function allocateFloat32Array(array) {
    const bytes = array.length * Float32Array.BYTES_PER_ELEMENT;
    const ptr = wasmModule._malloc(bytes);
    const heap = new Float32Array(wasmModule.HEAPF32.buffer, ptr, array.length);
    heap.set(array);
    return { ptr, length: array.length, free: () => wasmModule._free(ptr) };
}

// Helper function to allocate memory for int8 array in the WASM heap
function allocateInt8Array(array) {
    const bytes = array.length * Int8Array.BYTES_PER_ELEMENT;
    const ptr = wasmModule._malloc(bytes);
    const heap = new Int8Array(wasmModule.HEAP8.buffer, ptr, array.length);
    heap.set(array);
    return { ptr, length: array.length, free: () => wasmModule._free(ptr) };
}

// Helper function to read a Float32Array from the WASM heap
function readFloat32Array(ptr, length) {
    return new Float32Array(wasmModule.HEAPF32.buffer, ptr, length);
}

// Helper function to read an Int8Array from the WASM heap
function readInt8Array(ptr, length) {
    return new Int8Array(wasmModule.HEAP8.buffer, ptr, length);
}

// Function to perform matrix multiplication using BitNet
function performMatrixMultiplication(matrixA, matrixB) {
    // Allocate memory for input matrices
    const inputPtr = allocateFloat32Array(matrixA.data);
    
    // Quantize the weights matrix
    // For each weight, we'll quantize to -1, 0, or 1
    const qWeightsData = new Int8Array(matrixB.rows * matrixB.cols);
    for (let i = 0; i < matrixB.data.length; i++) {
        const val = matrixB.data[i];
        if (Math.abs(val) < 0.05) {
            qWeightsData[i] = 0;
        } else if (val > 0) {
            qWeightsData[i] = 1;
        } else {
            qWeightsData[i] = -1;
        }
    }
    const qWeightsPtr = allocateInt8Array(qWeightsData);
    
    // Allocate memory for output matrix
    const outputData = new Float32Array(matrixA.rows * matrixB.cols);
    const outputPtr = allocateFloat32Array(outputData);
    
    // Allocate memory for scales
    const scalesData = new Float32Array(matrixB.rows);
    // Calculate scales based on the maximum absolute value in each row
    for (let i = 0; i < matrixB.rows; i++) {
        let maxVal = 0;
        for (let j = 0; j < matrixB.cols; j++) {
            const val = Math.abs(matrixB.data[i * matrixB.cols + j]);
            if (val > maxVal) maxVal = val;
        }
        scalesData[i] = maxVal > 0 ? maxVal : 1.0;
    }
    const scalesPtr = allocateFloat32Array(scalesData);
    
    // Allocate memory for LUT scales
    const lutScalesData = new Float32Array(1);
    lutScalesData[0] = 127.0; // Default LUT scale
    const lutScalesPtr = allocateFloat32Array(lutScalesData);
    
    // Allocate memory for LUT biases (not used in this implementation)
    const lutBiasesData = new Float32Array(1);
    lutBiasesData[0] = 0.0;
    const lutBiasesPtr = allocateFloat32Array(lutBiasesData);
    
    try {
        console.log("Calling ggml_bitnet_mul_mat_task_compute");
        // Call the BitNet matrix multiplication function
        wasmModule._ggml_bitnet_mul_mat_task_compute(
            inputPtr.ptr,
            scalesPtr.ptr,
            qWeightsPtr.ptr,
            lutScalesPtr.ptr,
            lutBiasesPtr.ptr,
            outputPtr.ptr,
            matrixA.rows,
            matrixA.cols,
            matrixB.cols,
            2 // 2-bit quantization
        );
        
        // Read the result
        const result = new Float32Array(matrixA.rows * matrixB.cols);
        const heapView = new Float32Array(wasmModule.HEAPF32.buffer, outputPtr.ptr, matrixA.rows * matrixB.cols);
        result.set(heapView);
        
        return createMatrix(result, matrixA.rows, matrixB.cols);
    } catch (e) {
        console.error("Error in matrix multiplication:", e);
        throw e;
    } finally {
        // Free allocated memory
        inputPtr.free();
        outputPtr.free();
        qWeightsPtr.free();
        scalesPtr.free();
        lutScalesPtr.free();
        lutBiasesPtr.free();
    }
}

// Function to transform a tensor using BitNet quantization
function transformTensor(tensorData) {
    // Allocate memory for input tensor
    const inputPtr = allocateFloat32Array(tensorData);
    
    // Allocate memory for output tensor (same size as input)
    const outputData = new Float32Array(tensorData.length);
    const outputPtr = allocateFloat32Array(outputData);
    
    try {
        console.log("Calling ggml_bitnet_transform_tensor");
        // Call the BitNet tensor transformation function
        wasmModule._ggml_bitnet_transform_tensor(
            inputPtr.ptr,
            outputPtr.ptr,
            tensorData.length,
            2 // 2-bit quantization
        );
        
        // Read the result
        const result = new Float32Array(tensorData.length);
        const heapView = new Float32Array(wasmModule.HEAPF32.buffer, outputPtr.ptr, tensorData.length);
        result.set(heapView);
        
        return {
            original: Array.from(tensorData),
            transformed: Array.from(result),
            message: "Tensor transformed successfully."
        };
    } catch (e) {
        console.error("Error in tensor transformation:", e);
        return {
            original: Array.from(tensorData),
            error: e.toString(),
            message: "Error transforming tensor."
        };
    } finally {
        // Free allocated memory
        inputPtr.free();
        outputPtr.free();
    }
}

// Function to load a model file
async function loadModelFile(modelPath) {
    const outputElement = document.getElementById('output');
    outputElement.innerHTML += `Loading model from ${modelPath}...<br>`;
    
    try {
        const response = await fetch(modelPath);
        if (!response.ok) {
            throw new Error(`Failed to fetch model: ${response.status} ${response.statusText}`);
        }
        
        const arrayBuffer = await response.arrayBuffer();
        outputElement.innerHTML += `Model loaded, size: ${(arrayBuffer.byteLength / (1024 * 1024)).toFixed(2)} MB<br>`;
        
        // Allocate memory in the WASM heap for the model data
        const modelDataPtr = wasmModule._malloc(arrayBuffer.byteLength);
        
        // Copy the model data to the WASM heap
        const modelDataHeap = new Uint8Array(wasmModule.HEAPU8.buffer, modelDataPtr, arrayBuffer.byteLength);
        modelDataHeap.set(new Uint8Array(arrayBuffer));
        
        outputElement.innerHTML += `Model data copied to WASM heap<br>`;
        
        return {
            ptr: modelDataPtr,
            size: arrayBuffer.byteLength,
            free: () => wasmModule._free(modelDataPtr)
        };
    } catch (error) {
        outputElement.innerHTML += `Error loading model: ${error.message}<br>`;
        console.error('Error loading model:', error);
        throw error;
    }
}

// Function to run inference with the loaded model
function runModelInference(modelData, inputText) {
    const outputElement = document.getElementById('output');
    outputElement.innerHTML += `Running inference with input: "${inputText}"<br>`;
    
    try {
        // For demonstration purposes, we'll just show that we've loaded the model
        // In a real implementation, we would:
        // 1. Tokenize the input text
        // 2. Create input tensors
        // 3. Run the model forward pass
        // 4. Process the output
        
        // Simulate tokenization by converting to simple character codes
        const tokens = Array.from(inputText).map(c => c.charCodeAt(0));
        outputElement.innerHTML += `Tokenized input (${tokens.length} tokens): ${tokens.slice(0, 10).join(', ')}...<br>`;
        
        // Simulate running the model
        outputElement.innerHTML += `Model size: ${(modelData.size / (1024 * 1024)).toFixed(2)} MB<br>`;
        outputElement.innerHTML += `Running BitNet inference (simulated)...<br>`;
        
        // In a real implementation, we would call the appropriate WASM functions here
        
        // Simulate output generation
        const outputText = `This is a simulated response from the BitNet model based on your input: "${inputText}". In a real implementation, we would process the model's output tokens and generate text.`;
        
        return outputText;
    } catch (error) {
        outputElement.innerHTML += `Error running inference: ${error.message}<br>`;
        console.error('Error running inference:', error);
        throw error;
    }
}

// This function will be called when the WASM module is loaded and initialized
function onWasmInitialized(wasmModuleInstance) {
    console.log('BitNet WASM Module Initialized and Ready.');
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
        console.log('Initializing BitNet...');
        outputElement.innerHTML += 'Initializing BitNet...<br>';
        
        // Initialize the WASM module
        if (typeof wasmModule._ggml_init === 'function') {
            wasmModule._ggml_init(0);
        } else {
            console.warn('_ggml_init function not found, skipping initialization');
        }
        
        if (typeof wasmModule._ggml_bitnet_init === 'function') {
            wasmModule._ggml_bitnet_init();
        } else {
            throw new Error('_ggml_bitnet_init function not found');
        }
        
        console.log('BitNet initialized successfully.');
        outputElement.innerHTML += 'BitNet initialized successfully.<br>';
    } catch (e) {
        console.error('Error initializing BitNet:', e);
        outputElement.innerHTML += `Error initializing BitNet: ${e}<br>`;
        if (statusElement) {
            statusElement.textContent = 'Error initializing BitNet.';
            statusElement.classList.add('error');
        }
        return;
    }
    
    // Set up event listeners for the demo buttons
    setupMatrixMultiplicationDemo();
    setupTensorTransformationDemo();
    setupModelInferenceDemo();
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
            let resultHTML = '<h4>Result Matrix:</h4><pre>';
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
            
            // Transform the tensor
            const result = transformTensor(tensorData);
            
            // Display the result
            let resultHTML = '<h4>Original Tensor:</h4><pre>';
            resultHTML += result.original.map(v => v.toFixed(4)).join(', ');
            resultHTML += '</pre>';
            resultHTML += `<p>${result.message}</p>`;
            
            resultElement.innerHTML = resultHTML;
        } catch (e) {
            console.error('Error in tensor transformation demo:', e);
            resultElement.innerHTML = `<span class="error">Error: ${e.message}</span>`;
        }
    });
}

// Set up the model inference demo
function setupModelInferenceDemo() {
    const runButton = document.getElementById('run-inference');
    if (!runButton) return;
    
    let modelData = null;
    
    // Load the model when the load button is clicked
    const loadButton = document.getElementById('load-model');
    if (loadButton) {
        loadButton.addEventListener('click', async () => {
            const modelPath = document.getElementById('model-path').value;
            const loadStatusElement = document.getElementById('load-status');
            
            try {
                loadStatusElement.textContent = 'Loading model...';
                loadStatusElement.className = '';
                
                // Free previous model data if it exists
                if (modelData) {
                    modelData.free();
                    modelData = null;
                }
                
                // Load the model
                modelData = await loadModelFile(modelPath);
                
                loadStatusElement.textContent = 'Model loaded successfully!';
                loadStatusElement.className = 'success';
                
                // Enable the run button
                runButton.disabled = false;
            } catch (e) {
                console.error('Error loading model:', e);
                loadStatusElement.textContent = `Error loading model: ${e.message}`;
                loadStatusElement.className = 'error';
                
                // Disable the run button
                runButton.disabled = true;
            }
        });
    }
    
    // Run inference when the run button is clicked
    runButton.addEventListener('click', () => {
        const inputText = document.getElementById('inference-input').value;
        const resultElement = document.getElementById('inference-result');
        
        try {
            if (!modelData) {
                throw new Error('Model not loaded. Please load a model first.');
            }
            
            // Run inference
            const outputText = runModelInference(modelData, inputText);
            
            // Display the result
            resultElement.innerHTML = `<h4>Model Output:</h4><div class="model-output">${outputText}</div>`;
        } catch (e) {
            console.error('Error in model inference demo:', e);
            resultElement.innerHTML = `<span class="error">Error: ${e.message}</span>`;
        }
    });
}

// Emscripten module configuration object
const moduleConfig = {
    print: function(text) {
        console.log('[WASM stdout]', text);
        const outputElement = document.getElementById('output');
        if (outputElement) outputElement.innerHTML += `[WASM stdout] ${text}<br>`;
    },
    printErr: function(text) {
        console.error('[WASM stderr]', text);
        const outputElement = document.getElementById('output');
        if (outputElement) outputElement.innerHTML += `[WASM stderr] ${text}<br>`;
    }
};

const initialStatus = document.getElementById('status');
const initialOutput = document.getElementById('output');

if (initialStatus) initialStatus.textContent = 'Loading BitNet WASM module...';
if (initialOutput) initialOutput.innerHTML = 'Loading BitNet WASM module...<br>';

function initializeWasm() {
    if (typeof ModuleFactory === 'function') {
        console.log('Module factory imported successfully. Initializing WASM...');
        if (initialOutput) initialOutput.innerHTML += 'Module factory imported successfully. Initializing WASM...<br>';
        
        ModuleFactory(moduleConfig).then((initializedInstance) => {
            onWasmInitialized(initializedInstance);
        }).catch(e => {
            console.error("Error initializing WASM module:", e);
            if (initialOutput) initialOutput.innerHTML += `Error initializing WASM module: ${e}<br>`;
            if (initialStatus) {
                initialStatus.textContent = 'Error initializing WASM module.';
                initialStatus.classList.add('error');
            }
        });
    } else {
        const currentTypeOfModuleFactory = typeof ModuleFactory;
        console.error(`Module factory not available or not a function (current type: ${currentTypeOfModuleFactory}). Check import and bitnet.js export.`);
        if (initialOutput) {
             initialOutput.innerHTML += `Module factory not available or not a function (current type: ${currentTypeOfModuleFactory}). Check import and bitnet.js export.<br>`;
        }
        if (initialStatus) {
            initialStatus.textContent = 'Error: Module factory not found.';
            initialStatus.classList.add('error');
        }
    }
}

// Initialize the WASM module when the document is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeWasm);
} else {
    initializeWasm();
}
