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
    if (!wasmModule || typeof wasmModule._malloc !== 'function') {
        console.error("WASM module or _malloc function not available");
        throw new Error("WASM module or _malloc function not available");
    }
    
    const bytes = array.length * Float32Array.BYTES_PER_ELEMENT;
    const ptr = wasmModule._malloc(bytes);
    
    if (!ptr) {
        console.error("Failed to allocate memory");
        throw new Error("Failed to allocate memory");
    }
    
    const heap = new Float32Array(wasmModule.HEAPF32.buffer, ptr, array.length);
    heap.set(array);
    
    return { 
        ptr, 
        length: array.length, 
        free: () => {
            if (wasmModule && typeof wasmModule._free === 'function') {
                wasmModule._free(ptr);
            }
        } 
    };
}

// Helper function to allocate memory for int8 array in the WASM heap
function allocateInt8Array(array) {
    if (!wasmModule || typeof wasmModule._malloc !== 'function') {
        console.error("WASM module or _malloc function not available");
        throw new Error("WASM module or _malloc function not available");
    }
    
    const bytes = array.length * Int8Array.BYTES_PER_ELEMENT;
    const ptr = wasmModule._malloc(bytes);
    
    if (!ptr) {
        console.error("Failed to allocate memory");
        throw new Error("Failed to allocate memory");
    }
    
    const heap = new Int8Array(wasmModule.HEAP8.buffer, ptr, array.length);
    heap.set(array);
    
    return { 
        ptr, 
        length: array.length, 
        free: () => {
            if (wasmModule && typeof wasmModule._free === 'function') {
                wasmModule._free(ptr);
            }
        } 
    };
}

// Helper function to read a Float32Array from the WASM heap
function readFloat32Array(ptr, length) {
    if (!wasmModule || !wasmModule.HEAPF32 || !wasmModule.HEAPF32.buffer) {
        console.error("WASM module or HEAPF32 not available");
        throw new Error("WASM module or HEAPF32 not available");
    }
    const result = new Float32Array(length);
    const heapView = new Float32Array(wasmModule.HEAPF32.buffer, ptr, length);
    result.set(heapView);
    return result;
}

// Helper function to read an Int8Array from the WASM heap
function readInt8Array(ptr, length) {
    if (!wasmModule || !wasmModule.HEAP8 || !wasmModule.HEAP8.buffer) {
        console.error("WASM module or HEAP8 not available");
        throw new Error("WASM module or HEAP8 not available");
    }
    const result = new Int8Array(length);
    const heapView = new Int8Array(wasmModule.HEAP8.buffer, ptr, length);
    result.set(heapView);
    return result;
}

// Function to perform matrix multiplication using BitNet
function performMatrixMultiplication(matrixA, matrixB) {
    try {
        console.log("Preparing for matrix multiplication");
        
        // Perform a simple matrix multiplication in JavaScript
        const result = new Float32Array(matrixA.rows * matrixB.cols);
        
        // Reshape matrices for multiplication
        const reshapedA = [];
        for (let i = 0; i < matrixA.rows; i++) {
            const row = [];
            for (let j = 0; j < matrixA.cols; j++) {
                row.push(matrixA.data[i * matrixA.cols + j]);
            }
            reshapedA.push(row);
        }
        
        const reshapedB = [];
        for (let i = 0; i < matrixB.rows; i++) {
            const row = [];
            for (let j = 0; j < matrixB.cols; j++) {
                row.push(matrixB.data[i * matrixB.cols + j]);
            }
            reshapedB.push(row);
        }
        
        // Quantize the weights matrix (for demonstration)
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
        
        // Calculate scales based on the maximum absolute value in each row
        const scalesData = new Float32Array(matrixB.rows);
        for (let i = 0; i < matrixB.rows; i++) {
            let maxVal = 0;
            for (let j = 0; j < matrixB.cols; j++) {
                const val = Math.abs(matrixB.data[i * matrixB.cols + j]);
                if (val > maxVal) maxVal = val;
            }
            scalesData[i] = maxVal > 0 ? maxVal : 1.0;
        }
        
        // Perform matrix multiplication with quantized weights
        for (let i = 0; i < matrixA.rows; i++) {
            for (let j = 0; j < matrixB.cols; j++) {
                let sum = 0;
                for (let k = 0; k < matrixA.cols; k++) {
                    // Get the quantized weight and apply the scale
                    const qWeight = qWeightsData[k * matrixB.cols + j];
                    const scale = scalesData[k];
                    const weight = qWeight * scale;
                    
                    // Multiply and accumulate
                    sum += matrixA.data[i * matrixA.cols + k] * weight;
                }
                result[i * matrixB.cols + j] = sum;
            }
        }
        
        console.log("Matrix multiplication completed (using JavaScript implementation)");
        
        return createMatrix(result, matrixA.rows, matrixB.cols);
    } catch (e) {
        console.error("Error in matrix multiplication:", e);
        throw e;
    }
}

// Function to transform a tensor using BitNet quantization
function transformTensor(tensorData) {
    try {
        console.log("Preparing for tensor transformation");
        
        // Perform a simple manual quantization for demonstration
        const transformedData = new Float32Array(tensorData.length);
        const maxVal = Math.max(...tensorData.map(Math.abs));
        
        console.log(`Max value in tensor: ${maxVal}`);
        
        for (let i = 0; i < tensorData.length; i++) {
            const val = tensorData[i];
            let qVal;
            
            if (Math.abs(val) < 0.1 * maxVal) {
                qVal = 0; // Zero
            } else if (val > 0) {
                qVal = 1; // Positive
            } else {
                qVal = -1; // Negative
            }
            
            transformedData[i] = qVal * maxVal;
        }
        
        return {
            original: Array.from(tensorData),
            transformed: Array.from(transformedData),
            message: "Tensor transformed successfully (using JavaScript implementation)."
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

    // Check available functions
    const availableFunctions = [];
    if (typeof wasmModule._ggml_init === 'function') availableFunctions.push('_ggml_init');
    if (typeof wasmModule._ggml_bitnet_init === 'function') availableFunctions.push('_ggml_bitnet_init');
    if (typeof wasmModule._ggml_bitnet_free === 'function') availableFunctions.push('_ggml_bitnet_free');
    if (typeof wasmModule._ggml_bitnet_mul_mat_task_compute === 'function') availableFunctions.push('_ggml_bitnet_mul_mat_task_compute');
    if (typeof wasmModule._ggml_bitnet_transform_tensor === 'function') availableFunctions.push('_ggml_bitnet_transform_tensor');
    
    console.log('Available WASM functions:', availableFunctions);
    outputElement.innerHTML += `Available WASM functions: ${availableFunctions.join(', ')}<br>`;

    // Initialize BitNet
    try {
        console.log('Initializing BitNet...');
        outputElement.innerHTML += 'Initializing BitNet...<br>';
        
        // Initialize the WASM module
        if (typeof wasmModule._ggml_init === 'function') {
            wasmModule._ggml_init(0);
            console.log('GGML initialized');
        } else {
            console.warn('_ggml_init function not found, skipping initialization');
            outputElement.innerHTML += 'Warning: _ggml_init function not found, skipping initialization<br>';
        }
        
        if (typeof wasmModule._ggml_bitnet_init === 'function') {
            wasmModule._ggml_bitnet_init();
            console.log('BitNet initialized');
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
            
            if (result.transformed) {
                resultHTML += '<h4>Transformed Tensor (Quantized):</h4><pre>';
                resultHTML += result.transformed.map(v => v.toFixed(4)).join(', ');
                resultHTML += '</pre>';
            }
            
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
