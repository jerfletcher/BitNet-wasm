// BitNet-WASM GitHub Pages Demo
import ModuleFactory from './bitnet.js';

// Global variable to store the WASM module instance
let wasmModule = null;

// Initialize Dexie.js database for model storage
const db = new Dexie('BitNetModelsDB');
db.version(1).stores({
    models: 'url, name, data, timestamp'
});

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

// Function to load a model file with IndexedDB caching (auto-convert GGUF to flat binary if needed)
async function loadModelFile(modelUrl) {
    const outputElement = document.getElementById('output');
    const loadStatusElement = document.getElementById('load-status');
    const progressContainer = document.querySelector('.progress-container');
    const progressBar = document.getElementById('download-progress');
    
    try {
        // Extract model name from URL
        const modelName = modelUrl.split('/').pop();
        outputElement.innerHTML += `Loading model: ${modelName}<br>`;
        
        // Check if model exists in IndexedDB
        const cachedModel = await db.models.get({ url: modelUrl });
        
        if (cachedModel) {
            outputElement.innerHTML += `Found cached model in IndexedDB (${(cachedModel.data.byteLength / (1024 * 1024)).toFixed(2)} MB)<br>`;
            loadStatusElement.textContent = 'Using cached model from IndexedDB';
            loadStatusElement.className = 'success';
            
            // Use the cached model data
            return processModelData(cachedModel.data, modelUrl);
        }
        
        // Model not in cache, download it
        outputElement.innerHTML += `Downloading model from ${modelUrl}...<br>`;
        loadStatusElement.textContent = 'Downloading model...';
        loadStatusElement.className = '';
        
        // Show progress bar
        progressContainer.style.display = 'block';
        progressBar.style.width = '0%';
        progressBar.textContent = '0%';
        
        // Fetch with progress tracking
        const response = await fetch(modelUrl);
        if (!response.ok) {
            throw new Error(`Failed to fetch model: ${response.status} ${response.statusText}`);
        }
        
        const contentLength = response.headers.get('content-length');
        const total = contentLength ? parseInt(contentLength, 10) : 0;
        let loaded = 0;
        
        const reader = response.body.getReader();
        const chunks = [];
        
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            chunks.push(value);
            loaded += value.length;
            
            if (total) {
                const percent = Math.round((loaded / total) * 100);
                progressBar.style.width = `${percent}%`;
                progressBar.textContent = `${percent}%`;
            }
        }
        
        // Combine chunks into a single ArrayBuffer
        const chunksAll = new Uint8Array(loaded);
        let position = 0;
        for (const chunk of chunks) {
            chunksAll.set(chunk, position);
            position += chunk.length;
        }
        
        const arrayBuffer = chunksAll.buffer;
        outputElement.innerHTML += `Model downloaded, size: ${(arrayBuffer.byteLength / (1024 * 1024)).toFixed(2)} MB<br>`;
        
        // Store in IndexedDB for future use
        try {
            await db.models.put({
                url: modelUrl,
                name: modelName,
                data: arrayBuffer,
                timestamp: new Date().getTime()
            });
            outputElement.innerHTML += `Model saved to IndexedDB cache<br>`;
        } catch (dbError) {
            console.warn('Failed to cache model in IndexedDB:', dbError);
            outputElement.innerHTML += `Warning: Could not cache model in IndexedDB: ${dbError.message}<br>`;
        }
        
        // Hide progress bar
        progressContainer.style.display = 'none';
        
        // Process the model data
        return processModelData(arrayBuffer, modelUrl);
    } catch (error) {
        // Hide progress bar on error
        if (progressContainer) progressContainer.style.display = 'none';
        
        outputElement.innerHTML += `Error loading model: ${error.message}<br>`;
        console.error('Error loading model:', error);
        throw error;
    }
}

// Helper function to process model data (convert GGUF to flat binary if needed)
function processModelData(arrayBuffer, modelUrl) {
    const outputElement = document.getElementById('output');
    
    try {
        // Check file extension
        const ext = modelUrl.split('.').pop().toLowerCase();
        
        if (ext === 'bin') {
            // Flat binary, just copy to WASM
            if (!wasmModule.HEAPU8 || !wasmModule.HEAPU8.buffer) {
                throw new Error("WASM memory not available. Module not initialized?");
            }
            const modelDataPtr = wasmModule._malloc(arrayBuffer.byteLength);
            if (!modelDataPtr) {
                throw new Error("Failed to allocate memory in WASM heap for model data.");
            }
            const modelDataHeap = new Uint8Array(wasmModule.HEAPU8.buffer, modelDataPtr, arrayBuffer.byteLength);
            modelDataHeap.set(new Uint8Array(arrayBuffer));
            outputElement.innerHTML += `Model data copied to WASM heap<br>`;
            return {
                ptr: modelDataPtr,
                size: arrayBuffer.byteLength,
                free: () => wasmModule._free(modelDataPtr)
            };
        } else if (ext === 'gguf') {
            // GGUF: convert to flat binary in WASM
            if (typeof wasmModule._gguf_to_flat !== 'function') {
                throw new Error('WASM gguf_to_flat function not found. Rebuild WASM with GGUF support.');
            }
            // Allocate GGUF buffer in WASM
            const ggufPtr = wasmModule._malloc(arrayBuffer.byteLength);
            wasmModule.HEAPU8.set(new Uint8Array(arrayBuffer), ggufPtr);
            // Guess output buffer size (2x input size)
            const flatCapacity = arrayBuffer.byteLength * 2;
            const flatPtr = wasmModule._malloc(flatCapacity);
            // Call conversion
            const flatSize = wasmModule._gguf_to_flat(ggufPtr, arrayBuffer.byteLength, flatPtr, flatCapacity);
            if (flatSize <= 0) {
                wasmModule._free(ggufPtr);
                wasmModule._free(flatPtr);
                throw new Error('GGUF to flat conversion failed in WASM.');
            }
            outputElement.innerHTML += `GGUF converted to flat binary in WASM (${flatSize} bytes)<br>`;
            // Free GGUF buffer
            wasmModule._free(ggufPtr);
            // Return flat buffer pointer
            return {
                ptr: flatPtr,
                size: flatSize,
                free: () => wasmModule._free(flatPtr)
            };
        } else {
            throw new Error('Unsupported model file type: ' + ext);
        }
    } catch (error) {
        outputElement.innerHTML += `Error processing model: ${error.message}<br>`;
        console.error('Error processing model:', error);
        throw error;
    }
}

// Function to run inference with the loaded model
function runModelInference(modelData, inputText) {
    const outputElement = document.getElementById('output');
    outputElement.innerHTML += `Running inference with input: "${inputText}"<br>`;
    try {
        if (!wasmModule.HEAP32 || !wasmModule.HEAP32.buffer) {
            throw new Error("WASM HEAP32 not available. Check build flags and exports.");
        }
        // Tokenize input as int32 tokens (char codes for demo)
        const tokens = Array.from(inputText).map(c => c.charCodeAt(0));
        outputElement.innerHTML += `Tokenized input (${tokens.length} tokens): ${tokens.slice(0, 10).join(', ')}...<br>`;
        // Allocate input and output buffers in WASM
        const inputBytes = tokens.length * 4;
        const inputPtr = wasmModule._malloc(inputBytes);
        const outputMaxTokens = 64;
        const outputBytes = outputMaxTokens * 4;
        const outputPtr = wasmModule._malloc(outputBytes);
        // Copy input tokens to WASM
        new Int32Array(wasmModule.HEAP32.buffer, inputPtr, tokens.length).set(tokens);
        // Call WASM inference function
        const nOut = wasmModule._bitnet_wasm_infer(
            modelData.ptr,
            inputPtr,
            tokens.length,
            outputPtr,
            outputMaxTokens
        );
        // Read output tokens from WASM
        const outTokens = Array.from(new Int32Array(wasmModule.HEAP32.buffer, outputPtr, nOut));
        // Free WASM memory
        wasmModule._free(inputPtr);
        wasmModule._free(outputPtr);
        outputElement.innerHTML += `WASM inference returned ${nOut} tokens: ${outTokens.slice(0, 10).join(', ')}...<br>`;
        // Convert output tokens to string (char codes for demo)
        const outputText = String.fromCharCode(...outTokens);
        return outputText;
    } catch (error) {
        outputElement.innerHTML += `Error running inference: ${error.message}<br>`;
        console.error('Error running inference:', error);
        throw error;
    }
}

// This function will be called when the WASM module is loaded and initialized
function onWasmInitialized(wasmModuleInstance) {
    wasmModule = wasmModuleInstance;
    // Wait for Emscripten runtime to be ready
    if (wasmModule.onRuntimeInitialized) {
        wasmModule.onRuntimeInitialized = () => {
            finishWasmInit();
        };
    } else {
        // If already initialized, just proceed
        finishWasmInit();
    }
}

function finishWasmInit() {
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
    setupMatrixMultiplicationDemo();
    setupTensorTransformationDemo();
    setupModelInferenceDemo();
    const loadButton = document.getElementById('load-model');
    if (loadButton) loadButton.disabled = false;
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
        // Enable the load button immediately
        loadButton.disabled = false;
        
        // Add database info to output
        const outputElement = document.getElementById('output');
        db.models.count().then(count => {
            if (count > 0) {
                outputElement.innerHTML += `Found ${count} model(s) in IndexedDB cache<br>`;
            }
        }).catch(err => {
            console.warn('Error checking IndexedDB:', err);
        });
        
        loadButton.addEventListener('click', async () => {
            const modelUrl = document.getElementById('model-path').value;
            const loadStatusElement = document.getElementById('load-status');
            
            try {
                loadStatusElement.textContent = 'Loading model...';
                loadStatusElement.className = '';
                
                // Free previous model data if it exists
                if (modelData) {
                    modelData.free();
                    modelData = null;
                }
                
                // Load the model with caching
                modelData = await loadModelFile(modelUrl);
                
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

// Function to manage the IndexedDB cache (clean old models)
async function manageModelCache() {
    try {
        const MAX_CACHE_AGE_DAYS = 30; // Keep models for 30 days
        const MAX_CACHE_SIZE_MB = 500; // Maximum cache size in MB
        
        // Get all models from the database
        const allModels = await db.models.toArray();
        
        if (allModels.length === 0) return;
        
        const now = new Date().getTime();
        const maxAgeMs = MAX_CACHE_AGE_DAYS * 24 * 60 * 60 * 1000;
        
        // Calculate total cache size
        let totalSizeMB = 0;
        const modelSizes = allModels.map(model => {
            const sizeMB = model.data.byteLength / (1024 * 1024);
            totalSizeMB += sizeMB;
            return {
                url: model.url,
                sizeMB,
                age: now - model.timestamp
            };
        });
        
        console.log(`IndexedDB cache: ${allModels.length} models, total size: ${totalSizeMB.toFixed(2)} MB`);
        
        // Models to delete (old models first, then by size if cache is too large)
        const toDelete = [];
        
        // First, mark old models for deletion
        for (const model of modelSizes) {
            if (model.age > maxAgeMs) {
                toDelete.push(model.url);
            }
        }
        
        // If cache is still too large, delete more models (oldest first)
        if (totalSizeMB > MAX_CACHE_SIZE_MB) {
            // Sort by age (oldest first)
            modelSizes.sort((a, b) => b.age - a.age);
            
            let currentSize = totalSizeMB;
            for (const model of modelSizes) {
                if (currentSize <= MAX_CACHE_SIZE_MB) break;
                if (!toDelete.includes(model.url)) {
                    toDelete.push(model.url);
                    currentSize -= model.sizeMB;
                }
            }
        }
        
        // Delete marked models
        if (toDelete.length > 0) {
            console.log(`Cleaning IndexedDB cache: removing ${toDelete.length} models`);
            await db.models.bulkDelete(toDelete);
        }
    } catch (error) {
        console.warn('Error managing model cache:', error);
    }
}

// Initialize the WASM module when the document is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        initializeWasm();
        // Manage cache after a short delay
        setTimeout(manageModelCache, 2000);
    });
} else {
    initializeWasm();
    // Manage cache after a short delay
    setTimeout(manageModelCache, 2000);
}
