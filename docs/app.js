// BitNet-WASM Demo - GitHub Pages
// This example demonstrates how to use BitNet-WASM in your own projects

// Import BitNet-WASM module from CDN (for GitHub Pages demo)
// In your own project, you would use a relative path like './bitnet.js'
import BitNetModule from 'https://cdn.jsdelivr.net/gh/jerfletcher/BitNet-wasm@latest/bitnet.js';

// Initialize Dexie.js for IndexedDB model storage
const db = new Dexie('BitNetModelsDB');
db.version(1).stores({
    models: 'url, name, data, timestamp, size'
});

// Global variables
let bitnet = null;
let modelData = null;

// DOM elements
const statusElement = document.getElementById('status');
const outputElement = document.getElementById('output');
const dbStatusElement = document.getElementById('db-status');
const cacheDetailsElement = document.getElementById('cache-details');
const loadModelButton = document.getElementById('load-model');
let cachedModelSelect = document.getElementById('cached-model-select');
if (!cachedModelSelect) {
    cachedModelSelect = document.createElement('select');
    cachedModelSelect.id = 'cached-model-select';
    cachedModelSelect.style.marginLeft = '8px';
    loadModelButton.parentNode.insertBefore(cachedModelSelect, loadModelButton);
}
const runInferenceButton = document.getElementById('run-inference');
const loadStatusElement = document.getElementById('load-status');
const progressContainer = document.querySelector('.progress-container');
const progressBar = document.getElementById('download-progress');
const saveProgressContainer = document.createElement('div');
saveProgressContainer.className = 'progress-container';
saveProgressContainer.style.display = 'none';
const saveProgressBar = document.createElement('div');
saveProgressBar.className = 'progress-bar';
saveProgressBar.id = 'save-progress';
saveProgressBar.style.width = '0%';
saveProgressBar.textContent = '0%';
saveProgressContainer.appendChild(saveProgressBar);
progressContainer.parentNode.insertBefore(saveProgressContainer, progressContainer.nextSibling);

// Tab functionality
document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
        // Remove active class from all tabs and content
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        
        // Add active class to clicked tab and corresponding content
        tab.classList.add('active');
        const tabId = tab.getAttribute('data-tab');
        document.getElementById(`${tabId}-content`).classList.add('active');
    });
});

// Initialize BitNet-WASM
async function initBitNet() {
    try {
        updateStatus('Loading BitNet-WASM module...', 'loading');
        outputElement.textContent = 'Loading BitNet-WASM module...\n';
        
        // Initialize the module
        bitnet = await BitNetModule();
        
        updateStatus('Initializing BitNet engine...', 'loading');
        outputElement.textContent += 'BitNet-WASM module loaded. Initializing...\n';
        
        // Initialize BitNet
        if (typeof bitnet._bitnet_init === 'function') {
            bitnet._bitnet_init();
            outputElement.textContent += 'BitNet engine initialized successfully.\n';
        } else {
            outputElement.textContent += 'Warning: _bitnet_init function not found. Using default initialization.\n';
        }
        
        // Check available functions
        const availableFunctions = [];
        for (const key in bitnet) {
            if (typeof bitnet[key] === 'function' && key.startsWith('_')) {
                availableFunctions.push(key);
            }
        }
        outputElement.textContent += `Available WASM functions: ${availableFunctions.length} found\n`;
        
        // Update status and enable buttons
        updateStatus('BitNet-WASM initialized successfully!', 'success');
        loadModelButton.disabled = false;
        
        // Enable inference button (will run in demo mode if no model is loaded)
        runInferenceButton.disabled = false;
        
        // Add note about demo mode
        outputElement.textContent += 'Note: Inference will run in demo mode if no model is loaded.\n';
        
        // Check IndexedDB cache
        updateDBStatus();
        
        // Setup event listeners
        setupEventListeners();
        
    } catch (error) {
        console.error('Failed to initialize BitNet-WASM:', error);
        updateStatus(`Failed to initialize: ${error.message}`, 'error');
        outputElement.textContent += `Error: ${error.message}\n`;
    }
}

// Update status element
function updateStatus(message, type = 'loading') {
    statusElement.textContent = message;
    statusElement.className = `status ${type}`;
}

// Update load status
function updateLoadStatus(message, type = '') {
    loadStatusElement.textContent = message;
    loadStatusElement.className = `status ${type}`;
    loadStatusElement.style.display = message ? 'block' : 'none';
}

// Setup event listeners
function setupEventListeners() {
    // Load model button
    loadModelButton.addEventListener('click', loadModel);
    
    // Run inference button
    runInferenceButton.addEventListener('click', runInference);
    
    // Matrix multiplication button
    document.getElementById('run-matmul').addEventListener('click', runMatrixMultiplication);
    
    // IndexedDB cache buttons
    document.getElementById('view-cache').addEventListener('click', viewCachedModels);
    document.getElementById('clear-cache').addEventListener('click', clearModelCache);
}

// Load a model from URL with IndexedDB caching
async function loadModel() {
    let modelUrl = document.getElementById('model-url').value.trim();
    if (cachedModelSelect && cachedModelSelect.value) {
        modelUrl = cachedModelSelect.value;
        document.getElementById('model-url').value = modelUrl;
    }
    
    if (!modelUrl) {
        updateLoadStatus('Please enter a valid model URL', 'error');
        return;
    }
    
    try {
        // Disable buttons during loading
        loadModelButton.disabled = true;
        
        // Free previous model data if it exists
        if (modelData) {
            if (typeof modelData.free === 'function') {
                modelData.free();
            }
            modelData = null;
        }
        
        updateLoadStatus('Loading model...', 'loading');
        outputElement.textContent += `Loading model from ${modelUrl}...\n`;
        
        // Extract model name from URL
        const modelName = modelUrl.split('/').pop();
        
        // Check if model exists in IndexedDB (single or chunked)
        let cachedModels = await db.models.where('url').startsWith(modelUrl).toArray();
        if (cachedModels.length > 0) {
            // Sort by part number if chunked
            if (cachedModels.length > 1) {
                cachedModels.sort((a, b) => {
                    const getPart = url => parseInt((url.split('#part')[1] || '0'), 10);
                    return getPart(a.url) - getPart(b.url);
                });
            }
            // Combine chunks
            let totalSize = cachedModels.reduce((sum, m) => sum + (m.data.byteLength || m.size || 0), 0);
            let combined = new Uint8Array(totalSize);
            let pos = 0;
            for (const m of cachedModels) {
                combined.set(new Uint8Array(m.data), pos);
                pos += m.data.byteLength || m.size || 0;
            }
            outputElement.textContent += `Found cached model in IndexedDB: ${modelName} (${formatSize(totalSize)})\n`;
            updateLoadStatus('Using cached model from IndexedDB', 'success');
            modelData = await loadModelFromArrayBuffer(combined.buffer, modelUrl);
            loadModelButton.disabled = false;
            return;
        }
        
        // Model not in cache, download it
        outputElement.textContent += `Downloading model from ${modelUrl}...\n`;
        updateLoadStatus('Downloading model...', 'loading');
        
        // Show progress bar
        progressContainer.style.display = 'block';
        progressBar.style.width = '0%';
        progressBar.textContent = '0%';
        
        try {
            // Fetch with progress tracking
            const response = await fetch(modelUrl);
            if (!response.ok) {
                throw new Error(`Failed to fetch model: ${response.status}`);
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
            outputElement.textContent += `Model downloaded: ${formatSize(arrayBuffer.byteLength)}\n`;


            // Store in IndexedDB for future use (chunked save workaround)
            try {
                // Save in 16MB chunks to avoid transaction abort
                const CHUNK_SIZE = 16 * 1024 * 1024;
                let totalParts = 1;
                if (arrayBuffer.byteLength > CHUNK_SIZE) {
                    totalParts = Math.ceil(arrayBuffer.byteLength / CHUNK_SIZE);
                }
                let offset = 0;
                let part = 0;
                if (totalParts > 1) {
                    saveProgressContainer.style.display = 'block';
                }
                while (offset < arrayBuffer.byteLength) {
                    const end = Math.min(offset + CHUNK_SIZE, arrayBuffer.byteLength);
                    const chunkData = arrayBuffer.slice(offset, end);
                    await db.models.put({
                        url: totalParts > 1 ? `${modelUrl}#part${part}` : modelUrl,
                        name: totalParts > 1 ? `${modelName} (part ${part})` : modelName,
                        data: chunkData,
                        timestamp: new Date().getTime(),
                        size: chunkData.byteLength
                    });
                    part++;
                    offset = end;
                    if (totalParts > 1) {
                        const percent = Math.round((part / totalParts) * 100);
                        saveProgressBar.style.width = `${percent}%`;
                        saveProgressBar.textContent = `${percent}%`;
                    }
                }
                if (totalParts > 1) {
                    outputElement.textContent += `Model saved to IndexedDB in ${part} parts\n`;
                    saveProgressContainer.style.display = 'none';
                } else {
                    outputElement.textContent += `Model saved to IndexedDB cache\n`;
                }
                updateDBStatus();
            } catch (dbError) {
                console.warn('Failed to cache model in IndexedDB:', dbError);
                outputElement.textContent += `Warning: Could not cache model in IndexedDB: ${dbError.message}\n`;
            }

            // Hide progress bar
            progressContainer.style.display = 'none';

            // Load the model
            modelData = await loadModelFromArrayBuffer(arrayBuffer, modelUrl);

            // Update status
            updateLoadStatus('Model loaded successfully!', 'success');
            
        } catch (error) {
            // Hide progress bar on error
            progressContainer.style.display = 'none';
            
            outputElement.textContent += `Error loading model: ${error.message}\n`;
            updateLoadStatus(`Error loading model: ${error.message}`, 'error');
            console.error('Error loading model:', error);
            
            // Add note about demo mode
            outputElement.textContent += `You can still run inference in demo mode.\n`;
        }
        
    } catch (error) {
        outputElement.textContent += `Error: ${error.message}\n`;
        updateLoadStatus(`Error: ${error.message}`, 'error');
        console.error('Error in loadModel:', error);
    } finally {
        // Re-enable load button
        loadModelButton.disabled = false;
    }
}

// Load model from ArrayBuffer
async function loadModelFromArrayBuffer(arrayBuffer, modelUrl) {
    try {
        // Allocate memory in WASM
        const modelSize = arrayBuffer.byteLength;
        const modelPtr = bitnet._malloc(modelSize);
        
        if (!modelPtr) {
            throw new Error('Failed to allocate memory for model');
        }
        
        // Copy model data to WASM memory
        const heapBytes = new Uint8Array(bitnet.HEAPU8.buffer, modelPtr, modelSize);
        heapBytes.set(new Uint8Array(arrayBuffer));
        
        outputElement.textContent += `Model data copied to WASM heap\n`;
        
        // Load model
        let success = false;
        if (typeof bitnet._bitnet_load_model === 'function') {
            success = bitnet._bitnet_load_model(modelPtr, modelSize) === 1;
        } else if (typeof bitnet._load_model === 'function') {
            success = bitnet._load_model(modelPtr, modelSize) === 1;
        } else {
            throw new Error('Model loading function not found in WASM module');
        }
        
        // Free temporary memory
        bitnet._free(modelPtr);
        
        if (!success) {
            throw new Error('Failed to load model in WASM');
        }
        
        outputElement.textContent += `Model loaded successfully\n`;
        
        // Return model data reference
        return {
            url: modelUrl,
            size: modelSize
        };
        
    } catch (error) {
        outputElement.textContent += `Error processing model: ${error.message}\n`;
        console.error('Error processing model:', error);
        throw error;
    }
}

// Run inference with the loaded model
async function runInference() {
    const inputText = document.getElementById('inference-input').value.trim();
    const resultElement = document.getElementById('inference-result');
    
    if (!inputText) {
        resultElement.textContent = 'Please enter some input text';
        return;
    }
    
    try {
        resultElement.textContent = 'Running inference...';
        
        // Check if model is loaded
        let isModelLoaded = false;
        if (modelData) {
            // Check if model is loaded in WASM
            if (typeof bitnet._bitnet_is_model_loaded === 'function') {
                isModelLoaded = bitnet._bitnet_is_model_loaded() === 1;
            } else if (typeof bitnet._is_model_loaded === 'function') {
                isModelLoaded = bitnet._is_model_loaded() === 1;
            }
        }
        
        // If no model is loaded, use demo mode
        if (!isModelLoaded) {
            outputElement.textContent += `No model loaded. Running in demo mode.\n`;
            
            // Simulate inference for demo purposes
            await new Promise(resolve => setTimeout(resolve, 1000));
            
            // Generate a simulated response
            const demoResponses = [
                "is becoming increasingly efficient and accessible through innovations like BitNet's 2-bit quantization, which enables neural networks to run directly in web browsers with minimal resource requirements.",
                "will be characterized by models that can run efficiently on edge devices. BitNet demonstrates this by using 2-bit quantization to dramatically reduce model size while maintaining reasonable performance.",
                "depends on making powerful models accessible to everyone. BitNet's approach of running directly in the browser with WebAssembly shows how AI can become more democratized and privacy-preserving."
            ];
            
            const randomResponse = demoResponses[Math.floor(Math.random() * demoResponses.length)];
            const result = inputText + " " + randomResponse;
            
            // Display simulated result
            resultElement.innerHTML = `<strong>Input:</strong>\n${inputText}\n\n<strong>Output (DEMO MODE):</strong>\n${result}\n\n<em>Note: This is a simulated response as no model is currently loaded.</em>`;
            
            // Enable the load model button with a note
            updateLoadStatus('Please load a real model for actual inference', 'warning');
            return;
        }
        
        outputElement.textContent += `Running inference with input: "${inputText.substring(0, 50)}${inputText.length > 50 ? '...' : ''}"\n`;
        
        // Allocate memory for input and output
        const inputBytes = bitnet.lengthBytesUTF8(inputText) + 1;
        const inputPtr = bitnet._malloc(inputBytes);
        const maxOutputLength = 1024; // Adjust based on your needs
        const outputPtr = bitnet._malloc(maxOutputLength);
        
        // Copy input text to WASM memory
        bitnet.stringToUTF8(inputText, inputPtr, inputBytes);
        
        // Run inference
        let outputLength = 0;
        const startTime = performance.now();
        
        if (typeof bitnet._bitnet_inference_run === 'function') {
            outputLength = bitnet._bitnet_inference_run(inputPtr, outputPtr, maxOutputLength);
        } else if (typeof bitnet._run_inference === 'function') {
            outputLength = bitnet._run_inference(inputPtr, outputPtr, maxOutputLength);
        } else {
            throw new Error('Inference function not found in WASM module');
        }
        
        const endTime = performance.now();
        const inferenceTime = (endTime - startTime).toFixed(2);
        
        // Read output from WASM memory
        let result = '';
        if (outputLength > 0) {
            const buffer = new Uint8Array(bitnet.HEAPU8.buffer, outputPtr, outputLength);
            let str = '';
            for (let i = 0; i < outputLength; i++) {
                if (buffer[i] === 0) break;
                str += String.fromCharCode(buffer[i]);
            }
            result = str;
        }
        
        // Free memory
        bitnet._free(inputPtr);
        bitnet._free(outputPtr);
        
        // Display result
        outputElement.textContent += `Inference completed in ${inferenceTime}ms\n`;
        resultElement.innerHTML = `<strong>Input:</strong>\n${inputText}\n\n<strong>Output:</strong>\n${result}\n\n<em>Inference time: ${inferenceTime}ms</em>`;
        
    } catch (error) {
        resultElement.textContent = `Error running inference: ${error.message}`;
        outputElement.textContent += `Error running inference: ${error.message}\n`;
        console.error('Error running inference:', error);
    }
}

// Run matrix multiplication
function runMatrixMultiplication() {
    const matrixAText = document.getElementById('matrix-a').value;
    const matrixBText = document.getElementById('matrix-b').value;
    const resultElement = document.getElementById('matmul-result');
    
    try {
        // Parse input matrices
        const matrixA = parseMatrix(matrixAText);
        const matrixB = parseMatrix(matrixBText);
        
        // Validate matrices
        if (matrixA.length !== 9 || matrixB.length !== 9) {
            throw new Error('Both matrices must have 9 elements (3x3)');
        }
        
        // Perform matrix multiplication (simulated for demo)
        // In a real implementation, you would use BitNet's WASM functions
        const result = simulateMatrixMultiplication(matrixA, matrixB);
        
        // Format output
        let output = 'Original Matrix A (3x3):\n';
        for (let i = 0; i < 3; i++) {
            output += matrixA.slice(i * 3, (i + 1) * 3).map(x => x.toFixed(4)).join('  ') + '\n';
        }
        
        output += '\nOriginal Matrix B (3x3):\n';
        for (let i = 0; i < 3; i++) {
            output += matrixB.slice(i * 3, (i + 1) * 3).map(x => x.toFixed(4)).join('  ') + '\n';
        }
        
        output += '\nResult (A × B):\n';
        for (let i = 0; i < 3; i++) {
            output += result.slice(i * 3, (i + 1) * 3).map(x => x.toFixed(4)).join('  ') + '\n';
        }
        
        resultElement.textContent = output;
        
    } catch (error) {
        resultElement.textContent = `Error: ${error.message}`;
        console.error('Error in matrix multiplication:', error);
    }
}

// Parse matrix from text input
function parseMatrix(text) {
    return text.split(',')
        .map(x => x.trim())
        .filter(x => x.length > 0)
        .map(x => parseFloat(x));
}

// Simulate matrix multiplication (3x3)
function simulateMatrixMultiplication(matrixA, matrixB) {
    const result = new Array(9).fill(0);
    
    for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
            let sum = 0;
            for (let k = 0; k < 3; k++) {
                sum += matrixA[i * 3 + k] * matrixB[k * 3 + j];
            }
            result[i * 3 + j] = sum;
        }
    }
    
    return result;
}

// Update IndexedDB status
async function updateDBStatus() {
    try {
        const count = await db.models.count();
        let totalSize = 0;
        
        if (count > 0) {
            const models = await db.models.toArray();
            totalSize = models.reduce((sum, model) => sum + (model.size || 0), 0);
            dbStatusElement.textContent = `${count} model(s) in cache, total size: ${formatSize(totalSize)}`;
            dbStatusElement.className = 'success';
            // Populate cached model select dropdown
            const modelGroups = {};
            models.forEach(model => {
                const baseUrl = model.url.split('#part')[0];
                if (!modelGroups[baseUrl]) modelGroups[baseUrl] = [];
                modelGroups[baseUrl].push(model);
            });
            let options = '<option value="">-- Select cached model --</option>';
            for (const baseUrl in modelGroups) {
                const group = modelGroups[baseUrl];
                const totalSize = group.reduce((sum, m) => sum + (m.size || 0), 0);
                options += `<option value="${baseUrl}">${group[0].name.replace(/ \(part.*\)/, '')} (${formatSize(totalSize)})</option>`;
            }
            cachedModelSelect.innerHTML = options;
        } else {
            dbStatusElement.textContent = 'No models in cache';
            dbStatusElement.className = '';
            if (cachedModelSelect) cachedModelSelect.innerHTML = '<option value="">-- No cached models --</option>';
        }
    } catch (error) {
        dbStatusElement.textContent = `Error checking cache: ${error.message}`;
        dbStatusElement.className = 'error';
        console.error('Error checking IndexedDB:', error);
    }
}

// View cached models
async function viewCachedModels() {
    try {
        const models = await db.models.toArray();
        
        if (models.length === 0) {
            cacheDetailsElement.textContent = 'No cached models found.';
            cacheDetailsElement.style.display = 'block';
            return;
        }
        
        // Group models by base URL (remove #partN)
        const modelGroups = {};
        models.forEach(model => {
            const baseUrl = model.url.split('#part')[0];
            if (!modelGroups[baseUrl]) modelGroups[baseUrl] = [];
            modelGroups[baseUrl].push(model);
        });
        let details = 'Cached Models:\n\n';
        let idx = 1;
        for (const baseUrl in modelGroups) {
            const group = modelGroups[baseUrl];
            // Sort by part
            group.sort((a, b) => {
                const getPart = url => parseInt((url.split('#part')[1] || '0'), 10);
                return getPart(a.url) - getPart(b.url);
            });
            const totalSize = group.reduce((sum, m) => sum + (m.size || 0), 0);
            const date = new Date(group[0].timestamp);
            details += `${idx}. ${group[0].name.replace(/ \(part.*\)/, '')}\n`;
            details += `   URL: ${baseUrl}\n`;
            details += `   Size: ${formatSize(totalSize)}\n`;
            details += `   Cached: ${date.toLocaleString()}\n`;
            details += `   <button onclick="loadCachedModel('${baseUrl}')">Load</button>\n\n`;
            idx++;
        }
        cacheDetailsElement.innerHTML = `<pre>${details}</pre>`;
        cacheDetailsElement.style.display = 'block';
        window.loadCachedModel = async function(baseUrl) {
            document.getElementById('model-url').value = baseUrl;
            await loadModel();
        };
        
    } catch (error) {
        cacheDetailsElement.textContent = `Error retrieving cached models: ${error.message}`;
        cacheDetailsElement.style.display = 'block';
        console.error('Error retrieving cached models:', error);
    }
}

// Clear model cache
async function clearModelCache() {
    try {
        if (confirm('Are you sure you want to clear all cached models?')) {
            dbStatusElement.textContent = 'Clearing cache...';
            dbStatusElement.className = 'loading';
            cacheDetailsElement.textContent = '';
            cacheDetailsElement.style.display = 'none';
            // Optionally show a spinner
            dbStatusElement.innerHTML = '<span class="spinner" style="display:inline-block;width:16px;height:16px;border:2px solid #ccc;border-top:2px solid #333;border-radius:50%;animation:spin 1s linear infinite;vertical-align:middle;margin-right:6px;"></span>Clearing cache...';
            await db.models.clear();
            dbStatusElement.textContent = 'Cache cleared successfully';
            dbStatusElement.className = 'success';
            cacheDetailsElement.textContent = 'No cached models found.';
            cacheDetailsElement.style.display = 'block';
            updateDBStatus();
        }
    } catch (error) {
        dbStatusElement.textContent = `Error clearing cache: ${error.message}`;
        dbStatusElement.className = 'error';
        console.error('Error clearing cache:', error);
    }
// Spinner animation CSS
const style = document.createElement('style');
style.innerHTML = `@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }`;
document.head.appendChild(style);
}

// Format file size
function formatSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', initBitNet);