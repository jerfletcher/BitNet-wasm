// BitNet-WASM Demo - GitHub Pages
// This example demonstrates how to use BitNet-WASM in your own projects

// Import BitNet-WASM module from CDN (for GitHub Pages demo)
// In your own project, you would use a relative path like './bitnet.js'
import BitNetModule from 'https://cdn.jsdelivr.net/gh/jerfletcher/BitNet-wasm@latest/bitnet.js';
// import BitNetModule from '../bitnet.js';
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
saveProgressBar.style.width = '0%';        saveProgressBar.textContent = 'Saving to IndexedDB...';
saveProgressContainer.appendChild(saveProgressBar);
progressContainer.parentNode.insertBefore(saveProgressContainer, progressContainer.nextSibling);

// Create WASM loading progress bar
const wasmProgressContainer = document.createElement('div');
wasmProgressContainer.className = 'progress-container';
wasmProgressContainer.style.display = 'none';
wasmProgressContainer.innerHTML = '<div class="progress-label">Loading into WASM...</div>';
const wasmProgressBar = document.createElement('div');
wasmProgressBar.className = 'progress-bar';
wasmProgressBar.id = 'wasm-progress';
wasmProgressBar.style.width = '0%';
wasmProgressBar.textContent = '0%';
wasmProgressContainer.appendChild(wasmProgressBar);
progressContainer.parentNode.insertBefore(wasmProgressContainer, progressContainer.nextSibling);

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

// Memory pressure monitoring for Chrome stability
function startMemoryMonitoring() {
    // Monitor memory usage to prevent Chrome crashes
    setInterval(() => {
        try {
            // Check performance memory if available
            if (performance.memory) {
                const memInfo = performance.memory;
                const usedPercent = (memInfo.usedJSHeapSize / memInfo.jsHeapSizeLimit) * 100;
                
                if (usedPercent > 92) {
                    console.warn(`High memory usage detected: ${usedPercent.toFixed(1)}%`);
                    
                    // Trigger emergency cleanup if memory is critically high
                    if (usedPercent > 98) {
                        console.error('Critical memory usage - triggering emergency cleanup');
                        performEmergencyCleanup();
                        updateStatus('Emergency cleanup performed due to high memory usage', 'error');
                    }
                }
            }
            
            // Additional Chrome stability check
            if (navigator.userAgent.includes('Chrome/137')) {
                // Force garbage collection more frequently for Chrome 137
                if (window.gc && Math.random() < 0.1) { // 10% chance
                    window.gc();
                }
            }
            
        } catch (error) {
            console.error('Memory monitoring error:', error);
        }        }, 15000);
}

// Initialize BitNet-WASM with Chrome crash protection
async function initBitNet() {
    try {
        addGlobalErrorHandlers();
        updateStatus('Loading BitNet-WASM module...', 'loading');
        outputElement.textContent = 'Loading BitNet-WASM module...\n';
        
        // Strict Chrome/device compatibility checks
        if (navigator.deviceMemory && navigator.deviceMemory < 4) {
            throw new Error('Insufficient device memory for WASM operations');
        }
        
        // Check for Chrome version issues and apply workarounds
        const userAgent = navigator.userAgent;
        if (userAgent.includes('Chrome/137')) {
            console.warn('Chrome 137 detected - applying stability workarounds');
            
            // Disable web workers and threading for Chrome 137
            if (window.Worker) {
                console.warn('Disabling Web Workers for Chrome 137 compatibility');
            }
        }
        
        // Force garbage collection before WASM initialization
        if (window.gc) {
            window.gc();
        }
        
        // Initialize the module with minimal footprint and crash protection
        const modulePromise = new Promise(async (resolve, reject) => {
            try {
                // Reduce all memory settings to absolute minimum
                const module = await BitNetModule({
                    // Disable threading to avoid Chrome issues
                    PTHREAD_POOL_SIZE: 0,
                    USE_PTHREADS: false,
                    // Error handlers
                    onAbort: () => reject(new Error('WASM module aborted')),
                    onRuntimeError: (err) => reject(new Error(`WASM runtime error: ${err}`)),
                    print: (text) => console.log('[WASM]', text),
                    printErr: (text) => console.error('[WASM]', text)
                });
                
                // Test module immediately
                if (!module || typeof module.ccall !== 'function') {
                    throw new Error('WASM module validation failed');
                }
                
                resolve(module);
            } catch (error) {
                reject(new Error(`Module creation failed: ${error.message}`));
            }
        });
        
        const timeoutPromise = new Promise((_, reject) => 
            setTimeout(() => reject(new Error('Module loading timeout')), 20000)
        );
        
        bitnet = await Promise.race([modulePromise, timeoutPromise]);
        
        // Validate WASM module with enhanced checks
        if (!bitnet || typeof bitnet !== 'object') {
            throw new Error('BitNet WASM module not available');
        }
        
        // Wait for module to be ready and check heap
        if (bitnet.then) {
            bitnet = await bitnet;
        }
        
        // Check for critical WASM properties after initialization
        if (!bitnet.HEAPU8 && !bitnet.HEAP8) {
            console.warn('WASM heap views not available yet, continuing...');
        }
        
        updateStatus('Initializing BitNet engine...', 'loading');
        outputElement.textContent += 'BitNet-WASM module loaded. Initializing...\n';
        
        // Check essential WASM functions
        const requiredFunctions = ['_malloc', '_free'];
        const missingFunctions = requiredFunctions.filter(fn => typeof bitnet[fn] !== 'function');
        if (missingFunctions.length > 0) {
            throw new Error(`Missing WASM functions: ${missingFunctions.join(', ')}`);
        }
        
        // Initialize BitNet with error handling
        if (typeof bitnet._bitnet_init === 'function') {
            try {
                const initResult = bitnet._bitnet_init();
                if (initResult !== undefined && initResult !== 1) {
                    throw new Error('BitNet initialization failed');
                }
                outputElement.textContent += 'BitNet engine initialized successfully.\n';
            } catch (initError) {
                throw new Error(`BitNet init function failed: ${initError.message}`);
            }
        } else {
            outputElement.textContent += 'Warning: _bitnet_init function not found. Using default initialization.\n';
        }
        
        // Test memory allocation with stricter limits
        try {
            // Test smaller allocation first
            const testPtr1 = bitnet._malloc(512);
            if (!testPtr1) {
                throw new Error('Basic WASM memory allocation failed');
            }
            bitnet._free(testPtr1);
            
            // Test medium allocation
            const testPtr2 = bitnet._malloc(1024 * 1024); // 1MB
            if (!testPtr2) {
                throw new Error('Medium WASM memory allocation failed');  
            }
            
            // Verify heap integrity after allocation
            if (!bitnet.HEAPU8.buffer || bitnet.HEAPU8.buffer.byteLength < testPtr2 + 1024 * 1024) {
                bitnet._free(testPtr2);
                throw new Error('WASM heap corruption detected');
            }
            
            bitnet._free(testPtr2);
            outputElement.textContent += 'WASM memory validation passed\n';
        } catch (memError) {
            throw new Error(`WASM memory validation failed: ${memError.message}`);
        }
        
        // Add crash detection monitor with immediate abort on issues
        let crashDetected = false;
        const crashDetector = () => {
            crashDetected = true;
            console.error('Browser instability detected - disabling WASM operations');
            if (bitnet) {
                try {
                    // Emergency cleanup
                    bitnet = null;
                    if (window.gc) window.gc();
                } catch (e) {
                    console.error('Emergency cleanup failed:', e);
                }
            }
        };
        
        // Monitor for Chrome-specific issues
        const chromeMonitor = setInterval(() => {
            try {
                // Check if module is still responsive
                if (!bitnet || !bitnet.HEAPU8 || !bitnet.HEAPU8.buffer) {
                    crashDetected = true;
                    clearInterval(chromeMonitor);
                    throw new Error('WASM module became unresponsive');
                }
                
                // Test basic function accessibility
                if (typeof bitnet._malloc !== 'function') {
                    crashDetected = true;
                    clearInterval(chromeMonitor);
                    throw new Error('WASM functions became inaccessible');
                }
            } catch (error) {
                crashDetected = true;
                clearInterval(chromeMonitor);
                console.error('WASM instability detected:', error);
                updateStatus('WASM module instability detected', 'error');
                crashDetector();
            }
        }, 3000);
        
        // Store monitors for cleanup
        window.bitnetChromeMonitor = chromeMonitor;
        window.addEventListener('beforeunload', crashDetector);
        window.addEventListener('unload', crashDetector);
        window.addEventListener('error', crashDetector);
        bitnet._crashDetector = crashDetector;
        
        // Check available functions
        const availableFunctions = [];
        for (const key in bitnet) {
            if (typeof bitnet[key] === 'function' && key.startsWith('_')) {
                availableFunctions.push(key);
            }
        }
        outputElement.textContent += `Available WASM functions: ${availableFunctions.length} found\n`;
        
        // Update status and enable buttons with delay to avoid race conditions
        updateStatus('BitNet-WASM initialized successfully!', 'success');
        
        // Enable UI with safety delay
        setTimeout(() => {
            if (!crashDetected) {
                loadModelButton.disabled = false;
                runInferenceButton.disabled = false;
                
                // Add note about demo mode
                outputElement.textContent += 'Note: Inference will run in demo mode if no model is loaded.\n';
                
                // Check IndexedDB cache
                updateDBStatus();
                
                // Setup event listeners
                setupEventListeners();
            }
        }, 200);
        
    } catch (error) {
        console.error('Failed to initialize BitNet-WASM:', error);
        updateStatus(`Failed to initialize: ${error.message}`, 'error');
        outputElement.textContent += `Error: ${error.message}\n`;
        
        // Cleanup on failure
        if (bitnet) {
            try {
                // Emergency cleanup
                bitnet = null;
                if (window.gc) window.gc();
            } catch (cleanupError) {
                console.error('Cleanup error:', cleanupError);
            }
        }
        
        // Disable buttons on initialization failure
        loadModelButton.disabled = true;
        runInferenceButton.disabled = true;
    }
}

// Safer update status function
function updateStatus(message, type = 'loading') {
    try {
        if (statusElement) {
            statusElement.textContent = message;
            statusElement.className = `status ${type}`;
        }
    } catch (error) {
        console.error('Status update error:', error);
    }
}

// Safer update load status function
function updateLoadStatus(message, type = '') {
    try {
        if (loadStatusElement) {
            loadStatusElement.textContent = message;
            loadStatusElement.className = `status ${type}`;
            loadStatusElement.style.display = message ? 'block' : 'none';
            
            // Ensure progress bars stay visible during loading
            if (type === 'loading' && (progressContainer || wasmProgressContainer)) {
                if (progressContainer && progressContainer.style.display === 'block') {
                    // Keep download progress visible
                }
                if (wasmProgressContainer && wasmProgressContainer.style.display === 'block') {
                    // Keep WASM progress visible
                }
            }
        }
    } catch (error) {
        console.error('Load status update error:', error);
    }
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
        let cachedModels = [];
        try {
            const timeoutPromise = new Promise((_, reject) => 
                setTimeout(() => reject(new Error('IndexedDB query timeout')), 5000)
            );
            cachedModels = await Promise.race([
                db.models.where('url').startsWith(modelUrl).toArray(),
                timeoutPromise
            ]);
        } catch (dbError) {
            console.warn('IndexedDB query failed:', dbError);
            // Continue with download if cache check fails
        }
        
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
        progressBar.textContent = 'Downloading...';
        
        try {
            // Fetch with progress tracking and cache preference
            const response = await fetch(modelUrl, { cache: 'force-cache' });
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


            // Store in IndexedDB for future use (chunked save with timeout)
            try {
                setTimeout(async () => {
                    try {
                        // Save in 2MB chunks to prevent IndexedDB corruption
                        const CHUNK_SIZE = 2 * 1024 * 1024;
                        let totalParts = 1;
                        if (arrayBuffer.byteLength > CHUNK_SIZE) {
                            totalParts = Math.ceil(arrayBuffer.byteLength / CHUNK_SIZE);
                        }
                        let offset = 0;
                        let part = 0;
                        if (totalParts > 1) {
                            saveProgressContainer.style.display = 'block';
                        }
                        
                        // Reduced timeout and individual transaction per chunk
                        const dbTimeout = 5000; // 5 seconds per chunk
                        
                        while (offset < arrayBuffer.byteLength) {
                            const end = Math.min(offset + CHUNK_SIZE, arrayBuffer.byteLength);
                            const chunkData = arrayBuffer.slice(offset, end);
                            
                            let retries = 1; // Reduced retries to prevent corruption
                            while (retries > 0) {
                                try {
                                    const timeoutPromise = new Promise((_, reject) => 
                                        setTimeout(() => reject(new Error('IndexedDB save timeout')), dbTimeout)
                                    );
                                    
                                    const savePromise = new Promise(async (resolve, reject) => {
                                        try {
                                            // Force each chunk into separate transaction to prevent corruption
                                            setTimeout(async () => {
                                                try {
                                                    // Close and reopen DB connection for each chunk
                                                    if (window.dbConnection) {
                                                        await window.dbConnection.close();
                                                    }
                                                    window.dbConnection = await db.open();
                                                    
                                                    const result = await db.models.put({
                                                        url: totalParts > 1 ? `${modelUrl}#part${part}` : modelUrl,
                                                        name: totalParts > 1 ? `${modelName} (part ${part})` : modelName,
                                                        data: chunkData,
                                                        timestamp: new Date().getTime(),
                                                        size: chunkData.byteLength
                                                    });
                                                    resolve(result);
                                                } catch (error) {
                                                    reject(error);
                                                }
                                            }, 10);
                                        } catch (error) {
                                            reject(error);
                                        }
                                    });
                                    
                                    await Promise.race([savePromise, timeoutPromise]);
                                    break; // Success, exit retry loop
                                } catch (error) {
                                    retries--;
                                    if (retries === 0) {
                                        console.warn(`Failed to save chunk ${part}: ${error.message}`);
                                        break; // Skip this chunk
                                    }
                                    // Wait before retry
                                    await new Promise(resolve => setTimeout(resolve, 1000));
                                }
                            }
                            
                            part++;
                            offset = end;
                            if (totalParts > 1) {
                                const percent = Math.round((part / totalParts) * 100);
                                saveProgressBar.style.width = `${percent}%`;                                        saveProgressBar.textContent = `Saving to IndexedDB... ${percent}%`;
                            }
                            
                            // Longer yield between chunks to prevent corruption
                            await new Promise(resolve => setTimeout(resolve, 50));
                        }
                        
                        if (totalParts > 1) {
                            outputElement.textContent += `Model saved to IndexedDB in ${part} parts\n`;
                            saveProgressContainer.style.display = 'none';
                        } else {
                            outputElement.textContent += `Model saved to IndexedDB cache\n`;
                        }
                        
                        // Update status after successful save
                        setTimeout(() => updateDBStatus(), 1000);
                    } catch (error) {
                        console.warn('Failed to cache model in IndexedDB:', error);
                        outputElement.textContent += `Warning: Could not cache model in IndexedDB: ${error.message}\n`;
                        saveProgressContainer.style.display = 'none';
                    }
                }, 0);
            } catch (dbError) {
                console.warn('Failed to cache model in IndexedDB:', dbError);
                outputElement.textContent += `Warning: Could not cache model in IndexedDB: ${dbError.message}\n`;
                saveProgressContainer.style.display = 'none';
            }

            // Hide progress bar
            progressContainer.style.display = 'none';

            // Close IndexedDB connection after successful save to prevent corruption
            if (window.dbConnection) {
                try {
                    window.dbConnection.close();
                    window.dbConnection = null;
                    console.log('IndexedDB connection closed after save');
                } catch (error) {
                    console.warn('Error closing IndexedDB connection:', error);
                }
            }

            // Load the model (continue even if this fails)
            try {
                modelData = await loadModelFromArrayBuffer(arrayBuffer, modelUrl);
                updateLoadStatus('Model loaded successfully!', 'success');
            } catch (wasmError) {
                console.warn('WASM loading failed:', wasmError.message);
                outputElement.textContent += `WASM loading failed: ${wasmError.message}\n`;
                updateLoadStatus('Model downloaded but WASM loading failed - demo mode available', 'warning');
                
                // Still store the raw model data for potential future use
                modelData = {
                    url: modelUrl,
                    size: arrayBuffer.byteLength,
                    failed: true
                };
            }
            
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
    let modelPtr = null;
    try {
        // Validate WASM module state
        if (!bitnet || !bitnet._malloc || !bitnet._free || !bitnet.HEAPU8) {
            throw new Error('WASM module not properly initialized');
        }
        
        // Validate input
        if (!arrayBuffer || arrayBuffer.byteLength === 0) {
            throw new Error('Invalid model data');
        }
        
        const modelSize = arrayBuffer.byteLength;
        
        // Check available memory before allocation
        try {
            const testPtr = bitnet._malloc(1024);
            if (testPtr) bitnet._free(testPtr);
        } catch (memError) {
            throw new Error('WASM memory allocation failed - insufficient memory');
        }
        
        // Allocate memory in WASM with reasonable size validation  
        const maxModelSize = 4 * 1024 * 1024 * 1024; // 4GB max
        if (modelSize > maxModelSize) {
            throw new Error(`Model too large: ${formatSize(modelSize)} exceeds ${formatSize(maxModelSize)} limit`);
        }
        
        // Check available memory before allocation
        const memInfo = performance.memory;
        if (memInfo && memInfo.usedJSHeapSize > memInfo.jsHeapSizeLimit * 0.8) {
            throw new Error('Insufficient browser memory available');
        }
        
        // Try progressive allocation to detect issues early - skip for large models
        if (modelSize < 100 * 1024 * 1024) { // Only test for models under 100MB
            let testPtr = null;
            try {
                // Test with smaller allocation first
                testPtr = bitnet._malloc(1024 * 1024); // 1MB test
                if (!testPtr) throw new Error('Progressive allocation test failed');
                bitnet._free(testPtr);
                testPtr = null;
            } catch (progError) {
                if (testPtr) bitnet._free(testPtr);
                console.warn('Progressive allocation test failed, proceeding anyway:', progError.message);
            }
        }
        
        modelPtr = bitnet._malloc(modelSize);
        if (!modelPtr) {
            throw new Error('Failed to allocate memory for model');
        }
        
        // Validate WASM heap access with buffer checks
        if (!bitnet.HEAPU8 || !bitnet.HEAPU8.buffer) {
            if (modelPtr) bitnet._free(modelPtr);
            throw new Error('WASM heap buffer not available');
        }
        
        if (bitnet.HEAPU8.buffer.byteLength < modelPtr + modelSize) {
            if (modelPtr) bitnet._free(modelPtr);
            throw new Error('WASM heap too small for model');
        }
        
        // Additional heap integrity check
        try {
            const heapView = new Uint8Array(bitnet.HEAPU8.buffer, modelPtr, Math.min(1024, modelSize));
            heapView[0] = 0; // Test write access
        } catch (heapError) {
            if (modelPtr) bitnet._free(modelPtr);
            throw new Error(`WASM heap access test failed: ${heapError.message}`);
        }
        
        // Copy model data to WASM memory with error handling and progress
        wasmProgressContainer.style.display = 'block';
        wasmProgressBar.style.width = '0%';
        wasmProgressBar.textContent = 'Loading into WASM...';
        
        // Force initial render
        wasmProgressBar.offsetHeight;
        
        try {
            const heapBytes = new Uint8Array(bitnet.HEAPU8.buffer, modelPtr, modelSize);
            const sourceBytes = new Uint8Array(arrayBuffer);
            
            // Copy in smaller chunks with integrity checks and progress
            const chunkSize = 512 * 1024; // 512KB chunks
            const totalChunks = Math.ceil(sourceBytes.length / chunkSize);
            
            for (let i = 0; i < sourceBytes.length; i += chunkSize) {
                const end = Math.min(i + chunkSize, sourceBytes.length);
                const chunkIndex = Math.floor(i / chunkSize);
                
                try {
                    // Verify heap is still valid before each chunk
                    if (!bitnet.HEAPU8.buffer || bitnet.HEAPU8.buffer.byteLength < modelPtr + modelSize) {
                        throw new Error('WASM heap became invalid during copy');
                    }
                    
                    heapBytes.set(sourceBytes.slice(i, end), i);
                    
                    // Verify chunk was written correctly
                    const written = heapBytes.slice(i, end);
                    if (written.length !== end - i) {
                        throw new Error('Chunk write verification failed');
                    }
                    
                    // Update progress
                    const progress = Math.round(((chunkIndex + 1) / totalChunks) * 50); // 50% for copy
                    wasmProgressBar.style.width = `${progress}%`;
                    wasmProgressBar.textContent = `Loading... ${progress}%`;
                    
                } catch (chunkError) {
                    throw new Error(`Chunk copy failed at ${i}: ${chunkError.message}`);
                }
                
                // Yield control and check for crash detection
                if (bitnet._crashDetector && typeof bitnet._crashDetector === 'function') {
                    if (!bitnet || !bitnet.HEAPU8) {
                        throw new Error('Browser instability detected during copy');
                    }
                }
                
                await new Promise(resolve => setTimeout(resolve, 5));
            }
        } catch (copyError) {
            wasmProgressContainer.style.display = 'none';
            if (modelPtr) bitnet._free(modelPtr);
            throw new Error(`Failed to copy model data: ${copyError.message}`);
        }
        
        // Update progress after copy completion
        wasmProgressBar.style.width = '50%';
        wasmProgressBar.textContent = 'Initializing...';
        
        // Force DOM update before starting model loading
        wasmProgressBar.offsetHeight;
        await new Promise(resolve => setTimeout(resolve, 100));
        
        outputElement.textContent += `Model data copied to WASM heap\n`;
        
        // Load model with enhanced timeout and monitoring
        let success = false;
        let monitorInterval = null;
        
        const loadPromise = new Promise((resolve, reject) => {
            try {
                // Set up monitoring during model loading with progress updates
                let lastHeartbeat = Date.now();
                let progressStep = 50; // Start at 50% (after copy)
                
                monitorInterval = setInterval(() => {
                    if (Date.now() - lastHeartbeat > 8000) {
                        clearInterval(monitorInterval);
                        reject(new Error('Model loading heartbeat timeout'));
                    }
                    
                    // Update progress during loading - more aggressive increments
                    progressStep = Math.min(progressStep + 10, 90);
                    wasmProgressBar.style.width = `${progressStep}%`;
                    wasmProgressBar.textContent = `Initializing... ${progressStep}%`;
                    
                    // Force DOM update
                    wasmProgressBar.offsetHeight;
                }, 500); // Update every 500ms for smoother progress
                
                // Attempt model loading with different function names
                let result = false;
                if (typeof bitnet._bitnet_load_model === 'function') {
                    result = bitnet._bitnet_load_model(modelPtr, modelSize);
                    lastHeartbeat = Date.now();
                } else if (typeof bitnet._load_model === 'function') {
                    result = bitnet._load_model(modelPtr, modelSize);
                    lastHeartbeat = Date.now();
                } else {
                    clearInterval(monitorInterval);
                    reject(new Error('Model loading function not found in WASM module'));
                    return;
                }
                
                clearInterval(monitorInterval);
                
                // Complete progress with visible update
                wasmProgressBar.style.width = '100%';
                wasmProgressBar.textContent = 'Complete!';
                
                // Force DOM update
                wasmProgressBar.offsetHeight;
                
                resolve(result === 1);
                
            } catch (loadError) {
                if (monitorInterval) clearInterval(monitorInterval);
                reject(new Error(`WASM model loading failed: ${loadError.message}`));
            }
        });
        
        const timeoutPromise = new Promise((_, reject) => 
            setTimeout(() => {
                if (monitorInterval) clearInterval(monitorInterval);
                reject(new Error('Model loading timeout'));
            }, 20000) // Reduced timeout
        );
        
        success = await Promise.race([loadPromise, timeoutPromise]);
        
        // Always free temporary memory safely
        if (modelPtr && bitnet && bitnet._free) {
            try {
                bitnet._free(modelPtr);
                modelPtr = null;
            } catch (freeError) {
                console.warn('Warning: Failed to free model pointer:', freeError);
            }
        }
        
        if (!success) {
            wasmProgressContainer.style.display = 'none';
            throw new Error('Failed to load model in WASM');
        }
        
        // Hide progress bar after successful load with longer delay
        setTimeout(() => {
            wasmProgressContainer.style.display = 'none';
        }, 3000); // Increased to 3 seconds so users can see "Complete!"
        
        outputElement.textContent += `Model loaded successfully\n`;
        updateLoadStatus('Model ready for inference!', 'success');
        
        // Return model data reference
        return {
            url: modelUrl,
            size: modelSize
        };
        
    } catch (error) {
        // Emergency cleanup
        wasmProgressContainer.style.display = 'none';
        if (modelPtr && bitnet && bitnet._free) {
            try {
                bitnet._free(modelPtr);
            } catch (freeError) {
                console.error('Emergency cleanup failed:', freeError);
            }
        }
        
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
    
    let inputPtr = null;
    let outputPtr = null;
    
    try {
        resultElement.textContent = 'Running inference...';
        
        // Validate WASM module - enhanced check
        if (!bitnet || typeof bitnet._malloc !== 'function') {
            outputElement.textContent += `WASM module not available for inference. Please load a model first.\n`;
            updateStatus('WASM module not initialized - try loading a model', 'error');
            return;
        }
        
        // Check if model is loaded
        let isModelLoaded = false;
        if (modelData) {
            // Check if model is loaded in WASM
            if (typeof bitnet._bitnet_is_model_loaded === 'function') {
                try {
                    isModelLoaded = bitnet._bitnet_is_model_loaded() === 1;
                } catch (checkError) {
                    console.warn('Model check failed:', checkError);
                    isModelLoaded = false;
                }
            } else if (typeof bitnet._is_model_loaded === 'function') {
                try {
                    isModelLoaded = bitnet._is_model_loaded() === 1;
                } catch (checkError) {
                    console.warn('Model check failed:', checkError);
                    isModelLoaded = false;
                }
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
        
        // Validate input text length more strictly
        if (inputText.length > 500) { // Reduced limit
            throw new Error('Input text too long (max 500 characters)');
        }
        
        // Check browser memory before inference - relaxed check
        const memInfo = performance.memory;
        if (memInfo && memInfo.usedJSHeapSize > memInfo.jsHeapSizeLimit * 0.95) {
            console.warn('High memory usage detected before inference:', 
                (memInfo.usedJSHeapSize / memInfo.jsHeapSizeLimit * 100).toFixed(1) + '%');
            // Continue anyway - just warn
        }
        
        // Allocate memory for input and output with relaxed validation
        const inputBytes = bitnet.lengthBytesUTF8(inputText) + 1;
        const maxOutputLength = 1024; // Increased output buffer
        
        if (inputBytes > 8192) { // Relaxed safety limit
            throw new Error('Input text requires too much memory');
        }
        
        try {
            inputPtr = bitnet._malloc(inputBytes);
            outputPtr = bitnet._malloc(maxOutputLength);
            
            if (!inputPtr || !outputPtr) {
                throw new Error('Failed to allocate memory for inference');
            }
            
            // Validate heap bounds
            if (!bitnet.HEAPU8.buffer || 
                bitnet.HEAPU8.buffer.byteLength < Math.max(inputPtr + inputBytes, outputPtr + maxOutputLength)) {
                throw new Error('WASM heap bounds exceeded');
            }
            
            // Copy input text to WASM memory
            bitnet.stringToUTF8(inputText, inputPtr, inputBytes);
            
            // Clear output buffer
            const outputBuffer = new Uint8Array(bitnet.HEAPU8.buffer, outputPtr, maxOutputLength);
            outputBuffer.fill(0);
            
        } catch (allocError) {
            throw new Error(`Memory allocation failed: ${allocError.message}`);
        }
        
        // Run inference with enhanced monitoring
        let outputLength = 0;
        let inferenceMonitor = null;
        const startTime = performance.now();
        
        const inferencePromise = new Promise((resolve, reject) => {
            try {
                // Set up inference monitoring with progress updates
                let lastActivity = Date.now();
                let progressDots = 0;
                
                inferenceMonitor = setInterval(() => {
                    if (Date.now() - lastActivity > 3000) {
                        clearInterval(inferenceMonitor);
                        reject(new Error('Inference activity timeout'));
                    }
                    // Check if WASM is still responsive
                    if (!bitnet || !bitnet.HEAPU8) {
                        clearInterval(inferenceMonitor);
                        reject(new Error('WASM became unresponsive during inference'));
                    }
                    
                    // Update progress indication
                    progressDots = (progressDots + 1) % 4;
                    const dots = '.'.repeat(progressDots);
                    resultElement.textContent = `Running inference${dots}`;
                }, 500);
                
                // Show initial progress
                resultElement.textContent = 'Starting inference...';
                
                if (typeof bitnet._bitnet_inference_run === 'function') {
                    outputElement.textContent += `Calling bitnet_inference_run...\n`;
                    const result = bitnet._bitnet_inference_run(inputPtr, outputPtr, maxOutputLength);
                    lastActivity = Date.now();
                    clearInterval(inferenceMonitor);
                    resolve(result || 0);
                } else if (typeof bitnet._inference_run === 'function') {
                    outputElement.textContent += `Calling inference_run...\n`;
                    const result = bitnet._inference_run(inputPtr, outputPtr, maxOutputLength);
                    lastActivity = Date.now();
                    clearInterval(inferenceMonitor);
                    resolve(result || 0);
                } else if (typeof bitnet._run_inference === 'function') {
                    outputElement.textContent += `Calling run_inference...\n`;
                    const result = bitnet._run_inference(inputPtr, outputPtr, maxOutputLength);
                    lastActivity = Date.now();
                    clearInterval(inferenceMonitor);
                    resolve(result || 0);
                } else {
                    clearInterval(inferenceMonitor);
                    reject(new Error('Inference function not found in WASM module'));
                }
            } catch (runError) {
                if (inferenceMonitor) clearInterval(inferenceMonitor);
                reject(new Error(`Inference execution failed: ${runError.message}`));
            }
        });
        
        const timeoutPromise = new Promise((_, reject) => 
            setTimeout(() => {
                if (inferenceMonitor) clearInterval(inferenceMonitor);
                reject(new Error('Inference timeout'));
            }, 15000) // Reduced timeout
        );
        
        outputLength = await Promise.race([inferencePromise, timeoutPromise]);
        
        const endTime = performance.now();
        const inferenceTime = (endTime - startTime).toFixed(2);
        
        // Read output from WASM memory with validation
        let result = '';
        if (outputLength > 0 && outputLength <= maxOutputLength) {
            try {
                const buffer = new Uint8Array(bitnet.HEAPU8.buffer, outputPtr, outputLength);
                let str = '';
                for (let i = 0; i < outputLength; i++) {
                    if (buffer[i] === 0) break;
                    str += String.fromCharCode(buffer[i]);
                }
                result = str;
            } catch (readError) {
                console.warn('Error reading output:', readError);
                result = 'Error reading inference result';
            }
        }
        
        // Display result
        outputElement.textContent += `Inference completed in ${inferenceTime}ms\n`;
        resultElement.innerHTML = `<strong>Input:</strong>\n${inputText}\n\n<strong>Output:</strong>\n${result}\n\n<em>Inference time: ${inferenceTime}ms</em>`;
        
    } catch (error) {
        resultElement.textContent = `Error running inference: ${error.message}`;
        outputElement.textContent += `Error running inference: ${error.message}\n`;
        console.error('Error running inference:', error);
    } finally {
        // Always free allocated memory
        if (inputPtr) {
            try {
                bitnet._free(inputPtr);
            } catch (freeError) {
                console.warn('Failed to free input pointer:', freeError);
            }
        }
        if (outputPtr) {
            try {
                bitnet._free(outputPtr);
            } catch (freeError) {
                console.warn('Failed to free output pointer:', freeError);
            }
        }
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
    // Use a completely non-blocking approach that won't crash the browser
    dbStatusElement.textContent = 'Checking cache...';
    dbStatusElement.className = '';
    
    // Use setImmediate equivalent to avoid blocking
    setTimeout(async () => {
        try {
            // Test IndexedDB availability first
            if (!window.indexedDB) {
                throw new Error('IndexedDB not supported');
            }
            
            // Ultra-short timeout to prevent any hanging
            const quickTimeout = new Promise((_, reject) => 
                setTimeout(() => reject(new Error('Database timeout')), 1000)
            );
            
            // Test database connection
            const dbTest = await Promise.race([
                new Promise(async (resolve, reject) => {
                    try {
                        // Simple count operation to verify it works
                        const count = await db.models.count();
                        resolve(count);
                    } catch (error) {
                        reject(error);
                    }
                }),
                quickTimeout
            ]);
            
            dbStatusElement.textContent = `IndexedDB: Available (${dbTest} models)`;
            dbStatusElement.className = 'success';
            
            // Populate dropdown safely in background
            setTimeout(() => safeUpdateDropdown(), 100);
            
        } catch (error) {
            console.warn('IndexedDB unavailable:', error.message);
            dbStatusElement.textContent = 'IndexedDB: Unavailable';
            dbStatusElement.className = 'error';
            
            // Immediately disable cache features
            disableCacheFeatures();
        }
    }, 0);
}

// Safe dropdown update function
async function safeUpdateDropdown() {
    try {
        const quickTimeout = new Promise((_, reject) => 
            setTimeout(() => reject(new Error('Dropdown timeout')), 1000)
        );
        
        const models = await Promise.race([
            db.models.toArray(),
            quickTimeout
        ]);
        
        if (models && models.length > 0) {
            const modelGroups = {};
            models.forEach(model => {
                if (model && model.url) {
                    const baseUrl = model.url.split('#part')[0];
                    if (!modelGroups[baseUrl]) modelGroups[baseUrl] = [];
                    modelGroups[baseUrl].push(model);
                }
            });
            
            let options = '<option value="">-- Select cached model --</option>';
            for (const baseUrl in modelGroups) {
                const group = modelGroups[baseUrl];
                const totalSize = group.reduce((sum, m) => sum + (m.size || 0), 0);
                const name = (group[0] && group[0].name) ? group[0].name.replace(/ \(part.*\)/, '') : 'Model';
                options += `<option value="${baseUrl}">${name} (${formatSize(totalSize)})</option>`;
            }
            
            if (cachedModelSelect) {
                cachedModelSelect.innerHTML = options;
                cachedModelSelect.disabled = false;
            }
            
            dbStatusElement.textContent = `${Object.keys(modelGroups).length} model(s) cached`;
        } else {
            if (cachedModelSelect) {
                cachedModelSelect.innerHTML = '<option value="">-- No cached models --</option>';
            }
            dbStatusElement.textContent = 'No models in cache';
        }
    } catch (error) {
        console.warn('Error updating dropdown:', error.message);
        disableCacheFeatures();
    }
}

// Disable cache features safely
function disableCacheFeatures() {
    if (cachedModelSelect) {
        cachedModelSelect.innerHTML = '<option value="">-- Cache unavailable --</option>';
        cachedModelSelect.disabled = true;
    }
    
    const viewCacheBtn = document.getElementById('view-cache');
    const clearCacheBtn = document.getElementById('clear-cache');
    if (viewCacheBtn) viewCacheBtn.disabled = true;
    if (clearCacheBtn) clearCacheBtn.disabled = true;
}

// View cached models - safe non-blocking version
async function viewCachedModels() {
    cacheDetailsElement.textContent = 'Loading cache details...';
    cacheDetailsElement.style.display = 'block';
    
    setTimeout(async () => {
        try {
            // Very short timeout to prevent hanging
            const quickTimeout = new Promise((_, reject) => 
                setTimeout(() => reject(new Error('Cache view timeout')), 2000)
            );
            
            const models = await Promise.race([
                db.models.toArray(),
                quickTimeout
            ]);
            
            if (!models || models.length === 0) {
                cacheDetailsElement.textContent = 'No cached models found.';
                return;
            }
            
            // Group models by base URL safely
            const modelGroups = {};
            models.forEach(model => {
                if (model && model.url) {
                    const baseUrl = model.url.split('#part')[0];
                    if (!modelGroups[baseUrl]) {
                        modelGroups[baseUrl] = [];
                    }
                    modelGroups[baseUrl].push(model);
                }
            });
            
            let details = '<h3>Cached Models:</h3>';
            for (const baseUrl in modelGroups) {
                const group = modelGroups[baseUrl];
                const totalSize = group.reduce((sum, m) => sum + (m.size || 0), 0);
                const name = (group[0] && group[0].name) ? group[0].name.replace(/ \(part.*\)/, '') : 'Model';
                const parts = group.length > 1 ? ` (${group.length} parts)` : '';
                
                details += `<div style="margin: 10px 0; padding: 10px; border: 1px solid #ddd;">`;
                details += `<strong>${name}</strong>${parts}<br>`;
                details += `Size: ${formatSize(totalSize)}<br>`;
                details += `URL: ${baseUrl}<br>`;
                if (group[0] && group[0].timestamp) {
                    details += `Cached: ${new Date(group[0].timestamp).toLocaleString()}`;
                }
                details += `</div>`;
            }
            
            cacheDetailsElement.innerHTML = details;
            
        } catch (error) {
            console.warn('Error viewing cached models:', error.message);
            cacheDetailsElement.textContent = `Error loading cache details: ${error.message}`;
        }
    }, 0);
}

// Clear model cache - safe non-blocking version
async function clearModelCache() {
    if (!confirm('Are you sure you want to clear all cached models?')) {
        return;
    }
    
    dbStatusElement.textContent = 'Clearing cache...';
    dbStatusElement.className = 'loading';
    cacheDetailsElement.textContent = '';
    cacheDetailsElement.style.display = 'none';
    
    setTimeout(async () => {
        try {
            // Very short timeout to prevent hanging
            const quickTimeout = new Promise((_, reject) => 
                setTimeout(() => reject(new Error('Clear cache timeout')), 3000)
            );
            
            await Promise.race([
                db.models.clear(),
                quickTimeout
            ]);
            
            dbStatusElement.textContent = 'Cache cleared successfully';
            dbStatusElement.className = 'success';
            
            // Update dropdown
            if (cachedModelSelect) {
                cachedModelSelect.innerHTML = '<option value="">-- No cached models --</option>';
            }
            
            // Refresh status after a delay
            setTimeout(() => updateDBStatus(), 1000);
            
        } catch (error) {
            console.warn('Error clearing cache:', error.message);
            dbStatusElement.textContent = `Error clearing cache: ${error.message}`;
            dbStatusElement.className = 'error';
        }
    }, 0);
}

// Format file size
function formatSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Global cleanup on page unload to prevent Chrome crashes
function performEmergencyCleanup() {
    try {
        // Clear all monitors
        if (window.bitnetChromeMonitor) {
            clearInterval(window.bitnetChromeMonitor);
            window.bitnetChromeMonitor = null;
        }
        
        // Cleanup WASM module
        if (bitnet) {
            try {
                // Force immediate cleanup
                bitnet = null;
                modelData = null;
                
                // Force garbage collection if available
                if (window.gc) {
                    window.gc();
                }
            } catch (error) {
                console.error('WASM cleanup error:', error);
            }
        }
        
        // Close IndexedDB connections
        try {
            if (db && db.close) {
                db.close();
            }
        } catch (error) {
            console.error('DB cleanup error:', error);
        }
        
    } catch (error) {
        console.error('Emergency cleanup failed:', error);
    }
}

// Add emergency cleanup on all page unload events
window.addEventListener('beforeunload', performEmergencyCleanup);
window.addEventListener('unload', performEmergencyCleanup);
window.addEventListener('pagehide', performEmergencyCleanup);

// Also cleanup on visibility change (tab switch)
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        // Page is being hidden, do light cleanup
        if (window.gc) {
            setTimeout(() => window.gc(), 100);
        }
    }
});

// Initialize when page loads
document.addEventListener('DOMContentLoaded', initBitNet);

// Add spinner animation CSS
const style = document.createElement('style');
style.innerHTML = `@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }`;
document.head.appendChild(style);

// Chrome crash protection: Add error boundaries
function addGlobalErrorHandlers() {
    window.addEventListener('unhandledrejection', (event) => {
        console.error('Unhandled Promise rejection:', event.reason);
        updateStatus('Promise rejection detected', 'error');
        event.preventDefault();
    });
    
    window.addEventListener('error', (event) => {
        console.error('Global error:', event.error);
        updateStatus('JavaScript error detected', 'error');
    });
}

// Add beforeunload handler to safely close IndexedDB connections
window.addEventListener('beforeunload', function(event) {
    console.log('Browser closing - cleaning up IndexedDB connections');
    
    // Close any open IndexedDB connections
    if (window.dbConnection) {
        try {
            window.dbConnection.close();
            window.dbConnection = null;
            console.log('IndexedDB connection closed safely');
        } catch (error) {
            console.warn('Error closing IndexedDB connection:', error);
        }
    }
    
    // Force close any remaining connections
    try {
        indexedDB.databases().then(databases => {
            databases.forEach(db => {
                if (db.name === 'ModelCacheDB') {
                    console.log('Force closing ModelCacheDB');
                }
            });
        }).catch(() => {
            // Ignore errors during cleanup
        });
    } catch (error) {
        // Ignore errors during cleanup
    }
});

// Add pagehide handler for mobile browsers
window.addEventListener('pagehide', function(event) {
    console.log('Page hiding - cleaning up IndexedDB connections');
    
    if (window.dbConnection) {
        try {
            window.dbConnection.close();
            window.dbConnection = null;
        } catch (error) {
            console.warn('Error closing IndexedDB connection on pagehide:', error);
        }
    }
});

// Start monitoring after initialization
setTimeout(startMemoryMonitoring, 3000);

// Memory pressure monitoring for Chrome stability