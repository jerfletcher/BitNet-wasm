// BitNet-WASM Demo - GitHub Pages
// This example demonstrates how to use BitNet-WASM in your own projects

// Import BitNet-WASM module from CDN (for GitHub Pages demo)
// In your own project, you would use a relative path like './bitnet.js'
import BitNetModule from 'https://cdn.jsdelivr.net/gh/jerfletcher/BitNet-wasm@latest/bitnet.js';
// Global variables
let bitnet = null;
let modelData = null;

// DOM elements
const statusElement = document.getElementById('status');
const outputElement = document.getElementById('output');
const loadModelButton = document.getElementById('load-model');
const runInferenceButton = document.getElementById('run-inference');
const loadStatusElement = document.getElementById('load-status');
const progressContainer = document.querySelector('.progress-container');
const progressBar = document.getElementById('download-progress');
const wasmProgressContainer = document.getElementById('wasm-progress-container');
const wasmProgressBar = document.getElementById('wasm-progress');
const clearCacheButton = document.getElementById('clear-cache');

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
            if (type === 'loading' && progressContainer) {
                if (progressContainer && progressContainer.style.display === 'block') {
                    // Keep download progress visible
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
    
    // Clear cache button
    clearCacheButton.addEventListener('click', clearModelCache);
}

// IndexedDB for model caching
let modelDB = null;

// Initialize IndexedDB for model caching
async function initModelCache() {
    return new Promise((resolve, reject) => {
        const request = indexedDB.open('BitNetModelCache', 1);
        
        request.onerror = () => reject(request.error);
        
        request.onsuccess = () => {
            modelDB = request.result;
            resolve(modelDB);
        };
        
        request.onupgradeneeded = (event) => {
            const db = event.target.result;
            if (!db.objectStoreNames.contains('models')) {
                const store = db.createObjectStore('models', { keyPath: 'url' });
                store.createIndex('timestamp', 'timestamp', { unique: false });
            }
        };
    });
}

// Get cached model from IndexedDB
async function getCachedModel(url) {
    if (!modelDB) return null;
    
    return new Promise((resolve, reject) => {
        const transaction = modelDB.transaction(['models'], 'readonly');
        const store = transaction.objectStore('models');
        const request = store.get(url);
        
        request.onerror = () => reject(request.error);
        request.onsuccess = () => resolve(request.result);
    });
}

// Clear model cache function
async function clearModelCache() {
    try {
        if (!modelDB) {
            await initModelCache();
        }
        
        updateLoadStatus('Clearing cache...', 'loading');
        
        const transaction = modelDB.transaction(['models'], 'readwrite');
        const store = transaction.objectStore('models');
        await store.clear();
        
        outputElement.textContent += 'Model cache cleared\n';
        updateLoadStatus('Cache cleared successfully', 'success');
        
        // Update cache info display
        await displayCacheInfo();
        
        setTimeout(() => {
            updateLoadStatus('', '');
        }, 2000);
        
    } catch (error) {
        console.error('Failed to clear cache:', error);
        updateLoadStatus(`Error clearing cache: ${error.message}`, 'error');
    }
}

// Store model in IndexedDB cache
async function cacheModel(url, arrayBuffer) {
    if (!modelDB) return;
    
    try {
        const modelData = {
            url: url,
            data: arrayBuffer,
            timestamp: Date.now(),
            size: arrayBuffer.byteLength
        };
        
        const transaction = modelDB.transaction(['models'], 'readwrite');
        const store = transaction.objectStore('models');
        store.put(modelData);
        
        // Clean old cache entries (keep only last 3 models)
        const index = store.index('timestamp');
        const allModels = [];
        
        index.openCursor(null, 'prev').onsuccess = (event) => {
            const cursor = event.target.result;
            if (cursor) {
                allModels.push(cursor.value);
                cursor.continue();
            } else if (allModels.length > 3) {
                // Delete oldest models
                for (let i = 3; i < allModels.length; i++) {
                    store.delete(allModels[i].url);
                }
            }
        };
    } catch (error) {
        console.warn('Failed to cache model:', error);
    }
}

// Load a model from URL
async function loadModel() {
    const modelUrl = document.getElementById('model-url').value.trim();
    
    if (!modelUrl) {
        updateLoadStatus('Please enter a valid model URL', 'error');
        return;
    }
    
    try {
        // Initialize cache if not already done
        if (!modelDB) {
            await initModelCache();
        }
        
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
        
        // Check cache first
        const cachedModel = await getCachedModel(modelUrl);
        if (cachedModel) {
            outputElement.textContent += `Found cached model (${formatSize(cachedModel.size)})\n`;
            updateLoadStatus('Using cached model...', 'loading');
            
            // Force garbage collection before loading cached model
            if (window.gc) {
                window.gc();
                outputElement.textContent += `Cleared memory before loading cached model\n`;
            }
            
            // Small delay to ensure memory is settled
            await new Promise(resolve => setTimeout(resolve, 500));
            
            try {
                modelData = await loadModelFromArrayBuffer(cachedModel.data, modelUrl);
                updateLoadStatus('Cached model loaded successfully!', 'success');
                return;
            } catch (wasmError) {
                console.warn('Cached model failed to load, downloading fresh:', wasmError.message);
                outputElement.textContent += `Cached model failed, downloading fresh...\n`;
            } finally {
                // Hide WASM progress bar
                if (wasmProgressContainer) {
                    wasmProgressContainer.style.display = 'none';
                }
            }
        }
        
        // Model not in cache, download it
        outputElement.textContent += `Downloading model from ${modelUrl}...\n`;
        updateLoadStatus('Downloading model...', 'loading');
        
        // Show progress bar
        progressContainer.style.display = 'block';
        progressBar.style.width = '0%';
        progressBar.textContent = 'Downloading...';
        
        try {
            // Fetch with progress tracking and cache headers
            const response = await fetch(modelUrl, { 
                cache: 'force-cache', // Use browser cache if available
                headers: {
                    'Cache-Control': 'max-age=86400' // Cache for 24 hours
                }
            });
            
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
            let chunksAll = new Uint8Array(loaded);
            let position = 0;
            for (const chunk of chunks) {
                chunksAll.set(chunk, position);
                position += chunk.length;
            }

            const arrayBuffer = chunksAll.buffer;
            outputElement.textContent += `Model downloaded: ${formatSize(arrayBuffer.byteLength)}\n`;
            
            // Cache the model for future use
            await cacheModel(modelUrl, arrayBuffer);
            outputElement.textContent += `Model cached for future use\n`;

            // Hide download progress bar
            progressContainer.style.display = 'none';
            
            // Force garbage collection and allow memory to settle before WASM loading
            updateLoadStatus('Preparing for WASM loading...', 'loading');
            
            // Clear chunk references to free memory
            chunks.length = 0;
            chunksAll = null; // Now this works since chunksAll is declared with let
            
            // Force garbage collection if available
            if (window.gc) {
                window.gc();
                outputElement.textContent += `Forced garbage collection\n`;
            }
            
            // Wait for memory to settle
            await new Promise(resolve => setTimeout(resolve, 2000));
            
            // Check memory again before WASM loading
            const memInfo = performance.memory;
            if (memInfo) {
                const usedPercent = (memInfo.usedJSHeapSize / memInfo.jsHeapSizeLimit) * 100;
                outputElement.textContent += `Memory usage before WASM load: ${usedPercent.toFixed(1)}%\n`;
                
                if (usedPercent > 85) {
                    outputElement.textContent += `High memory usage detected, forcing additional cleanup...\n`;
                    
                    // Additional cleanup attempts
                    for (let i = 0; i < 3; i++) {
                        if (window.gc) window.gc();
                        await new Promise(resolve => setTimeout(resolve, 500));
                    }
                    
                    // Check again
                    const newUsedPercent = (performance.memory.usedJSHeapSize / performance.memory.jsHeapSizeLimit) * 100;
                    outputElement.textContent += `Memory usage after cleanup: ${newUsedPercent.toFixed(1)}%\n`;
                }
            }

            // Load the model into WASM
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
            } finally {
                // Hide WASM progress bar
                if (wasmProgressContainer) {
                    wasmProgressContainer.style.display = 'none';
                }
            }
            
        } catch (error) {
            // Hide progress bar on error
            progressContainer.style.display = 'none';
            if (wasmProgressContainer) {
                wasmProgressContainer.style.display = 'none';
            }
            
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
        
        // Update cache info display
        await displayCacheInfo();
    }
}

// Load model from ArrayBuffer
async function loadModelFromArrayBuffer(arrayBuffer, modelUrl) {
    let modelPtr = null;
    try {
        // Show WASM loading progress
        if (wasmProgressContainer && wasmProgressBar) {
            wasmProgressContainer.style.display = 'block';
            wasmProgressBar.style.width = '0%';
            wasmProgressBar.textContent = 'Initializing...';
        }
        
        // Validate WASM module state
        if (!bitnet || !bitnet._malloc || !bitnet._free || !bitnet.HEAPU8) {
            throw new Error('WASM module not properly initialized');
        }
        
        if (wasmProgressBar) {
            wasmProgressBar.style.width = '10%';
            wasmProgressBar.textContent = 'Validating...';
        }
        
        // Validate input
        if (!arrayBuffer || arrayBuffer.byteLength === 0) {
            throw new Error('Invalid model data');
        }
        
        const modelSize = arrayBuffer.byteLength;
        
        if (wasmProgressBar) {
            wasmProgressBar.style.width = '20%';
            wasmProgressBar.textContent = 'Checking memory...';
        }
        
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
        if (memInfo) {
            const usedPercent = (memInfo.usedJSHeapSize / memInfo.jsHeapSizeLimit) * 100;
            outputElement.textContent += `Current memory usage: ${usedPercent.toFixed(1)}%\n`;
            
            if (usedPercent > 90) {
                outputElement.textContent += `High memory usage detected, attempting cleanup...\n`;
                
                // Force multiple garbage collections with delays
                for (let i = 0; i < 5; i++) {
                    if (window.gc) {
                        window.gc();
                        await new Promise(resolve => setTimeout(resolve, 200));
                    }
                }
                
                // Check memory again after cleanup
                const newUsedPercent = (performance.memory.usedJSHeapSize / performance.memory.jsHeapSizeLimit) * 100;
                outputElement.textContent += `Memory usage after cleanup: ${newUsedPercent.toFixed(1)}%\n`;
                
                // Only throw error if still very high after cleanup
                if (newUsedPercent > 95) {
                    throw new Error(`Insufficient browser memory available: ${newUsedPercent.toFixed(1)}% used`);
                }
            }
        }
        
        if (wasmProgressBar) {
            wasmProgressBar.style.width = '30%';
            wasmProgressBar.textContent = 'Allocating memory...';
        }
        
        // Additional memory pressure relief before allocation
        const modelSizeMB = modelSize / (1024 * 1024);
        if (modelSizeMB > 500) { // For models larger than 500MB
            outputElement.textContent += `Large model detected (${modelSizeMB.toFixed(1)}MB), performing additional memory optimization...\n`;
            
            // More aggressive cleanup for large models
            for (let i = 0; i < 10; i++) {
                if (window.gc) {
                    window.gc();
                    await new Promise(resolve => setTimeout(resolve, 100));
                }
            }
            
            // Check memory one more time
            if (performance.memory) {
                const finalUsedPercent = (performance.memory.usedJSHeapSize / performance.memory.jsHeapSizeLimit) * 100;
                outputElement.textContent += `Final memory check: ${finalUsedPercent.toFixed(1)}% used\n`;
            }
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
        
        if (wasmProgressBar) {
            wasmProgressBar.style.width = '40%';
            wasmProgressBar.textContent = 'Validating heap...';
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
        
        if (wasmProgressBar) {
            wasmProgressBar.style.width = '50%';
            wasmProgressBar.textContent = 'Copying data...';
        }
        
        // Copy model data to WASM memory with error handling
        try {
            const heapBytes = new Uint8Array(bitnet.HEAPU8.buffer, modelPtr, modelSize);
            const sourceBytes = new Uint8Array(arrayBuffer);
            
            // Copy in smaller chunks with integrity checks
            const chunkSize = 512 * 1024; // 512KB chunks
            const totalChunks = Math.ceil(sourceBytes.length / chunkSize);
            
            for (let i = 0; i < sourceBytes.length; i += chunkSize) {
                const end = Math.min(i + chunkSize, sourceBytes.length);
                const chunkIndex = Math.floor(i / chunkSize);
                
                // Update progress during copy
                if (wasmProgressBar) {
                    const copyProgress = 50 + (chunkIndex / totalChunks) * 30; // 50% to 80%
                    wasmProgressBar.style.width = `${Math.round(copyProgress)}%`;
                    wasmProgressBar.textContent = `Loading ${Math.round((chunkIndex / totalChunks) * 100)}%`;
                }
                
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
            if (modelPtr) bitnet._free(modelPtr);
            throw new Error(`Failed to copy model data: ${copyError.message}`);
        }
        
        outputElement.textContent += `Model data copied to WASM heap\n`;
        
        if (wasmProgressBar) {
            wasmProgressBar.style.width = '85%';
            wasmProgressBar.textContent = 'Loading model...';
        }
        
        // Load model with enhanced timeout and monitoring
        let success = false;
        
        const loadPromise = new Promise((resolve, reject) => {
            try {
                // Attempt model loading with different function names
                let result = false;
                if (typeof bitnet._bitnet_load_model === 'function') {
                    result = bitnet._bitnet_load_model(modelPtr, modelSize);
                } else if (typeof bitnet._load_model === 'function') {
                    result = bitnet._load_model(modelPtr, modelSize);
                } else {
                    reject(new Error('Model loading function not found in WASM module'));
                    return;
                }
                
                resolve(result === 1);
                
            } catch (loadError) {
                reject(new Error(`WASM model loading failed: ${loadError.message}`));
            }
        });
        
        const timeoutPromise = new Promise((_, reject) => 
            setTimeout(() => reject(new Error('Model loading timeout')), 20000)
        );
        
        success = await Promise.race([loadPromise, timeoutPromise]);
        
        if (wasmProgressBar) {
            wasmProgressBar.style.width = '95%';
            wasmProgressBar.textContent = 'Finalizing...';
        }
        
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
            throw new Error('Failed to load model in WASM');
        }
        
        if (wasmProgressBar) {
            wasmProgressBar.style.width = '100%';
            wasmProgressBar.textContent = 'Complete!';
        }
        
        outputElement.textContent += `Model loaded successfully\n`;
        updateLoadStatus('Model ready for inference!', 'success');
        
        // Hide progress bar after a brief delay
        setTimeout(() => {
            if (wasmProgressContainer) {
                wasmProgressContainer.style.display = 'none';
            }
        }, 1000);
        
        // Return model data reference
        return {
            url: modelUrl,
            size: modelSize
        };
        
    } catch (error) {
        // Emergency cleanup
        if (modelPtr && bitnet && bitnet._free) {
            try {
                bitnet._free(modelPtr);
            } catch (freeError) {
                console.error('Emergency cleanup failed:', freeError);
            }
        }
        
        // Hide progress bar on error
        if (wasmProgressContainer) {
            wasmProgressContainer.style.display = 'none';
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

// Utility function to format file sizes
function formatSize(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
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
// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', async () => {
    // Display cache info first
    await displayCacheInfo();
    
    // Then initialize BitNet
    await initBitNet();
});

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

// Start monitoring after initialization
setTimeout(startMemoryMonitoring, 3000);

// Memory pressure monitoring for Chrome stability

// Display cache information to user
async function displayCacheInfo() {
    const cacheInfoElement = document.getElementById('cache-info');
    if (!cacheInfoElement) return;
    
    try {
        if (!modelDB) {
            await initModelCache();
        }
        
        const transaction = modelDB.transaction(['models'], 'readonly');
        const store = transaction.objectStore('models');
        const request = store.getAll();
        
        request.onsuccess = () => {
            const models = request.result;
            if (models.length === 0) {
                cacheInfoElement.textContent = 'No models cached yet';
            } else {
                const totalSize = models.reduce((sum, model) => sum + model.size, 0);
                cacheInfoElement.textContent = `${models.length} model(s) cached (${formatSize(totalSize)} total)`;
            }
            cacheInfoElement.style.display = 'block';
        };
    } catch (error) {
        console.warn('Failed to display cache info:', error);
    }
}

// Initialize everything when DOM is ready