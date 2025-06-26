// Minimal test to isolate alignment fault
const Module = require('./bitnet.js');

async function testMinimal() {
    try {
        console.log('🚀 Minimal BitNet alignment test...');
        
        const wasmModule = await Module();
        console.log('✅ Module loaded');
        
        // Initialize
        const initResult = wasmModule.ccall('bitnet_init', 'number', [], []);
        console.log('Init result:', initResult);
        
        if (initResult !== 1) {
            console.log('⚠️ Init returned:', initResult, 'but continuing...');
        }
        console.log('✅ BitNet initialized');
        
        // Try to load a minimal file to isolate the alignment issue
        const fs = require('fs');
        const modelBuffer = fs.readFileSync('/Users/a206413899/development/BitNet-wasm/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf');
        
        console.log(`📂 Model size: ${modelBuffer.length} bytes`);
        
        // Create WASM memory buffer
        const modelSize = modelBuffer.length;
        const modelPtr = wasmModule._malloc(modelSize);
        
        if (!modelPtr) {
            throw new Error('Failed to allocate WASM memory for model');
        }
        
        console.log(`💾 Allocated ${modelSize} bytes at WASM address 0x${modelPtr.toString(16)}`);
        
        // Check alignment
        console.log(`🔍 Memory alignment check: ptr=0x${modelPtr.toString(16)}, aligned_4=${modelPtr % 4 === 0}, aligned_8=${modelPtr % 8 === 0}, aligned_16=${modelPtr % 16 === 0}`);
        
        // Copy data in smaller chunks to isolate where alignment fault occurs
        const chunkSize = 1024 * 1024; // 1MB chunks
        let written = 0;
        
        try {
            for (let offset = 0; offset < modelSize; offset += chunkSize) {
                const end = Math.min(offset + chunkSize, modelSize);
                const chunk = modelBuffer.subarray(offset, end);
                
                console.log(`Writing chunk: offset=${offset}, size=${chunk.length}, ptr=0x${(modelPtr + offset).toString(16)}`);
                
                // Check chunk alignment
                const chunkPtr = modelPtr + offset;
                console.log(`Chunk alignment: ptr=0x${chunkPtr.toString(16)}, aligned_4=${chunkPtr % 4 === 0}, aligned_8=${chunkPtr % 8 === 0}`);
                
                wasmModule.HEAPU8.set(chunk, modelPtr + offset);
                written += chunk.length;
                
                console.log(`✅ Chunk written successfully, total: ${written} bytes`);
                
                // Try a small test read to see if we can detect alignment issues
                if (offset === 0) {
                    console.log('🔍 Testing first bytes read...');
                    const testBytes = wasmModule.HEAPU8.subarray(modelPtr, modelPtr + 16);
                    console.log('First 16 bytes:', Array.from(testBytes).map(b => b.toString(16).padStart(2, '0')).join(' '));
                }
            }
            
            console.log('💾 Model copied to WASM memory successfully');
            
            // Now try the actual load (this is likely where alignment fault occurs)
            console.log('🔄 Attempting model load...');
            
            const loadResult = wasmModule.ccall('bitnet_load_model_from_memory', 'number', 
                ['number', 'number'], [modelPtr, modelSize]);
                
            console.log('Load result:', loadResult);
            
        } catch (error) {
            console.error('💥 Error during data copy or load:', error.message);
            throw error;
        } finally {
            // Clean up
            wasmModule._free(modelPtr);
        }
        
    } catch (error) {
        console.error('💥 Test failed:', error.message);
        throw error;
    }
}

testMinimal().catch(console.error);
