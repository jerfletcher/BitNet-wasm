const fs = require('fs');

async function quickTest() {
    console.log('🚀 Quick BitNet test starting...');
    
    try {
        // Import the BitNet module (CommonJS style)
        const BitNetModule = require('./bitnet.js');
        const bitnet = await BitNetModule();
        
        console.log('✅ Module loaded');
        
        // Initialize
        bitnet.bitnet_init();
        console.log('✅ BitNet initialized');
        
        // Load model
        const modelPath = './models/ggml-model-i2_s.gguf';
        if (!fs.existsSync(modelPath)) {
            console.log('❌ Model file not found:', modelPath);
            return;
        }
        
        const modelData = fs.readFileSync(modelPath);
        console.log(`📂 Model loaded: ${modelData.length} bytes`);
        
        // Copy to WASM memory
        const dataPtr = bitnet._malloc(modelData.length);
        bitnet.HEAPU8.set(modelData, dataPtr);
        console.log('💾 Model copied to WASM memory');
        
        // Load model
        console.log('🔄 Loading model...');
        const loadResult = bitnet.bitnet_load_model(dataPtr, modelData.length);
        bitnet._free(dataPtr);
        
        if (loadResult === 1) {
            console.log('✅ Model loaded successfully!');
            
            // Test simple inference
            console.log('🧠 Testing inference...');
            const inputText = "Hello";
            const outputBuffer = bitnet._malloc(256);
            
            const outputLen = bitnet.bitnet_inference_run(inputText, outputBuffer, 256);
            
            if (outputLen > 0) {
                const outputText = bitnet.UTF8ToString(outputBuffer);
                console.log('✅ Inference successful!');
                console.log('📝 Input:', inputText);
                console.log('🎯 Output:', outputText);
                console.log('📊 Output length:', outputLen);
            } else {
                console.log('❌ Inference returned no output');
            }
            
            bitnet._free(outputBuffer);
            
        } else {
            console.log('❌ Model loading failed');
        }
        
    } catch (error) {
        console.error('💥 Error:', error.message);
    }
}

quickTest();
