const fs = require('fs');

async function quickTest() {
    console.log('🚀 Quick BitNet test starting...');
    
    try {
        // Import the BitNet module (CommonJS style)
        const BitNetModule = require('../bitnet.js');
        const bitnet = await BitNetModule();
        
        console.log('✅ Module loaded');
        
        // Create wrapper functions using ccall/cwrap
        const bitnet_init = bitnet.cwrap('bitnet_init', null, []);
        const bitnet_load_model_from_memory = bitnet.cwrap('bitnet_load_model_from_memory', 'number', ['number', 'number']);
        const bitnet_run_inference_simple = bitnet.cwrap('bitnet_run_inference_simple', 'string', ['string', 'number']);
        const bitnet_is_model_loaded = bitnet.cwrap('bitnet_is_model_loaded', 'number', []);
        const bitnet_get_vocab_size = bitnet.cwrap('bitnet_get_vocab_size', 'number', []);
        const bitnet_get_embedding_dim = bitnet.cwrap('bitnet_get_embedding_dim', 'number', []);
        const bitnet_get_num_layers = bitnet.cwrap('bitnet_get_num_layers', 'number', []);
        
        // Initialize BitNet
        bitnet_init();
        console.log('✅ BitNet initialized');
        
        // Load model
        const modelPath = 'models/tiny/tinyllama-1.1b-chat-v1.0.q4_0.gguf';
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
        
        // Load model using new interface
        console.log('🔄 Loading model...');
        const loadResult = bitnet_load_model_from_memory(dataPtr, modelData.length);
        bitnet._free(dataPtr);
        
        if (loadResult === 1) {
            console.log('✅ Model loaded successfully!');
            
            // Get model info
            const vocabSize = bitnet_get_vocab_size();
            const embeddingDim = bitnet_get_embedding_dim();
            const numLayers = bitnet_get_num_layers();
            
            console.log(`📊 Model Info:`);
            console.log(`   Vocabulary size: ${vocabSize}`);
            console.log(`   Embedding dimension: ${embeddingDim}`);
            console.log(`   Number of layers: ${numLayers}`);
            
            // Test simple inference
            console.log('🧠 Testing inference...');
            const inputText = "Hello";
            const maxTokens = 32;
            
            console.log(`📝 Input: "${inputText}"`);
            console.log(`🎯 Generating up to ${maxTokens} tokens...`);
            
            const outputText = bitnet_run_inference_simple(inputText, maxTokens);
            
            if (outputText && outputText.length > 0) {
                console.log('✅ Inference successful!');
                console.log(`🎯 Output: "${outputText}"`);
                console.log(`� Output length: ${outputText.length} characters`);
            } else {
                console.log('❌ Inference returned no output');
            }
            
        } else {
            console.log('❌ Model loading failed');
        }
        
    } catch (error) {
        console.error('💥 Error:', error.message);
        console.error(error.stack);
    }
}

quickTest();
