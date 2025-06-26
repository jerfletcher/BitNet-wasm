#!/usr/bin/env node

const fs = require('fs');

function analyzeAlignment() {
    console.log('🔍 BitNet WASM Alignment Analysis\n');
    
    // Check the model file
    const modelPath = '/Users/a206413899/development/BitNet-wasm/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf';
    
    if (!fs.existsSync(modelPath)) {
        console.log('❌ Model file not found');
        return;
    }
    
    const stats = fs.statSync(modelPath);
    console.log(`📊 Model: ${(stats.size / 1024 / 1024).toFixed(1)} MB`);
    
    // Read first few KB to check headers
    const buffer = fs.readFileSync(modelPath, { length: 1024 });
    
    console.log('\n🔍 File Header Analysis:');
    console.log(`Magic: ${buffer.toString('ascii', 0, 4)}`);
    
    if (buffer.toString('ascii', 0, 4) !== 'GGUF') {
        console.log('❌ Not a GGUF file');
        return;
    }
    
    const version = buffer.readUInt32LE(4);
    console.log(`Version: ${version}`);
    
    // The key insight: BitNet i2_s uses 2-bit ternary quantization
    // This means each weight uses only 2 bits, packed 16 weights per 32-bit word
    // This creates alignment issues when WASM tries to access individual weights
    
    console.log('\n⚠️  BitNet i2_s Alignment Issues:');
    console.log('1. 2-bit ternary quantization (values: -1, 0, +1)');
    console.log('2. 16 weights packed per 32-bit word');
    console.log('3. WASM requires 4-byte aligned memory access');
    console.log('4. BitNet kernels expect specific bit patterns');
    
    console.log('\n🔧 Potential Solutions:');
    console.log('1. ✅ Use Q4_0 or Q8_0 quantization instead of i2_s');
    console.log('2. ✅ Export model with standard llama.cpp quantization');
    console.log('3. ✅ Use original BitNet model with F16 precision');
    console.log('4. ❌ Fix WASM alignment (requires BitNet source changes)');
    
    console.log('\n📋 Recommended Next Steps:');
    console.log('1. Download a Q4_0 quantized version of the model');
    console.log('2. Or use the original F16 BitNet model'); 
    console.log('3. Test with standard llama.cpp quantization formats');
    
    // Check if we have alternative models
    const modelsDir = '/Users/a206413899/development/BitNet-wasm/models/BitNet-b1.58-2B-4T/';
    
    if (fs.existsSync(modelsDir)) {
        console.log('\n📁 Available model files:');
        const files = fs.readdirSync(modelsDir);
        
        for (const file of files) {
            if (file.endsWith('.gguf')) {
                const filePath = `${modelsDir}${file}`;
                const fileStats = fs.statSync(filePath);
                const sizeMB = (fileStats.size / 1024 / 1024).toFixed(1);
                
                console.log(`  ${file}: ${sizeMB} MB`);
                
                // Recommend better formats
                if (file.includes('q4_0') || file.includes('q8_0') || file.includes('f16')) {
                    console.log(`    ✅ WASM compatible format`);
                } else if (file.includes('i2_s')) {
                    console.log(`    ❌ WASM alignment issues`);
                }
            }
        }
    }
    
    console.log('\n🎯 Model Recommendation:');
    console.log('For WASM compatibility, use models with these quantizations:');
    console.log('- Q4_0: Good size/quality balance, WASM aligned');
    console.log('- Q8_0: Higher quality, still WASM compatible');  
    console.log('- F16: Best quality, larger size, fully compatible');
    console.log('- Avoid: i2_s, i4_s (BitNet specific, alignment issues)');
}

analyzeAlignment();
