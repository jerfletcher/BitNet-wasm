#!/usr/bin/env node

const fs = require('fs');

// Test with a different model entirely - try Phi-3.5 mini which is known to work well with WASM
const phi3Model = 'models/test/Phi-3.5-mini-instruct-Q4_K_M.gguf';

console.log('🔍 BitNet WASM Quick Fix - Alternative Model Test');
console.log('================================================');

// Check if we need to download Phi-3.5 instead
if (!fs.existsSync(phi3Model)) {
    console.log('');
    console.log('💡 The issue appears to be with WASM memory alignment during model loading.');
    console.log('Let\'s try a different approach:');
    console.log('');
    console.log('1️⃣ Try a smaller, simpler model known to work with WASM:');
    console.log('   cd models/test');
    console.log('   curl -L -o "Phi-3.5-mini-instruct-Q4_K_M.gguf" \\');
    console.log('        "https://huggingface.co/microsoft/Phi-3.5-mini-instruct-gguf/resolve/main/Phi-3.5-mini-instruct-Q4_K_M.gguf"');
    console.log('');
    console.log('2️⃣ OR try rebuilding with different WASM memory settings:');
    console.log('   Edit build.sh and change:');
    console.log('   - INITIAL_MEMORY=128MB (instead of 256MB)');
    console.log('   - Remove SAFE_HEAP=1 (might cause alignment checks)');
    console.log('   - Add -s MAXIMUM_MEMORY=1GB');
    console.log('');
    console.log('3️⃣ OR the issue might be BitNet-specific WASM incompatibility:');
    console.log('   The BitNet kernels may have alignment requirements incompatible with WASM');
    console.log('   regardless of the model quantization format.');
    console.log('');
    
    // Check current build settings
    if (fs.existsSync('build.sh')) {
        const buildContent = fs.readFileSync('build.sh', 'utf8');
        console.log('🔧 Current build.sh analysis:');
        
        if (buildContent.includes('SAFE_HEAP=1')) {
            console.log('   ⚠️  SAFE_HEAP=1 detected - this may cause alignment faults');
        }
        
        if (buildContent.includes('INITIAL_MEMORY=256MB')) {
            console.log('   📏 INITIAL_MEMORY=256MB - try reducing to 128MB');
        }
        
        if (!buildContent.includes('MAXIMUM_MEMORY')) {
            console.log('   🔄 Missing MAXIMUM_MEMORY - add -s MAXIMUM_MEMORY=1GB');
        }
        
        console.log('');
        console.log('4️⃣ Quick fix attempt - modify build.sh:');
        console.log('   Would you like me to create a modified build script? (Y/n)');
    }
} else {
    console.log('✅ Alternative model found, testing...');
    // Update paths to use Phi-3.5
    // ... rest of test logic
}
