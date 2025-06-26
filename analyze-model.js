#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

function analyzeGGUFModel(filePath) {
    console.log(`📊 Analyzing GGUF model: ${filePath}`);
    
    const buffer = fs.readFileSync(filePath);
    console.log(`📏 File size: ${buffer.length} bytes (${(buffer.length / 1024 / 1024).toFixed(1)} MB)`);
    
    // Read GGUF magic number
    const magic = buffer.toString('ascii', 0, 4);
    console.log(`🔮 Magic: "${magic}"`);
    
    if (magic !== 'GGUF') {
        console.log('❌ Not a valid GGUF file');
        return;
    }
    
    // Read version
    const version = buffer.readUInt32LE(4);
    console.log(`📱 Version: ${version}`);
    
    // Read tensor count (64-bit)
    const tensorCount = Number(buffer.readBigUInt64LE(8));
    console.log(`🎯 Tensor count: ${tensorCount}`);
    
    // Read metadata count (64-bit)
    const metadataCount = Number(buffer.readBigUInt64LE(16));
    console.log(`📋 Metadata count: ${metadataCount}`);
    
    let offset = 24; // After header
    
    console.log('\n📋 Metadata:');
    
    // Read metadata
    for (let i = 0; i < metadataCount; i++) {
        if (offset >= buffer.length - 8) break;
        
        // Read key length (64-bit)
        const keyLen = Number(buffer.readBigUInt64LE(offset));
        offset += 8;
        
        if (offset + keyLen >= buffer.length) break;
        
        // Read key
        const key = buffer.toString('utf8', offset, offset + keyLen);
        offset += keyLen;
        
        if (offset >= buffer.length - 4) break;
        
        // Read value type
        const valueType = buffer.readUInt32LE(offset);
        offset += 4;
        
        let value = 'unknown';
        
        switch (valueType) {
            case 8: // UINT8
                if (offset < buffer.length) {
                    value = buffer.readUInt8(offset);
                    offset += 1;
                }
                break;
            case 9: // INT8  
                if (offset < buffer.length) {
                    value = buffer.readInt8(offset);
                    offset += 1;
                }
                break;
            case 10: // UINT16
                if (offset + 1 < buffer.length) {
                    value = buffer.readUInt16LE(offset);
                    offset += 2;
                }
                break;
            case 11: // INT16
                if (offset + 1 < buffer.length) {
                    value = buffer.readInt16LE(offset);
                    offset += 2;
                }
                break;
            case 12: // UINT32
                if (offset + 3 < buffer.length) {
                    value = buffer.readUInt32LE(offset);
                    offset += 4;
                }
                break;
            case 13: // INT32
                if (offset + 3 < buffer.length) {
                    value = buffer.readInt32LE(offset);
                    offset += 4;
                }
                break;
            case 14: // FLOAT32
                if (offset + 3 < buffer.length) {
                    value = buffer.readFloatLE(offset);
                    offset += 4;
                }
                break;
            case 15: // BOOL
                if (offset < buffer.length) {
                    value = buffer.readUInt8(offset) === 1;
                    offset += 1;
                }
                break;
            case 16: // STRING
                if (offset + 7 < buffer.length) {
                    const strLen = Number(buffer.readBigUInt64LE(offset));
                    offset += 8;
                    if (offset + strLen <= buffer.length) {
                        value = buffer.toString('utf8', offset, offset + strLen);
                        offset += strLen;
                    }
                }
                break;
            case 17: // ARRAY
                // Skip arrays for now - complex parsing
                if (offset + 7 < buffer.length) {
                    const arrType = buffer.readUInt32LE(offset);
                    offset += 4;
                    const arrLen = Number(buffer.readBigUInt64LE(offset));
                    offset += 8;
                    value = `[array of ${arrLen} items, type ${arrType}]`;
                    // Skip actual array data - would need proper parsing
                    offset += arrLen * getSizeForType(arrType);
                }
                break;
            default:
                value = `[type ${valueType}]`;
                break;
        }
        
        // Print important metadata
        if (key.includes('tokenizer') || key.includes('quantization') || key.includes('model') || 
            key.includes('arch') || key.includes('bitnet') || key.includes('general')) {
            console.log(`  ${key}: ${value}`);
        }
    }
    
    console.log('\n🧮 Tensor Info:');
    
    // Analyze first few tensors for alignment info
    for (let i = 0; i < Math.min(5, tensorCount) && offset < buffer.length - 32; i++) {
        if (offset >= buffer.length - 8) break;
        
        // Read tensor name length (64-bit)
        const nameLen = Number(buffer.readBigUInt64LE(offset));
        offset += 8;
        
        if (offset + nameLen >= buffer.length) break;
        
        // Read tensor name
        const name = buffer.toString('utf8', offset, offset + nameLen);
        offset += nameLen;
        
        if (offset >= buffer.length - 16) break;
        
        // Read dimensions count
        const nDims = buffer.readUInt32LE(offset);
        offset += 4;
        
        // Read type
        const type = buffer.readUInt32LE(offset);
        offset += 4;
        
        console.log(`  Tensor ${i}: "${name}"`);
        console.log(`    Type: ${type} (${getTypeDescription(type)})`);
        console.log(`    Dimensions: ${nDims}`);
        
        // Read dimensions
        const dims = [];
        for (let d = 0; d < nDims && offset + 7 < buffer.length; d++) {
            dims.push(Number(buffer.readBigUInt64LE(offset)));
            offset += 8;
        }
        console.log(`    Shape: [${dims.join(', ')}]`);
        
        if (offset >= buffer.length - 8) break;
        
        // Read data offset (64-bit)
        const dataOffset = Number(buffer.readBigUInt64LE(offset));
        offset += 8;
        
        console.log(`    Data offset: ${dataOffset} (0x${dataOffset.toString(16)})`);
        console.log(`    Alignment: ${dataOffset % 4 === 0 ? '4-byte' : 'unaligned'}, ${dataOffset % 8 === 0 ? '8-byte' : 'not-8-byte'}, ${dataOffset % 16 === 0 ? '16-byte' : 'not-16-byte'}`);
        
        // Check for i2_s quantization issues
        if (type === 16 || getTypeDescription(type).includes('i2')) { // BitNet i2_s is typically type 16
            console.log(`    ⚠️  BitNet quantization detected - alignment critical!`);
            
            // Analyze data for patterns that might cause alignment issues
            if (dataOffset < buffer.length - 64) {
                const sample = buffer.subarray(dataOffset, Math.min(dataOffset + 64, buffer.length));
                console.log(`    First bytes: ${Array.from(sample.slice(0, 16)).map(b => '0x' + b.toString(16).padStart(2, '0')).join(' ')}`);
                
                // Check for potential alignment issues in data
                let hasAlignmentIssues = false;
                for (let j = 0; j < Math.min(32, sample.length - 4); j += 4) {
                    const word = sample.readUInt32LE(j);
                    if (word === 0 && j > 0) continue; // Skip zero padding
                    
                    // BitNet i2_s uses specific bit patterns that might not align properly
                    const bits = word.toString(2).padStart(32, '0');
                    if (bits.includes('10101010') || bits.includes('01010101')) {
                        hasAlignmentIssues = true;
                        break;
                    }
                }
                
                if (hasAlignmentIssues) {
                    console.log(`    ❌ Potential BitNet alignment pattern detected`);
                }
            }
        }
    }
}

function getSizeForType(type) {
    switch (type) {
        case 8: case 9: case 15: return 1;   // UINT8, INT8, BOOL
        case 10: case 11: return 2;          // UINT16, INT16  
        case 12: case 13: case 14: return 4; // UINT32, INT32, FLOAT32
        case 16: return 8;                   // STRING (length)
        default: return 4;
    }
}

function getTypeDescription(type) {
    const types = {
        0: 'F32', 1: 'F16', 2: 'Q4_0', 3: 'Q4_1', 4: 'Q4_2', 5: 'Q4_3',
        6: 'Q5_0', 7: 'Q5_1', 8: 'Q8_0', 9: 'Q8_1', 10: 'Q2_K', 11: 'Q3_K',
        12: 'Q4_K', 13: 'Q5_K', 14: 'Q6_K', 15: 'Q8_K', 16: 'I8', 17: 'I16',
        18: 'I32', 19: 'I64', 20: 'F64', 21: 'IQ2_XXS', 22: 'IQ2_XS', 23: 'IQ3_XXS',
        24: 'IQ1_S', 25: 'IQ4_NL', 26: 'IQ3_S', 27: 'IQ2_S', 28: 'IQ4_XS',
        29: 'I1', 30: 'BF16', 31: 'Q4_0_4_4', 32: 'Q4_0_4_8', 33: 'Q4_0_8_8'
    };
    
    // Check for BitNet specific types (often use custom type numbers)
    if (type >= 100 && type <= 110) {
        return `BitNet_i2_s (type ${type})`;
    }
    
    return types[type] || `Unknown(${type})`;
}

// Analyze the model
const modelPath = process.argv[2] || '/Users/a206413899/development/BitNet-wasm/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf';

if (!fs.existsSync(modelPath)) {
    console.log(`❌ Model file not found: ${modelPath}`);
    console.log('Usage: node analyze-model.js [model-path]');
    process.exit(1);
}

analyzeGGUFModel(modelPath);
