// BitNet-WASM Test Script
// This script tests the functionality of the BitNet-WASM module

// Import the BitNet module
import BitNetModule from '../bitnet.js';

// Test results
const testResults = {
    passed: 0,
    failed: 0,
    total: 0
};

// Test function
function test(name, testFn) {
    testResults.total++;
    console.log(`Running test: ${name}`);
    try {
        testFn();
        console.log(`✅ Test passed: ${name}`);
        testResults.passed++;
    } catch (error) {
        console.error(`❌ Test failed: ${name}`);
        console.error(error);
        testResults.failed++;
    }
}

// Helper function to compare arrays with a tolerance
function compareArrays(a, b, tolerance = 1e-5) {
    if (a.length !== b.length) {
        throw new Error(`Array lengths don't match: ${a.length} vs ${b.length}`);
    }
    
    for (let i = 0; i < a.length; i++) {
        if (Math.abs(a[i] - b[i]) > tolerance) {
            throw new Error(`Arrays differ at index ${i}: ${a[i]} vs ${b[i]}`);
        }
    }
    
    return true;
}

// Initialize the BitNet module
BitNetModule().then(wasmModule => {
    console.log('BitNet WASM module loaded');
    
    // Test BitNet initialization
    test('BitNet initialization', () => {
        wasmModule._ggml_bitnet_init();
        // If no error is thrown, the test passes
    });
    
    // Test matrix multiplication
    test('Matrix multiplication', () => {
        // Create input matrix: 2x3
        const inputData = new Float32Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        const inputPtr = wasmModule._malloc(inputData.length * Float32Array.BYTES_PER_ELEMENT);
        wasmModule.HEAPF32.set(inputData, inputPtr / Float32Array.BYTES_PER_ELEMENT);
        
        // Create quantized weights matrix: 3x2
        const qWeightsData = new Int8Array([1, 0, -1, 1, 0, -1]);
        const qWeightsPtr = wasmModule._malloc(qWeightsData.length * Int8Array.BYTES_PER_ELEMENT);
        wasmModule.HEAP8.set(qWeightsData, qWeightsPtr);
        
        // Create scales
        const scalesData = new Float32Array([0.5, 0.5, 0.5]);
        const scalesPtr = wasmModule._malloc(scalesData.length * Float32Array.BYTES_PER_ELEMENT);
        wasmModule.HEAPF32.set(scalesData, scalesPtr / Float32Array.BYTES_PER_ELEMENT);
        
        // Create LUT scales
        const lutScalesData = new Float32Array([127.0]);
        const lutScalesPtr = wasmModule._malloc(lutScalesData.length * Float32Array.BYTES_PER_ELEMENT);
        wasmModule.HEAPF32.set(lutScalesData, lutScalesPtr / Float32Array.BYTES_PER_ELEMENT);
        
        // Create LUT biases
        const lutBiasesData = new Float32Array([0.0]);
        const lutBiasesPtr = wasmModule._malloc(lutBiasesData.length * Float32Array.BYTES_PER_ELEMENT);
        wasmModule.HEAPF32.set(lutBiasesData, lutBiasesPtr / Float32Array.BYTES_PER_ELEMENT);
        
        // Create output matrix: 2x2
        const outputPtr = wasmModule._malloc(2 * 2 * Float32Array.BYTES_PER_ELEMENT);
        
        // Call the BitNet matrix multiplication function
        wasmModule._ggml_bitnet_mul_mat_task_compute(
            inputPtr,
            scalesPtr,
            qWeightsPtr,
            lutScalesPtr,
            lutBiasesPtr,
            outputPtr,
            2, // rows of input matrix
            3, // cols of input matrix
            2, // cols of weight matrix
            2  // 2-bit quantization
        );
        
        // Read the result
        const result = new Float32Array(2 * 2);
        for (let i = 0; i < 2 * 2; i++) {
            result[i] = wasmModule.HEAPF32[outputPtr / Float32Array.BYTES_PER_ELEMENT + i];
        }
        
        // Expected result: input * weights with quantization
        // For this simple test, we're just checking that the result is not all zeros
        const hasNonZero = result.some(val => val !== 0);
        if (!hasNonZero) {
            throw new Error('Matrix multiplication result is all zeros');
        }
        
        // Free memory
        wasmModule._free(inputPtr);
        wasmModule._free(qWeightsPtr);
        wasmModule._free(scalesPtr);
        wasmModule._free(lutScalesPtr);
        wasmModule._free(lutBiasesPtr);
        wasmModule._free(outputPtr);
    });
    
    // Test tensor transformation
    test('Tensor transformation', () => {
        // Create input tensor
        const inputData = new Float32Array([0.5, -0.3, 0.1, 0.0, 0.7, -0.2, 0.4, -0.6]);
        const inputPtr = wasmModule._malloc(inputData.length * Float32Array.BYTES_PER_ELEMENT);
        wasmModule.HEAPF32.set(inputData, inputPtr / Float32Array.BYTES_PER_ELEMENT);
        
        // Create output tensor
        const outputPtr = wasmModule._malloc(inputData.length * Float32Array.BYTES_PER_ELEMENT);
        
        // Call the BitNet tensor transformation function
        wasmModule._ggml_bitnet_transform_tensor(
            inputPtr,
            outputPtr,
            inputData.length,
            2 // 2-bit quantization
        );
        
        // Read the result
        const result = new Float32Array(inputData.length);
        for (let i = 0; i < inputData.length; i++) {
            result[i] = wasmModule.HEAPF32[outputPtr / Float32Array.BYTES_PER_ELEMENT + i];
        }
        
        // For this simple test, we're just checking that the result is not all zeros
        const hasNonZero = result.some(val => val !== 0);
        if (!hasNonZero) {
            throw new Error('Tensor transformation result is all zeros');
        }
        
        // Free memory
        wasmModule._free(inputPtr);
        wasmModule._free(outputPtr);
    });
    
    // Test BitNet cleanup
    test('BitNet cleanup', () => {
        wasmModule._ggml_bitnet_free();
        // If no error is thrown, the test passes
    });
    
    // Print test results
    console.log(`\nTest Results: ${testResults.passed}/${testResults.total} tests passed`);
    if (testResults.failed > 0) {
        console.error(`${testResults.failed} tests failed`);
    } else {
        console.log('All tests passed!');
    }
}).catch(error => {
    console.error('Failed to load BitNet WASM module:', error);
});