# BitNet WASM Tests Directory

This directory contains all test scripts and diagnostic tools for the BitNet WASM implementation.

## 🧪 Test Scripts

### Primary Tests
- **`quick-test.js`** - Main test script for loading models and running inference
- **`test-minimal.js`** - Minimal test with reduced memory requirements

### Diagnostic Tools  
- **`analyze-model.js`** - Analyzes GGUF model format and quantization types
- **`diagnose-alignment.js`** - Detects memory alignment issues in models
- **`quick-fix.js`** - Interactive troubleshooting guide

## 🔧 How to Run Tests

### Basic Test
```bash
cd tests
node quick-test.js
```

### Minimal Memory Test
```bash
cd tests  
node test-minimal.js
```

### Model Analysis
```bash
cd tests
node analyze-model.js ../models/your-model.gguf
```

### Alignment Diagnosis
```bash
cd tests
node diagnose-alignment.js ../models/your-model.gguf
```

## 📊 Test Results Interpretation

### Success Indicators
- ✅ Model loads without alignment faults
- ✅ Context creation succeeds  
- ✅ Basic token decode works
- ✅ Valid logits generated
- ✅ Text generation produces coherent output

### Common Error Patterns

#### 1. Alignment Fault
```
Aborted(alignment fault)
```
**Cause**: SAFE_HEAP=1 in build configuration  
**Solution**: Rebuild without SAFE_HEAP=1

#### 2. Memory Access Out of Bounds
```
RuntimeError: memory access out of bounds
```
**Cause**: Insufficient WASM memory or large model  
**Solution**: Increase memory limits or reduce context size

#### 3. Model Loading Failure
```
Failed to load model from file
```
**Cause**: Incompatible model format or corrupted file  
**Solution**: Use Q4_0/Q8_0 quantization instead of i2_s

## 🎯 Current Status Summary

As documented in the main README, the BitNet WASM implementation has the following status:

- **Build System**: ✅ Working
- **Model Loading**: ✅ Working (with compatible formats)
- **Memory Management**: ⚠️ Needs tuning for larger models
- **Inference Pipeline**: 🔄 In progress (memory bounds issue)
- **Text Generation**: 🔄 Pending memory fix

## 🚀 Next Steps

1. Fix memory bounds issue by reducing context size
2. Test with smaller models (100-200MB range)
3. Validate text generation quality
4. Optimize for production use
