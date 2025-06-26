# BitNet WASM Project Structure

This project is now cleanly organized to reference 3rdparty code directly:

## Core Project Files
- `src/` - Custom WASM wrapper code only
- `3rdparty/BitNet/` - BitNet repository (submodule)
- `build.sh` - Build script that references 3rdparty sources
- `CMakeLists.txt` - Build configuration

## Generated Files  
- `bitnet.wasm` / `bitnet.js` - Built WASM module
- `include/` - Generated kernel headers (copied from 3rdparty presets)
- `emsdk/` - Emscripten SDK

## Test & Demo Files
- `examples/` - Web demos
- `quick-test.js` - Node.js test
- `test-real-model.js` - Model testing
- `test.html` - Browser test

## Benefits of This Structure

1. **Automatic Updates**: Any updates to the BitNet repository are automatically included when rebuilding
2. **Clean Separation**: Only custom WASM wrapper code is maintained locally
3. **Easy Maintenance**: No duplicate code to keep in sync
4. **Clear Dependencies**: Build system explicitly references 3rdparty sources

To rebuild with updated 3rdparty code:
```bash
./build.sh
```
