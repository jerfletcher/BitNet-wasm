# Source Directory

This directory contains only the custom BitNet WASM wrapper code:

- `bitnet_inference.cpp/h` - Main BitNet inference wrapper for WASM
- `bitnet_main.js` - JavaScript interface for the WASM module  
- `build-info.cpp` - Build information utilities
- `CMakeLists.txt` - Build configuration referencing 3rdparty code

All BitNet and llama.cpp source code is referenced from `../3rdparty/BitNet/` to ensure automatic updates when the upstream repository is updated.
