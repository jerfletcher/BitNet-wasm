# BitNet-WASM: WebAssembly Package for BitNet Operations

This project provides a complete solution for running BitNet operations in the browser using WebAssembly (WASM). It includes a build system, core BitNet operations, and example code for matrix multiplication, tensor transformation, and model loading. The implementation is based on the BitNet paper and uses the Emscripten SDK to compile C++ code to WebAssembly.

## Build Process

The build process is now simplified with the `setup_and_build.sh` script, which handles all dependencies and builds the WASM module.

### Prerequisites

*   A Linux-like environment with bash
*   Internet connection (for downloading dependencies)
*   Basic build tools (will be installed if missing)

### Automated Setup and Build

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/jerfletcher/BitNet-wasm.git
    cd BitNet-wasm
    ```

2.  **Run the Setup and Build Script:**
    ```bash
    ./setup_and_build.sh
    ```
    
    This script performs the following actions:
    *   Installs git if not already installed
    *   Initializes git submodules (llama.cpp and llama-cpp-wasm)
    *   Downloads and installs Emscripten SDK 4.0.8
    *   Creates a models directory
    *   Installs Python and the huggingface_hub package
    *   Downloads a sample BitNet model from Hugging Face
    *   Runs the build script to compile the WASM module
    *   Provides instructions for testing the example

### Manual Build (Advanced)

If you prefer to manually control the build process, you can use the `build.sh` script directly:

1.  **Ensure Prerequisites:**
    *   Emscripten SDK 4.0.8 installed and activated
    *   Git submodules initialized

2.  **Run the Build Script:**
    ```bash
    ./build.sh
    ```
    
    This script performs the following actions:
    *   Copies necessary kernel header files
    *   Creates stub implementations for required GGML functions
    *   Compiles the C/C++ sources using `emcc`
    *   Generates `bitnet.wasm` (the WASM module) and `bitnet.js` (the JavaScript loader/glue code)
    *   Logs build output to `emcc_stdout.log` and `emcc_stderr.log`

## WASM Module Interface

The compiled WASM module (`bitnet.wasm` loaded via `bitnet.js`) exports the following C functions.
You can call these from JavaScript using `Module.ccall` or `Module.cwrap` after the module is loaded. Remember to prefix C function names with an underscore when using `EXPORTED_FUNCTIONS` or directly accessing them on the `Module` object (e.g., `Module._ggml_init`).

### Exported Functions

*   `void ggml_init(struct ggml_init_params params)`
    *   **C Signature:** `void ggml_init(struct ggml_init_params params);` (from `ggml.h`)
    *   **JS Access Example:** `Module._ggml_init(0);` (passing NULL for params)
    *   **Note:** `struct ggml_init_params` would need to be created in WASM memory if non-default initialization is needed. `params_ptr` would be a pointer to this structure.
    *   **Status:** Available.

*   `int64_t ggml_nelements(const struct ggml_tensor * tensor)`
    *   **C Signature:** `int64_t ggml_nelements(const struct ggml_tensor * tensor);` (from `ggml.h`)
    *   **JS Access Example:** `Module._ggml_nelements(tensor_ptr)`
    *   **Note:** Requires a valid `ggml_tensor` pointer. Creating and managing tensors is currently problematic (see "Limitations").
    *   **Status:** Available (but of limited use without tensor creation).

*   `void ggml_bitnet_init(void)`
    *   **C Signature:** `void ggml_bitnet_init(void);` (from `ggml-bitnet.h`, defined in `ggml-bitnet-lut.cpp`)
    *   **JS Access Example:** `Module._ggml_bitnet_init()`
    *   **Status:** Available. Initializes BitNet specific internal states (related to LUT kernels).

*   `void ggml_bitnet_free(void)`
    *   **C Signature:** `void ggml_bitnet_free(void);` (from `ggml-bitnet.h`, defined in `ggml-bitnet-lut.cpp`)
    *   **JS Access Example:** `Module._ggml_bitnet_free()`
    *   **Status:** Available. Frees resources allocated by `ggml_bitnet_init`.

## Current Status & Limitations

The build process has been updated:
*   The Emscripten SDK (currently v4.0.8) is installed in the project directory (`./emsdk`). The `build.sh` script sources `emsdk_env.sh` and calls `emcc` directly.
*   The `build.sh` script exports specific functions needed for BitNet operations. These functions are available on the `Module` object in JavaScript (prefixed with an underscore).

**Build Artifacts:**
*   `bitnet.js` (JavaScript glue code)
*   `bitnet.wasm` (WebAssembly module)

**Key Available Functions (callable from JavaScript via `Module._functionName`):**
*   `ggml_init` - Initialize the GGML library
*   `ggml_bitnet_init` - Initialize BitNet-specific resources
*   `ggml_bitnet_free` - Free BitNet-specific resources
*   `ggml_nelements` - Get the number of elements in a tensor
*   `ggml_bitnet_transform_tensor` - Transform a tensor using BitNet quantization
*   `ggml_bitnet_mul_mat_task_compute` - Perform matrix multiplication with BitNet quantization
*   Many other `ggml_*` functions (due to `EXPORT_ALL=1`).

## Implementation Details

### BitNet Core Functions

The core BitNet functionality is implemented in `src/ggml-bitnet-lut.cpp` with the following key components:

1. **Memory Management**:
   * `ggml_bitnet_init()` - Allocates memory for lookup tables and other resources
   * `ggml_bitnet_free()` - Frees allocated resources

2. **Matrix Multiplication**:
   * `ggml_bitnet_mul_mat_task_compute()` - Performs matrix multiplication with BitNet quantization
   * Supports 2-bit quantization for weights (-1, 0, 1)
   * Uses scaling factors for improved accuracy

3. **Tensor Transformation**:
   * `ggml_bitnet_transform_tensor()` - Transforms a tensor using BitNet quantization
   * Quantizes floating-point values to 2-bit representation

### JavaScript Integration

The JavaScript integration is implemented in `example/main.js` with the following key components:

1. **Memory Management**:
   * Helper functions for allocating and freeing memory in the WASM heap
   * Functions for transferring data between JavaScript and WASM

2. **Matrix Operations**:
   * Functions for parsing and creating matrices
   * Implementation of matrix multiplication using BitNet functions

3. **Model Loading**:
   * Functions for loading a BitNet model from a GGUF file
   * Memory management for the loaded model

4. **User Interface**:
   * Event handlers for the demo buttons
   * Functions for displaying results

**Current Implementation Status:**
1.  **Core BitNet Functions:**
    *   The essential C++ functions `ggml_bitnet_mul_mat_task_compute` and `ggml_bitnet_transform_tensor` (defined in `src/ggml-bitnet-lut.cpp`) have been implemented with basic functionality.
    *   These implementations support 2-bit quantization for weights (-1, 0, 1) and matrix multiplication with scaling factors.
    *   The matrix multiplication function has been optimized to properly handle memory allocation and data transfer between JavaScript and WASM.
    *   The tensor transformation function has been implemented to demonstrate BitNet quantization.
2.  **Basic Model Loading:**
    *   The example now includes functionality to load a BitNet model in GGUF format.
    *   The current implementation loads the model into memory but does not yet perform full inference with the model.
3.  **Demonstration UI:**
    *   The example UI demonstrates matrix multiplication, tensor transformation, and model loading.
    *   The matrix multiplication demo allows users to input custom matrices and see the results of BitNet quantized multiplication.
    *   The tensor transformation demo shows how BitNet quantization affects tensor values.

**Conclusion:** The WASM module can be built, loaded, and basic BitNet operations can be performed. The matrix multiplication and tensor transformation demos are fully functional. However, **full model inference requires additional work as outlined below.**

## Next Steps for Full Functionality

While the core BitNet functions have been implemented and the example has been updated to demonstrate basic functionality, several key steps are still needed to achieve full model inference capabilities:

1.  **Complete the Model Inference Pipeline:**
    *   **Implement Model Parsing:** Add functionality to parse the GGUF model format and extract weights, configuration, and vocabulary.
    *   **Implement Tokenization:** Add a tokenizer to convert input text to token IDs that can be processed by the model.
    *   **Implement Forward Pass:** Create a complete forward pass through the model's layers using the implemented BitNet operations.
    *   **Implement Output Processing:** Add functionality to convert model output logits to tokens and then to text.

2.  **Optimize BitNet Operations:**
    *   **Performance Optimization:** Optimize the current BitNet operations for better performance in WebAssembly.
    *   **Memory Management:** Improve memory management to handle larger models efficiently.
    *   **SIMD Support:** Add SIMD (Single Instruction, Multiple Data) support for faster matrix operations where available.
    *   **Quantization Improvements:** Extend the quantization support to handle different bit-widths and quantization schemes.

3.  **Enhance JavaScript API:**
    *   **Create High-Level API:** Develop a more user-friendly JavaScript API that abstracts the low-level WASM calls.
    *   **Add Streaming Support:** Implement streaming for both input and output to handle long sequences.
    *   **Add Progress Reporting:** Add callbacks for progress reporting during model loading and inference.
    *   **Error Handling:** Improve error handling and reporting throughout the API.

4.  **Improve Model Support:**
    *   **Model Configuration:** Add support for different model architectures and configurations.
    *   **Model Conversion Tools:** Provide tools to convert models from other formats (e.g., PyTorch, Hugging Face) to the GGUF format.
    *   **Model Compression:** Implement techniques to further compress models for web deployment.

5.  **Documentation and Examples:**
    *   **API Documentation:** Create comprehensive documentation for the JavaScript API.
    *   **Usage Examples:** Provide more examples demonstrating different use cases.
    *   **Integration Guides:** Create guides for integrating with popular web frameworks.
    *   **Performance Benchmarks:** Add benchmarks to help users understand performance characteristics.

6.  **Testing and Validation:**
    *   **Unit Tests:** Develop comprehensive unit tests for all components.
    *   **Integration Tests:** Create integration tests for the full inference pipeline.
    *   **Browser Compatibility:** Test across different browsers and devices.
    *   **Validation Suite:** Develop a validation suite to compare outputs with reference implementations.

## Example Usage

An example demonstrating how to use the BitNet WASM module is provided in the `example/` directory:
*   `example/index.html` - The HTML interface with three demo sections:
    * BitNet Model Inference Demo - Load a BitNet model and run simulated inference
    * Matrix Multiplication Demo - Perform matrix multiplication with BitNet quantization
    * Tensor Transformation Demo - Transform tensors using BitNet quantization
*   `example/main.js` - The JavaScript code that interacts with the WASM module

### Running the Example

If you used the `setup_and_build.sh` script, all the necessary components should already be set up. You can simply:

1.  Start a simple HTTP server in the `BitNet-wasm` root directory:
    ```bash
    python3 -m http.server 8000
    ```
2.  Open `http://localhost:8000/example/index.html` in your browser to see the demo.
3.  Open `http://localhost:8000/tests/index.html` in your browser to run the tests.

If you're setting up manually:

1.  Ensure `bitnet.js` and `bitnet.wasm` are built using `./build.sh`.
2.  Download a BitNet model in GGUF format:
    ```bash
    # Create a models directory
    mkdir -p models
    
    # Download a BitNet model from Hugging Face
    # You can use the huggingface_hub Python package:
    python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='microsoft/BitNet-b1.58-2B-4T-gguf', filename='ggml-model-i2_s.gguf', local_dir='models')"
    ```
3.  Start a simple HTTP server in the `BitNet-wasm` root directory:
    ```bash
    python3 -m http.server 8000
    ```
4.  Open `http://localhost:8000/example/index.html` in your browser.

### Example Features

The example demonstrates:

1. **Model Loading**:
   * Loading a BitNet model from a GGUF file
   * Displaying model information
   * Simulating inference with the loaded model

2. **Matrix Operations**:
   * Matrix multiplication with BitNet quantization
   * Visualization of matrix operation results

3. **Tensor Transformations**:
   * Transforming tensors using BitNet quantization
   * Displaying the transformation results

**Note:** While the matrix multiplication and tensor transformation demos perform actual computations using the implemented BitNet functions, the model inference demo currently simulates the inference process. Full model inference requires implementing the additional components described in the "Next Steps" section.

## Native GGUF-to-Flat Conversion Test (C++/WASM)

This project includes a C++ parser and converter for BitNet GGUF models, used both in WASM and as a native test. This is required to run real BitNet models in the browser or natively.

### How to Run the Native Test

1. Build the test program:
   ```bash
   g++ -Iinclude -o test_gguf_to_flat src/test_gguf_to_flat.cpp src/ggml-bitnet-lut.cpp
   ```
2. Run the test with a GGUF model:
   ```bash
   ./test_gguf_to_flat models/ggml-model-i2_s.gguf out.flat
   ```
   This will attempt to convert the GGUF model to a flat binary file (`out.flat`).
   The program prints detailed debug output about the GGUF parsing process.

### Current Status and Troubleshooting
- The parser reads the GGUF header and kv pairs, robustly skipping known types.
- If it encounters an unknown kv type, it breaks out of the kv loop and assumes the tensor section starts.
- In the current BitNet GGUF file, after about 24 kvs, a kv with an unknown type (e.g., type=2 for key `token_embd.weight`) is encountered. The parser then tries to parse the tensor section.
- However, the next bytes do not correspond to a valid tensor section (the name length is implausible, e.g., a huge number), so tensor parsing aborts.
- No tensors are parsed, and the conversion fails with `tok_embeddings.weight not found or wrong type`.

**This means the parser desynchronizes at custom kv types.**

#### To continue troubleshooting:
- Inspect the bytes at the kv/tensor boundary in the GGUF file to determine the correct way to skip unknown kv types.
- Compare with the Python GGUF parser or BitNet's own conversion scripts in `3rdparty/BitNet/utils/` to see how they handle these kvs.
- Adjust the C++ parser to either:
  - Correctly skip unknown/custom kv types (by reading the correct length and skipping), or
  - Use a more robust heuristic to detect the start of the tensor section.
- Add more debug prints for the raw bytes at the kv/tensor boundary to help diagnose the issue.

See `README_native_test.md` for more details and troubleshooting steps.

---

## Project Status and What Remains

- **WASM pipeline:**
  - Matrix multiplication and tensor transformation are fully functional in the browser.
  - Model loading and GGUF-to-flat conversion logic is implemented, but full model inference is not yet complete.
  - The JS loader can call the WASM GGUF-to-flat converter, but the parser needs to be fixed for some BitNet GGUF files.

- **Native pipeline:**
  - The native test for GGUF-to-flat conversion is implemented and debuggable.
  - The parser currently fails at custom kv types in some BitNet GGUF files (see above).

- **What remains to be implemented:**
  - Fix the GGUF parser to robustly handle all kv types and correctly find the tensor section for all BitNet GGUF files.
  - Complete the forward pass and output processing for real model inference in both WASM and native.
  - Add a tokenizer and output-to-text logic for end-to-end text generation.
  - Improve error handling, documentation, and add more usage examples.

**Summary:**
- The project supports BitNet quantized ops and model loading in WASM, and can convert models natively (with caveats).
- The GGUF-to-flat parser is the main remaining blocker for full model inference.
- See `README_native_test.md` for the latest troubleshooting state and how to continue development.

---

