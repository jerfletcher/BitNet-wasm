# BitNet-WASM: WebAssembly Package for BitNet Operations

This project provides a build process to compile parts of the BitNet-wasm C++ code into a WebAssembly (WASM) package.
The build uses Docker and the Emscripten SDK.

## Build Process

The build is orchestrated by the `build.sh` script.

### Prerequisites

*   Docker
*   A Linux-like environment with bash (the script uses `sudo dockerd` if Docker is not running, which might require passwordless sudo or manual Docker daemon startup).

### Steps

1.  **Clone the Repository (if not already done for this session):**
    ```bash
    # git clone https://github.com/jerfletcher/BitNet-wasm.git
    # cd BitNet-wasm
    # git submodule update --init --recursive 3rdparty/llama.cpp
    # (Note: The build script currently uses llama.cpp commit 5eb47b72 from the submodule)
    ```

2.  **Run the Build Script:**
    From the root of the `BitNet-wasm` directory:
    ```bash
    ./build.sh
    ```
    This script performs the following actions:
    *   Sets up the Emscripten Docker environment (`emscripten/emsdk:4.0.8`).
    *   Copies necessary kernel header files.
    *   Compiles the C/C++ sources (`src/*.cpp`, `3rdparty/llama.cpp/ggml/src/*.c`, `3rdparty/llama.cpp/ggml/src/*.cpp`) using `emcc`.
    *   Generates `bitnet.wasm` (the WASM module) and `bitnet.js` (the JavaScript loader/glue code) in the root directory.
    *   Logs stdout and stderr from `emcc` to `emcc_stdout.log` and `emcc_stderr.log` respectively.

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

1.  Ensure `bitnet.js` and `bitnet.wasm` are built using `./build.sh`.
2.  Download a BitNet model in GGUF format:
    ```bash
    # Create a models directory
    mkdir -p models
    
    # Download a BitNet model from Hugging Face
    # You can use the huggingface_hub Python package:
    python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='microsoft/BitNet-b1.58-2B-4T-gguf', filename='ggml-model-i2_s.gguf', local_dir='models')"
    ```
3.  Start a simple HTTP server in the `BitNet-wasm` root directory:
    ```bash
    python -m http.server 8000
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

