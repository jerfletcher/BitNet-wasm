#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "ggml-bitnet.h"
#include <emscripten.h> // For EMSCRIPTEN_KEEPALIVE

#define GGML_BITNET_MAX_NODES 8192

static bool initialized = false;
static bitnet_tensor_extra * bitnet_tensor_extras = nullptr;
static size_t bitnet_tensor_extras_index = 0;

// Helper function for aligned memory allocation
static void * aligned_malloc(size_t size) {
#if defined(_WIN32)
    return _aligned_malloc(size, 64);
#else
    void * ptr = nullptr;
    posix_memalign(&ptr, 64, size);
    return ptr;
#endif
}

// Helper function for freeing aligned memory
static void aligned_free(void * ptr) {
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

// Helper function for per-tensor quantization
static void per_tensor_quant(int k, void* lut_scales_, void* b_) {
    bitnet_float_type* lut_scales = (bitnet_float_type*)lut_scales_;
    bitnet_float_type* b = (bitnet_float_type*)b_;
    
    // Find the maximum absolute value in the tensor
    bitnet_float_type max_val = 0.0f;
    for (int i = 0; i < k; i++) {
        bitnet_float_type abs_val = fabsf(b[i]);
        if (abs_val > max_val) {
            max_val = abs_val;
        }
    }
    
    // Calculate the scale factor
    *lut_scales = 127.0f / max_val;
}

extern "C" {

EMSCRIPTEN_KEEPALIVE
void ggml_bitnet_init(void) {
    printf("ggml_bitnet_init: initializing BitNet resources\n");
    
    if (initialized) {
        printf("ggml_bitnet_init: already initialized\n");
        return;
    }
    
    // Allocate memory for tensor extras
    bitnet_tensor_extras = (bitnet_tensor_extra *)aligned_malloc(GGML_BITNET_MAX_NODES * sizeof(bitnet_tensor_extra));
    if (!bitnet_tensor_extras) {
        printf("ggml_bitnet_init: failed to allocate memory for tensor extras\n");
        return;
    }
    
    // Initialize tensor extras
    memset(bitnet_tensor_extras, 0, GGML_BITNET_MAX_NODES * sizeof(bitnet_tensor_extra));
    bitnet_tensor_extras_index = 0;
    
    initialized = true;
    printf("ggml_bitnet_init: initialization complete\n");
}

EMSCRIPTEN_KEEPALIVE
void ggml_bitnet_free(void) {
    printf("ggml_bitnet_free: freeing BitNet resources\n");
    
    if (!initialized) {
        printf("ggml_bitnet_free: not initialized\n");
        return;
    }
    
    // Free tensor extras
    if (bitnet_tensor_extras) {
        // Free any allocated resources in tensor extras
        for (size_t i = 0; i < bitnet_tensor_extras_index; i++) {
            if (bitnet_tensor_extras[i].qweights) {
                aligned_free(bitnet_tensor_extras[i].qweights);
            }
            if (bitnet_tensor_extras[i].scales) {
                aligned_free(bitnet_tensor_extras[i].scales);
            }
        }
        
        aligned_free(bitnet_tensor_extras);
        bitnet_tensor_extras = nullptr;
    }
    
    bitnet_tensor_extras_index = 0;
    initialized = false;
    printf("ggml_bitnet_free: resources freed\n");
}

EMSCRIPTEN_KEEPALIVE
void ggml_bitnet_mul_mat_task_compute(
    void * src0, void * scales, void * qlut, 
    void * lut_scales, void * lut_biases, void * dst, 
    int n, int k, int m, int bits) {
    
    printf("ggml_bitnet_mul_mat_task_compute: computing matrix multiplication\n");
    
    if (!initialized) {
        printf("ggml_bitnet_mul_mat_task_compute: BitNet not initialized\n");
        return;
    }
    
    // Check input parameters
    if (!src0 || !scales || !qlut || !lut_scales || !dst) {
        printf("ggml_bitnet_mul_mat_task_compute: null pointer provided\n");
        return;
    }
    
    // Cast pointers to appropriate types
    bitnet_float_type * src0_f = (bitnet_float_type *)src0;
    bitnet_float_type * scales_f = (bitnet_float_type *)scales;
    int8_t * qlut_i8 = (int8_t *)qlut;
    bitnet_float_type * lut_scales_f = (bitnet_float_type *)lut_scales;
    bitnet_float_type * dst_f = (bitnet_float_type *)dst;
    
    // Perform matrix multiplication with BitNet quantization
    // This is a simplified implementation for WASM
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            bitnet_float_type sum = 0.0f;
            
            // Compute dot product with quantized weights
            for (int l = 0; l < k; l++) {
                // Get the quantized weight and input value
                int8_t q_weight = qlut_i8[j * k + l];
                bitnet_float_type input_val = src0_f[i * k + l];
                
                // Accumulate the product
                sum += q_weight * input_val;
            }
            
            // Apply scaling factors
            sum = sum / (*lut_scales_f) * scales_f[j];
            
            // Store the result
            dst_f[i * m + j] = sum;
        }
    }
    
    printf("ggml_bitnet_mul_mat_task_compute: computation complete\n");
}

EMSCRIPTEN_KEEPALIVE
void ggml_bitnet_transform_tensor(
    void * input_ptr, 
    void * output_ptr, 
    int length, 
    int bits) {
    
    printf("ggml_bitnet_transform_tensor: transforming tensor with %d bits\n", bits);
    
    if (!initialized) {
        printf("ggml_bitnet_transform_tensor: BitNet not initialized\n");
        return;
    }
    
    if (!input_ptr || !output_ptr) {
        printf("ggml_bitnet_transform_tensor: null pointer provided\n");
        return;
    }
    
    // Cast pointers to appropriate types
    bitnet_float_type * input_data = (bitnet_float_type *)input_ptr;
    bitnet_float_type * output_data = (bitnet_float_type *)output_ptr;
    
    // Find the maximum absolute value for scaling
    bitnet_float_type max_val = 0.0f;
    for (int i = 0; i < length; i++) {
        bitnet_float_type abs_val = fabsf(input_data[i]);
        if (abs_val > max_val) {
            max_val = abs_val;
        }
    }
    
    // Calculate the scale factor
    bitnet_float_type scale = max_val > 0.0f ? 1.0f / max_val : 1.0f;
    
    // Quantize and dequantize the tensor
    for (int i = 0; i < length; i++) {
        // Quantize to -1, 0, 1 (or more levels based on bits)
        int8_t q_val;
        bitnet_float_type val = input_data[i];
        
        if (bits == 1) {
            // 1-bit quantization: -1, 1
            q_val = (val >= 0) ? 1 : -1;
        } else if (bits == 2) {
            // 2-bit quantization: -1, 0, 1
            if (fabsf(val) < 0.1f * max_val) {
                q_val = 0; // Zero
            } else if (val > 0) {
                q_val = 1; // Positive
            } else {
                q_val = -1; // Negative
            }
        } else {
            // Default to 2-bit
            if (fabsf(val) < 0.1f * max_val) {
                q_val = 0; // Zero
            } else if (val > 0) {
                q_val = 1; // Positive
            } else {
                q_val = -1; // Negative
            }
        }
        
        // Dequantize and store in output
        output_data[i] = q_val * max_val;
    }
    
    printf("ggml_bitnet_transform_tensor: transformation complete\n");
}

} // end extern "C"
