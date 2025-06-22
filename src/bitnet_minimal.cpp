#include <stdio.h>

// Minimal BitNet implementation for testing
extern "C" {
    void ggml_bitnet_init(void) {
        printf("[ggml_bitnet_init] BitNet initialization (minimal implementation)\n");
    }
    
    void ggml_bitnet_free(void) {
        printf("[ggml_bitnet_free] BitNet cleanup (minimal implementation)\n");
    }
}