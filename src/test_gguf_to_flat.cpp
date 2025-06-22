#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

// Declare the gguf_to_flat function from your WASM code
extern "C" int gguf_to_flat(const void* gguf_buf, size_t gguf_size, void* out_flat, size_t out_flat_size);

int main(int argc, char** argv) {
    if (argc != 3) {
        printf("Usage: %s <input.gguf> <output.flat>\n", argv[0]);
        return 1;
    }
    const char* input_path = argv[1];
    const char* output_path = argv[2];

    // Read GGUF file
    FILE* f = fopen(input_path, "rb");
    if (!f) {
        printf("Failed to open input file: %s\n", input_path);
        return 1;
    }
    fseek(f, 0, SEEK_END);
    size_t gguf_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    void* gguf_buf = malloc(gguf_size);
    if (!gguf_buf) {
        printf("Failed to allocate memory for GGUF file\n");
        fclose(f);
        return 1;
    }
    fread(gguf_buf, 1, gguf_size, f);
    fclose(f);
    printf("[test_gguf_to_flat] Read GGUF file of size %zu bytes\n", gguf_size);

    // Allocate large output buffer (2GB max)
    size_t out_flat_size = 2ULL * 1024 * 1024 * 1024;
    void* out_flat = malloc(out_flat_size);
    if (!out_flat) {
        printf("Failed to allocate output buffer\n");
        free(gguf_buf);
        return 1;
    }

    // Call gguf_to_flat
    printf("[test_gguf_to_flat] Calling gguf_to_flat...\n");
    int flat_bytes = gguf_to_flat(gguf_buf, gguf_size, out_flat, out_flat_size);
    printf("gguf_to_flat returned: %d\n", flat_bytes);
    if (flat_bytes < 0) {
        printf("Conversion failed!\n");
        free(gguf_buf);
        free(out_flat);
        return 2;
    }

    // Write output
    FILE* fout = fopen(output_path, "wb");
    if (!fout) {
        printf("Failed to open output file: %s\n", output_path);
        free(gguf_buf);
        free(out_flat);
        return 1;
    }
    fwrite(out_flat, 1, flat_bytes, fout);
    fclose(fout);
    printf("Wrote %d bytes to %s\n", flat_bytes, output_path);

    free(gguf_buf);
    free(out_flat);
    return 0;
}
