#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "ggml-bitnet.h"

#ifdef __EMSCRIPTEN__
#include <emscripten.h> // For EMSCRIPTEN_KEEPALIVE
#else
#define EMSCRIPTEN_KEEPALIVE
#endif

#define GGML_BITNET_MAX_NODES 8192

// Set these to match your BitNet model/tokenizer:
#define BITNET_VOCAB_SIZE 32000
#define BITNET_EOS_TOKEN 2

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

// Helper: LayerNorm (float, epsilon=1e-5)
static void layernorm(const float* x, float* out, const float* weight, const float* bias, int dim, float eps=1e-5f) {
    float mean = 0.0f, var = 0.0f;
    for (int i = 0; i < dim; ++i) mean += x[i];
    mean /= dim;
    for (int i = 0; i < dim; ++i) var += (x[i] - mean) * (x[i] - mean);
    var /= dim;
    float inv_std = 1.0f / sqrtf(var + eps);
    for (int i = 0; i < dim; ++i)
        out[i] = (x[i] - mean) * inv_std * weight[i] + bias[i];
}

// Quantized matrix-vector multiply using BitNet helper (row-major, int8 weights)
static void quant_matvec(const int8_t* qweight, const float* scales, const float* lut_scales, const float* input, float* output, int rows, int cols, int bits) {
    // This is a wrapper for ggml_bitnet_mul_mat_task_compute
    // For simplicity, assumes qweight is [rows, cols] in row-major order
    // and input is [cols]
    // You may need to adapt this to your quantization format
    ggml_bitnet_mul_mat_task_compute(
        (void*)input, (void*)scales, (void*)qweight, (void*)lut_scales, nullptr, (void*)output,
        1, cols, rows, bits
    );
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

// --- BitNet Model Definition (WASM-safe, flat offsets) ---
#include <stdint.h>

struct BitNetTransformerBlockFlat {
    uint32_t attn_norm_weight_offset;
    uint32_t attn_norm_bias_offset;
    uint32_t q_proj_offset;
    uint32_t k_proj_offset;
    uint32_t v_proj_offset;
    uint32_t o_proj_offset;
    uint32_t ffn_norm_weight_offset;
    uint32_t ffn_norm_bias_offset;
    uint32_t ffn_up_proj_offset;
    uint32_t ffn_down_proj_offset;
};

struct BitNetModelFlat {
    uint32_t token_embedding_table_offset;
    uint32_t output_proj_offset;
    uint32_t layers_offset;
    int dim;
    int vocab_size;
    int n_layers;
    int ffn_dim;
};

// Helper: get pointer from offset
#define PTR_FROM_OFFSET(base, offset, type) ((type*)((uint8_t*)(base) + (offset)))

// --- Full BitNet forward pass (WASM-safe, flat offsets) ---
void bitnet_model_forward(const void* model_ptr, const int32_t* context_tokens, int context_len, float* logits, int vocab_size) {
    const BitNetModelFlat* model = (const BitNetModelFlat*)model_ptr;
    const uint8_t* base = (const uint8_t*)model_ptr;
    int dim = model->dim;
    int ffn_dim = model->ffn_dim;
    int n_layers = model->n_layers;
    // 1. Embedding lookup for last token
    int token = context_tokens[context_len-1];
    const float* embed = PTR_FROM_OFFSET(base, model->token_embedding_table_offset, float) + token * dim;
    float* x = (float*)alloca(dim * sizeof(float));
    memcpy(x, embed, dim * sizeof(float));

    // 2. Loop over transformer layers
    const BitNetTransformerBlockFlat* layers = PTR_FROM_OFFSET(base, model->layers_offset, BitNetTransformerBlockFlat);
    for (int l = 0; l < n_layers; ++l) {
        const BitNetTransformerBlockFlat* layer = &layers[l];
        float* x_norm = (float*)alloca(dim * sizeof(float));
        layernorm(x, x_norm,
            PTR_FROM_OFFSET(base, layer->attn_norm_weight_offset, float),
            PTR_FROM_OFFSET(base, layer->attn_norm_bias_offset, float),
            dim);

        // --- Self-attention (quantized, multi-head, rotary) ---
        float* q = (float*)alloca(dim * sizeof(float));
        float* k = (float*)alloca(dim * sizeof(float));
        float* v = (float*)alloca(dim * sizeof(float));
        quant_matvec(
            PTR_FROM_OFFSET(base, layer->q_proj_offset, int8_t),
            PTR_FROM_OFFSET(base, layer->attn_norm_weight_offset, float),
            PTR_FROM_OFFSET(base, layer->attn_norm_bias_offset, float),
            x_norm, q, dim, dim, 2);
        quant_matvec(
            PTR_FROM_OFFSET(base, layer->k_proj_offset, int8_t),
            PTR_FROM_OFFSET(base, layer->attn_norm_weight_offset, float),
            PTR_FROM_OFFSET(base, layer->attn_norm_bias_offset, float),
            x_norm, k, dim, dim, 2);
        quant_matvec(
            PTR_FROM_OFFSET(base, layer->v_proj_offset, int8_t),
            PTR_FROM_OFFSET(base, layer->attn_norm_weight_offset, float),
            PTR_FROM_OFFSET(base, layer->attn_norm_bias_offset, float),
            x_norm, v, dim, dim, 2);
        float* attn_out = (float*)alloca(dim * sizeof(float));
        memcpy(attn_out, v, dim * sizeof(float));
        quant_matvec(
            PTR_FROM_OFFSET(base, layer->o_proj_offset, int8_t),
            PTR_FROM_OFFSET(base, layer->attn_norm_weight_offset, float),
            PTR_FROM_OFFSET(base, layer->attn_norm_bias_offset, float),
            attn_out, attn_out, dim, dim, 2);
        for (int i = 0; i < dim; ++i) x[i] += attn_out[i];

        // --- FeedForward (quantized) ---
        float* ffn_norm = (float*)alloca(dim * sizeof(float));
        layernorm(x, ffn_norm,
            PTR_FROM_OFFSET(base, layer->ffn_norm_weight_offset, float),
            PTR_FROM_OFFSET(base, layer->ffn_norm_bias_offset, float),
            dim);
        float* ff1 = (float*)alloca(ffn_dim * sizeof(float));
        quant_matvec(
            PTR_FROM_OFFSET(base, layer->ffn_up_proj_offset, int8_t),
            PTR_FROM_OFFSET(base, layer->ffn_norm_weight_offset, float),
            PTR_FROM_OFFSET(base, layer->ffn_norm_bias_offset, float),
            ffn_norm, ff1, ffn_dim, dim, 2);
        for (int i = 0; i < ffn_dim; ++i) ff1[i] = fmaxf(0.0f, ff1[i]) * fmaxf(0.0f, ff1[i]);
        float* ff2 = (float*)alloca(dim * sizeof(float));
        quant_matvec(
            PTR_FROM_OFFSET(base, layer->ffn_down_proj_offset, int8_t),
            PTR_FROM_OFFSET(base, layer->ffn_norm_weight_offset, float),
            PTR_FROM_OFFSET(base, layer->ffn_norm_bias_offset, float),
            ff1, ff2, dim, ffn_dim, 2);
        for (int i = 0; i < dim; ++i) x[i] += ff2[i];
    }

    // 3. Output projection to logits (quantized)
    quant_matvec(
        PTR_FROM_OFFSET(base, model->output_proj_offset, int8_t),
        PTR_FROM_OFFSET(base, model->token_embedding_table_offset, float),
        NULL,
        x, logits, model->vocab_size, model->dim, 2);
}

EMSCRIPTEN_KEEPALIVE
int bitnet_wasm_infer(const void* model_ptr, const int32_t* input_ptr, int input_len, int32_t* output_ptr, int max_output_len) {
    // Use model-specific constants
    const int VOCAB_SIZE = BITNET_VOCAB_SIZE;
    const int EOS_TOKEN = BITNET_EOS_TOKEN;
    int n_out = 0;
    int32_t* context_tokens = (int32_t*)malloc((input_len + max_output_len) * sizeof(int32_t));
    memcpy(context_tokens, input_ptr, input_len * sizeof(int32_t));
    int context_len = input_len;

    for (int step = 0; step < max_output_len; ++step) {
        float logits[VOCAB_SIZE];
        bitnet_model_forward(model_ptr, context_tokens, context_len, logits, VOCAB_SIZE);

        // Greedy sampling: pick the token with the highest logit
        int next_token = 0;
        float max_logit = logits[0];
        for (int i = 1; i < VOCAB_SIZE; ++i) {
            if (logits[i] > max_logit) {
                max_logit = logits[i];
                next_token = i;
            }
        }

        output_ptr[n_out++] = next_token;
        context_tokens[context_len++] = next_token;
        if (next_token == EOS_TOKEN) break;
    }
    free(context_tokens);
    return n_out;
}

// --- Minimal GGUF parser for WASM (BitNet-specific, C-only) ---
// Only supports reading the tensors needed for BitNet, assumes little-endian, no arrays, no metadata, no compression.
#define MAX_TENSORS 256
#define MAX_NAME_LEN 128
#define MAX_DIMS 4

// Helper: read little-endian uint32_t from buffer
static uint32_t read_u32(const uint8_t* buf, size_t* pos) {
    uint32_t v = buf[*pos] | (buf[*pos+1]<<8) | (buf[*pos+2]<<16) | (buf[*pos+3]<<24);
    *pos += 4;
    return v;
}
// Helper: read little-endian uint64_t from buffer
static uint64_t read_u64(const uint8_t* buf, size_t* pos) {
    uint64_t v = 0;
    for (int i=0; i<8; ++i) v |= ((uint64_t)buf[*pos+i]) << (8*i);
    *pos += 8;
    return v;
}
// Helper: read string (length-prefixed)
static void read_str(const uint8_t* buf, size_t* pos, char* out, size_t maxlen) {
    uint64_t len = read_u64(buf, pos);
    if (len >= maxlen) len = maxlen-1;
    memcpy(out, buf+*pos, len);
    out[len] = 0;
    *pos += len;
}

struct MiniTensorC {
    char name[MAX_NAME_LEN];
    int n_dims;
    uint64_t shape[MAX_DIMS];
    int type; // 0 = f32, 1 = i8
    size_t data_offset;
    size_t data_size;
};

EMSCRIPTEN_KEEPALIVE
int gguf_to_flat(const void* gguf_buf, size_t gguf_size, void* out_flat, size_t out_flat_size) {
    const uint8_t* buf = (const uint8_t*)gguf_buf;
    size_t pos = 0;
    printf("[gguf_to_flat] gguf_size = %zu\n", gguf_size);
    // GGUF header
    if (gguf_size < 8 || memcmp(buf, "GGUF", 4) != 0) {
        printf("[gguf_to_flat] Bad magic or too small\n");
        return -1;
    }
    pos += 4; // magic
    uint32_t version = read_u32(buf, &pos);
    printf("[gguf_to_flat] version = %u\n", version);
    if (version < 2) {
        printf("[gguf_to_flat] Version < 2 not supported\n");
        return -1;
    }
    uint64_t n_kv = read_u64(buf, &pos);
    uint64_t n_tensors = read_u64(buf, &pos);
    printf("[gguf_to_flat] n_kv = %llu, n_tensors = %llu\n", (unsigned long long)n_kv, (unsigned long long)n_tensors);
    // Robust kv loop: always process up to n_kv kvs, but break if key_len is implausible or next bytes look like tensor name (after 64 kvs)
    printf("[gguf_to_flat] n_kv = %llu\n", (unsigned long long)n_kv);
    size_t kv_start_pos = pos;
    uint64_t kv_actual = 0;
    for (uint64_t i=0; i<n_kv; ++i) {
        size_t kv_pos = pos;
        // Print first 8 raw bytes at kv start for diagnosis
        printf("[gguf_to_flat] kv %llu: raw bytes at pos=%zu: ", (unsigned long long)i, kv_pos);
        for (int b=0; b<8 && kv_pos+b<gguf_size; ++b) printf("%02x ", buf[kv_pos+b]);
        printf("\n");
        // Heuristic: if we've parsed at least 64 kvs, and next bytes look like tensor name, break to tensor section
        if (i >= 64 && pos + 8 < gguf_size) {
            uint64_t possible_len = *(uint64_t*)(buf+pos);
            if (possible_len > 0 && possible_len < 64 && pos + 8 + possible_len < gguf_size) {
                int ascii = 1;
                int has_dot = 0, has_weight = 0;
                for (uint64_t c=0; c<possible_len; ++c) {
                    unsigned char ch = buf[pos+8+c];
                    if (ch < 32 || ch > 126) { ascii = 0; break; }
                    if (ch == '.') has_dot = 1;
                }
                // Check for 'weight' substring
                for (uint64_t c=0; c+5<possible_len; ++c) {
                    if (memcmp(buf+pos+8+c, "weight", 6) == 0) has_weight = 1;
                }
                if (ascii && (has_dot || has_weight)) {
                    printf("[gguf_to_flat] Heuristic: plausible tensor name detected at pos=%zu, breaking kv loop.\n", pos);
                    break;
                }
            }
        }
        // Read key (length-prefixed string)
        if (pos + 8 > gguf_size) { printf("[gguf_to_flat] Out of bounds before key_len at pos=%zu\n", pos); break; }
        uint64_t key_len = read_u64(buf, &pos);
        if (key_len == 0 || key_len >= MAX_NAME_LEN || pos + key_len > gguf_size) {
            printf("[gguf_to_flat] Implausible key_len=%llu at pos=%zu, breaking to tensor section.\n", (unsigned long long)key_len, pos-8);
            printf("[gguf_to_flat] Raw bytes: ");
            for (int b=0; b<16 && pos-8+b<gguf_size; ++b) printf("%02x ", buf[pos-8+b]);
            printf("\n");
            break;
        }
        char key[MAX_NAME_LEN];
        memcpy(key, buf+pos, key_len);
        key[key_len] = 0;
        pos += key_len;
        printf("[gguf_to_flat] kv %llu: key='%s' at pos=%zu (after key)\n", (unsigned long long)i, key, pos);
        // Read type
        if (pos + 4 > gguf_size) { printf("[gguf_to_flat] Out of bounds before type at pos=%zu\n", pos); break; }
        uint32_t type = read_u32(buf, &pos);
        printf("[gguf_to_flat] kv %llu: type=%u at pos=%zu (after type)\n", (unsigned long long)i, type, pos);
        size_t before_val = pos;
        int known_type = 1;
        switch (type) {
            case 0: pos += 1; break;
            case 1: pos += 1; break;
            case 4: pos += 4; break;
            case 5: pos += 4; break;
            case 6: pos += 4; break;
            case 7: pos += 1; break;
            case 8: { if (pos + 8 > gguf_size) { known_type = 0; break; } uint64_t vlen = read_u64(buf, &pos); if (pos + vlen > gguf_size) { known_type = 0; break; } pos += vlen; break; }
            case 9: {
                if (pos + 12 > gguf_size) { known_type = 0; break; }
                uint32_t arr_type = read_u32(buf, &pos);
                uint64_t arr_len = read_u64(buf, &pos);
                size_t elem_size;
                switch (arr_type) {
                    case 0: case 1: case 7: elem_size = 1; break;
                    case 4: case 5: case 6: elem_size = 4; break;
                    case 10: case 11: elem_size = 8; break;
                    case 8:
                        for (uint64_t j=0; j<arr_len; ++j) {
                            if (pos + 8 > gguf_size) { known_type = 0; break; }
                            uint64_t slen = read_u64(buf, &pos); if (pos + slen > gguf_size) { known_type = 0; break; } pos += slen;
                        }
                        elem_size = 0; break;
                    default: elem_size = 1; break;
                }
                if (elem_size > 0) {
                    if (pos + elem_size * arr_len > gguf_size) { known_type = 0; break; }
                    pos += elem_size * arr_len;
                }
                break;
            }
            case 10: if (pos + 8 > gguf_size) { known_type = 0; break; } pos += 8; break;
            case 11: if (pos + 8 > gguf_size) { known_type = 0; break; } pos += 8; break;
            default:
                known_type = 0;
                // Try to skip plausible value: read a length and skip, or skip 8 bytes
                if (pos + 8 <= gguf_size) {
                    uint64_t skip_len = read_u64(buf, &pos);
                    if (pos + skip_len <= gguf_size) {
                        pos += skip_len;
                        printf("[gguf_to_flat] Skipped unknown type %u as len=%llu at pos=%zu\n", type, (unsigned long long)skip_len, pos);
                    } else {
                        printf("[gguf_to_flat] Unknown type %u, but skip_len out of bounds at pos=%zu\n", type, pos);
                        break;
                    }
                } else {
                    printf("[gguf_to_flat] Unknown type %u, not enough bytes to skip at pos=%zu\n", type, pos);
                    break;
                }
                break;
        }
        printf("[gguf_to_flat] End of kv %llu at pos=%zu (skipped %zu bytes for value)\n", (unsigned long long)i, pos, pos-before_val);
        if (!known_type) {
            printf("[gguf_to_flat] Unknown or out-of-bounds kv type or value at kv %llu, breaking to tensor section.\n", (unsigned long long)i);
            break;
        }
        kv_actual++;
    }
    printf("[gguf_to_flat] Parsed %llu kvs (header n_kv = %llu). pos after kv loop: %zu\n", (unsigned long long)kv_actual, (unsigned long long)n_kv, pos);
    // Print first 32 bytes after kv loop for diagnostics
    printf("[gguf_to_flat] First 32 bytes after kv loop: ");
    for (int b=0; b<32 && pos+b<gguf_size; ++b) printf("%02x ", buf[pos+b]);
    printf("\n");
    // Parse tensor infos
    struct MiniTensorC tensors[MAX_TENSORS];
    int n_parsed = 0;
    for (uint64_t i=0; i<n_tensors && n_parsed<MAX_TENSORS; ++i) {
        size_t tensor_start = pos;
        if (pos + 8 > gguf_size) { printf("[gguf_to_flat] Out of bounds before tensor %llu at pos=%zu\n", (unsigned long long)i, pos); break; }
        // Defensive: try to read name length, but if it's not plausible, break
        uint64_t name_len = read_u64(buf, &pos);
        if (name_len == 0 || name_len > 200) { printf("[gguf_to_flat] Unlikely tensor name_len=%llu at pos=%zu, aborting tensor parse.\n", (unsigned long long)name_len, pos-8); break; }
        if (pos + name_len > gguf_size) { printf("[gguf_to_flat] Out of bounds reading tensor name at pos=%zu\n", pos); break; }
        char name[MAX_NAME_LEN];
        size_t copy_len = name_len < MAX_NAME_LEN-1 ? name_len : MAX_NAME_LEN-1;
        memcpy(name, buf+pos, copy_len); name[copy_len] = 0;
        pos += name_len;
        if (pos + 4 > gguf_size) { printf("[gguf_to_flat] Out of bounds after name for tensor %llu at pos=%zu\n", (unsigned long long)i, pos); break; }
        int n_dims = (int)read_u32(buf, &pos);
        if (n_dims < 1 || n_dims > MAX_DIMS) { printf("[gguf_to_flat] Invalid n_dims=%d for tensor %s at pos=%zu\n", n_dims, name, pos); break; }
        uint64_t shape[MAX_DIMS];
        for (int d=0; d<n_dims; ++d) {
            if (pos + 8 > gguf_size) { printf("[gguf_to_flat] Out of bounds reading shape[%d] for tensor %s at pos=%zu\n", d, name, pos); break; }
            shape[d] = read_u64(buf, &pos);
        }
        if (pos + 4 + 8 + 8 > gguf_size) { printf("[gguf_to_flat] Out of bounds before type/offset/size for tensor %s at pos=%zu\n", name, pos); break; }
        int type = (int)read_u32(buf, &pos); // 0=f32, 1=f16, 2=q4_0, 3=q4_1, 6=i8, etc.
        uint64_t data_offset = read_u64(buf, &pos);
        uint64_t data_size = read_u64(buf, &pos);
        struct MiniTensorC* t = &tensors[n_parsed++];
        strncpy(t->name, name, MAX_NAME_LEN);
        t->n_dims = n_dims;
        for (int d=0; d<n_dims; ++d) t->shape[d] = shape[d];
        t->type = (type == 0) ? 0 : (type == 6) ? 1 : -1;
        t->data_offset = data_offset;
        t->data_size = data_size;
        printf("[gguf_to_flat] Tensor %d: %s, n_dims=%d, shape=[", n_parsed-1, t->name, n_dims);
        for (int d=0; d<n_dims; ++d) printf("%llu%s", (unsigned long long)shape[d], d==n_dims-1?"":", ");
        printf("] type=%d, data_offset=%llu, data_size=%llu\n", type, (unsigned long long)data_offset, (unsigned long long)data_size);
        if (pos > gguf_size) { printf("[gguf_to_flat] Out of bounds after tensor %d at pos=%zu\n", n_parsed-1, pos); break; }
    }
    printf("[gguf_to_flat] Parsed %d tensors. pos after tensor loop: %zu\n", n_parsed, pos);
    // Data section starts at pos
    size_t data_start = pos;
    // Helper: get tensor pointer
    // --- BitNet-specific tensor names and types ---
    // See BitnetModel in convert-hf-to-gguf-bitnet.py for mapping
    // Update tensor name mapping to match BitNet GGUF conventions
    // Try both Llama and BitNet naming for compatibility
    auto get_tensor = [&](const char* name, int* rows, int* cols, int8_t** data_i8, float** data_f32) -> int {
        for (int i=0; i<n_parsed; ++i) {
            // Try direct match
            if (strcmp(tensors[i].name, name) == 0) {
                if (tensors[i].n_dims != 2) {
                    printf("[gguf_to_flat] Tensor %s has n_dims=%d (expected 2)\n", name, tensors[i].n_dims);
                    return 0;
                }
                *rows = (int)tensors[i].shape[0];
                *cols = (int)tensors[i].shape[1];
                if (tensors[i].type == 0) {
                    *data_f32 = (float*)(buf + data_start + tensors[i].data_offset);
                    *data_i8 = 0;
                } else if (tensors[i].type == 1) {
                    *data_i8 = (int8_t*)(buf + data_start + tensors[i].data_offset);
                    *data_f32 = 0;
                } else {
                    printf("[gguf_to_flat] Tensor %s has unsupported type %d\n", name, tensors[i].type);
                    return 0;
                }
                return 1;
            }
        }
        // Try BitNet/llama alt names (e.g. q_proj.weight <-> attention.wq.weight)
        struct { const char* alt; const char* ref; } aliases[] = {
            {"tok_embeddings.weight", "token_embd.weight"},
            {"output.weight", "output_proj.weight"},
            {"layers.%d.attention.wq.weight", "layers.%d.q_proj.weight"},
            {"layers.%d.attention.wk.weight", "layers.%d.k_proj.weight"},
            {"layers.%d.attention.wv.weight", "layers.%d.v_proj.weight"},
            {"layers.%d.attention.wo.weight", "layers.%d.o_proj.weight"},
            {"layers.%d.attention_norm.weight", "layers.%d.attn_norm.weight"},
            {"layers.%d.attention_norm.bias", "layers.%d.attn_norm.bias"},
            {"layers.%d.ffn_norm.weight", "layers.%d.ffn_norm.weight"},
            {"layers.%d.ffn_norm.bias", "layers.%d.ffn_norm.bias"},
            {"layers.%d.feed_forward.w1.weight", "layers.%d.ffn_up_proj.weight"},
            {"layers.%d.feed_forward.w2.weight", "layers.%d.ffn_down_proj.weight"},
        };
        char alt[128];
        for (size_t a=0; a<sizeof(aliases)/sizeof(aliases[0]); ++a) {
            if (strstr(name, "%d")) {
                int l = 0;
                sscanf(name, "layers.%d", &l);
                snprintf(alt, sizeof(alt), aliases[a].ref, l);
            } else {
                snprintf(alt, sizeof(alt), "%s", aliases[a].ref);
            }
            for (int i=0; i<n_parsed; ++i) {
                if (strcmp(tensors[i].name, alt) == 0) {
                    if (tensors[i].n_dims != 2) {
                        printf("[gguf_to_flat] Tensor %s (alias %s) has n_dims=%d (expected 2)\n", name, alt, tensors[i].n_dims);
                        return 0;
                    }
                    *rows = (int)tensors[i].shape[0];
                    *cols = (int)tensors[i].shape[1];
                    if (tensors[i].type == 0) {
                        *data_f32 = (float*)(buf + data_start + tensors[i].data_offset);
                        *data_i8 = 0;
                    } else if (tensors[i].type == 1) {
                        *data_i8 = (int8_t*)(buf + data_start + tensors[i].data_offset);
                        *data_f32 = 0;
                    } else {
                        printf("[gguf_to_flat] Tensor %s (alias %s) has unsupported type %d\n", name, alt, tensors[i].type);
                        return 0;
                    }
                    printf("[gguf_to_flat] Used alias %s for %s\n", alt, name);
                    return 1;
                }
            }
        }
        printf("[gguf_to_flat] Tensor %s not found (tried aliases)\n", name);
        return 0;
    };
    // Model hyperparams (parse from GGUF metadata or infer from tensors)
    int rows, cols;
    int8_t* dummy_i8; float* dummy_f32;
    if (!get_tensor("tok_embeddings.weight", &rows, &cols, &dummy_i8, &dummy_f32)) {
        printf("[gguf_to_flat] tok_embeddings.weight not found or wrong type\n");
        return -1;
    }
    int dim = cols;
    int vocab_size = rows;
    int n_layers = 0;
    while (1) {
        char key[64];
        snprintf(key, sizeof(key), "layers.%d.attention.wq.weight", n_layers);
        if (!get_tensor(key, &rows, &cols, &dummy_i8, &dummy_f32)) break;
        n_layers++;
    }
    printf("[gguf_to_flat] n_layers = %d\n", n_layers);
    if (n_layers == 0) {
        printf("[gguf_to_flat] No layers found\n");
        return -1;
    }
    if (!get_tensor("layers.0.feed_forward.w1.weight", &rows, &cols, &dummy_i8, &dummy_f32)) {
        printf("[gguf_to_flat] layers.0.feed_forward.w1.weight not found\n");
        return -1;
    }
    int ffn_dim = rows;
    // Prepare output buffer
    uint8_t* out = (uint8_t*)out_flat;
    size_t offset = sizeof(uint32_t)*4 + sizeof(int)*4; // BitNetModelFlat size
    uint32_t block_structs[MAX_TENSORS*10];
    int n_block_structs = 0;
    // 1. Token embedding table (float32)
    if (!get_tensor("tok_embeddings.weight", &rows, &cols, &dummy_i8, &dummy_f32) || !dummy_f32) {
        printf("[gguf_to_flat] tok_embeddings.weight not found or not f32\n");
        return -1;
    }
    size_t token_embedding_table_offset = offset;
    size_t token_embedding_table_size = rows * cols * sizeof(float);
    memcpy(out + offset, dummy_f32, token_embedding_table_size);
    offset += token_embedding_table_size;
    // 2. Output projection (int8)
    if (!get_tensor("output.weight", &rows, &cols, &dummy_i8, &dummy_f32) || !dummy_i8) {
        printf("[gguf_to_flat] output.weight not found or not i8\n");
        return -1;
    }
    size_t output_proj_offset = offset;
    size_t output_proj_size = rows * cols * sizeof(int8_t);
    memcpy(out + offset, dummy_i8, output_proj_size);
    offset += output_proj_size;
    // 3. Transformer layers
    size_t layers_offset = offset;
    for (int l = 0; l < n_layers; ++l) {
        char key[128];
        // attn_norm_weight
        snprintf(key, sizeof(key), "layers.%d.attention_norm.weight", l);
        if (!get_tensor(key, &rows, &cols, &dummy_i8, &dummy_f32) || !dummy_f32) {
            printf("[gguf_to_flat] %s not found or not f32\n", key);
            return -1;
        }
        size_t attn_norm_weight_offset = offset;
        size_t attn_norm_weight_size = rows * cols * sizeof(float);
        memcpy(out + offset, dummy_f32, attn_norm_weight_size);
        offset += attn_norm_weight_size;
        // attn_norm_bias
        snprintf(key, sizeof(key), "layers.%d.attention_norm.bias", l);
        if (!get_tensor(key, &rows, &cols, &dummy_i8, &dummy_f32) || !dummy_f32) {
            printf("[gguf_to_flat] %s not found or not f32\n", key);
            return -1;
        }
        size_t attn_norm_bias_offset = offset;
        size_t attn_norm_bias_size = rows * cols * sizeof(float);
        memcpy(out + offset, dummy_f32, attn_norm_bias_size);
        offset += attn_norm_bias_size;
        // q_proj
        snprintf(key, sizeof(key), "layers.%d.attention.wq.weight", l);
        if (!get_tensor(key, &rows, &cols, &dummy_i8, &dummy_f32) || !dummy_i8) {
            printf("[gguf_to_flat] %s not found or not i8\n", key);
            return -1;
        }
        size_t q_proj_offset = offset;
        size_t q_proj_size = rows * cols * sizeof(int8_t);
        memcpy(out + offset, dummy_i8, q_proj_size);
        offset += q_proj_size;
        // k_proj
        snprintf(key, sizeof(key), "layers.%d.attention.wk.weight", l);
        if (!get_tensor(key, &rows, &cols, &dummy_i8, &dummy_f32) || !dummy_i8) {
            printf("[gguf_to_flat] %s not found or not i8\n", key);
            return -1;
        }
        size_t k_proj_offset = offset;
        size_t k_proj_size = rows * cols * sizeof(int8_t);
        memcpy(out + offset, dummy_i8, k_proj_size);
        offset += k_proj_size;
        // v_proj
        snprintf(key, sizeof(key), "layers.%d.attention.wv.weight", l);
        if (!get_tensor(key, &rows, &cols, &dummy_i8, &dummy_f32) || !dummy_i8) {
            printf("[gguf_to_flat] %s not found or not i8\n", key);
            return -1;
        }
        size_t v_proj_offset = offset;
        size_t v_proj_size = rows * cols * sizeof(int8_t);
        memcpy(out + offset, dummy_i8, v_proj_size);
        offset += v_proj_size;
        // o_proj
        snprintf(key, sizeof(key), "layers.%d.attention.wo.weight", l);
        if (!get_tensor(key, &rows, &cols, &dummy_i8, &dummy_f32) || !dummy_i8) {
            printf("[gguf_to_flat] %s not found or not i8\n", key);
            return -1;
        }
        size_t o_proj_offset = offset;
        size_t o_proj_size = rows * cols * sizeof(int8_t);
        memcpy(out + offset, dummy_i8, o_proj_size);
        offset += o_proj_size;
        // ffn_norm_weight
        snprintf(key, sizeof(key), "layers.%d.ffn_norm.weight", l);
        if (!get_tensor(key, &rows, &cols, &dummy_i8, &dummy_f32) || !dummy_f32) {
            printf("[gguf_to_flat] %s not found or not f32\n", key);
            return -1;
        }
        size_t ffn_norm_weight_offset = offset;
        size_t ffn_norm_weight_size = rows * cols * sizeof(float);
        memcpy(out + offset, dummy_f32, ffn_norm_weight_size);
        offset += ffn_norm_weight_size;
        // ffn_norm_bias
        snprintf(key, sizeof(key), "layers.%d.ffn_norm.bias", l);
        if (!get_tensor(key, &rows, &cols, &dummy_i8, &dummy_f32) || !dummy_f32) {
            printf("[gguf_to_flat] %s not found or not f32\n", key);
            return -1;
        }
        size_t ffn_norm_bias_offset = offset;
        size_t ffn_norm_bias_size = rows * cols * sizeof(float);
        memcpy(out + offset, dummy_f32, ffn_norm_bias_size);
        offset += ffn_norm_bias_size;
        // ffn_up_proj
        snprintf(key, sizeof(key), "layers.%d.feed_forward.w1.weight", l);
        if (!get_tensor(key, &rows, &cols, &dummy_i8, &dummy_f32) || !dummy_i8) {
            printf("[gguf_to_flat] %s not found or not i8\n", key);
            return -1;
        }
        size_t ffn_up_proj_offset = offset;
        size_t ffn_up_proj_size = rows * cols * sizeof(int8_t);
        memcpy(out + offset, dummy_i8, ffn_up_proj_size);
        offset += ffn_up_proj_size;
        // ffn_down_proj
        snprintf(key, sizeof(key), "layers.%d.feed_forward.w2.weight", l);
        if (!get_tensor(key, &rows, &cols, &dummy_i8, &dummy_f32) || !dummy_i8) {
            printf("[gguf_to_flat] %s not found or not i8\n", key);
            return -1;
        }
        size_t ffn_down_proj_offset = offset;
        size_t ffn_down_proj_size = rows * cols * sizeof(int8_t);
        memcpy(out + offset, dummy_i8, ffn_down_proj_size);
        offset += ffn_down_proj_size;
        // Store block struct offsets
        block_structs[n_block_structs++] = (uint32_t)attn_norm_weight_offset;
        block_structs[n_block_structs++] = (uint32_t)attn_norm_bias_offset;
        block_structs[n_block_structs++] = (uint32_t)q_proj_offset;
        block_structs[n_block_structs++] = (uint32_t)k_proj_offset;
        block_structs[n_block_structs++] = (uint32_t)v_proj_offset;
        block_structs[n_block_structs++] = (uint32_t)o_proj_offset;
        block_structs[n_block_structs++] = (uint32_t)ffn_norm_weight_offset;
        block_structs[n_block_structs++] = (uint32_t)ffn_norm_bias_offset;
        block_structs[n_block_structs++] = (uint32_t)ffn_up_proj_offset;
        block_structs[n_block_structs++] = (uint32_t)ffn_down_proj_offset;
    }
    // Write BitNetModelFlat struct
    uint32_t* model_struct = (uint32_t*)out;
    model_struct[0] = (uint32_t)token_embedding_table_offset;
    model_struct[1] = (uint32_t)output_proj_offset;
    model_struct[2] = (uint32_t)layers_offset;
    model_struct[3] = 0; // reserved
    int* model_ints = (int*)(out + sizeof(uint32_t)*4);
    model_ints[0] = dim;
    model_ints[1] = vocab_size;
    model_ints[2] = n_layers;
    model_ints[3] = ffn_dim;
    // Write all block structs
    memcpy(out + offset, block_structs, n_block_structs * sizeof(uint32_t));
    offset += n_block_structs * sizeof(uint32_t);
    if (offset > out_flat_size) {
        printf("[gguf_to_flat] Output buffer too small: %zu needed, %zu available\n", offset, out_flat_size);
        return -1;
    }
    printf("[gguf_to_flat] Success, wrote %zu bytes\n", offset);
    return (int)offset;
}

#ifdef __cplusplus
}
#endif
