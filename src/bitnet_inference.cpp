#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <memory>
#include <map>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#include <emscripten/bind.h>
#endif

// GGUF file format structures
struct gguf_header {
    char magic[4];      // "GGUF"
    uint32_t version;
    uint64_t n_tensors;
    uint64_t n_kv;
};

struct gguf_kv_pair {
    std::string key;
    uint32_t value_type;
    std::vector<uint8_t> value_data;
};

struct gguf_tensor_info {
    std::string name;
    uint32_t n_dimensions;
    std::vector<uint64_t> dimensions;
    uint32_t type;
    uint64_t offset;
};

// BitNet model structure
struct BitNetModel {
    std::vector<gguf_kv_pair> metadata;
    std::vector<gguf_tensor_info> tensors;
    std::vector<uint8_t> tensor_data;
    bool loaded = false;
    
    // Model parameters
    uint32_t vocab_size = 0;
    uint32_t n_embd = 0;
    uint32_t n_head = 0;
    uint32_t n_layer = 0;
    uint32_t n_ctx = 0;
};

// Global model instance
static BitNetModel g_model;

// GGUF value types
enum gguf_type {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
};

// Helper function to read data from buffer
template<typename T>
T read_value(const uint8_t*& data, size_t& remaining) {
    if (remaining < sizeof(T)) {
        throw std::runtime_error("Insufficient data to read value");
    }
    T value;
    std::memcpy(&value, data, sizeof(T));
    data += sizeof(T);
    remaining -= sizeof(T);
    return value;
}

// Read string from GGUF
std::string read_string(const uint8_t*& data, size_t& remaining) {
    uint64_t len = read_value<uint64_t>(data, remaining);
    if (remaining < len) {
        throw std::runtime_error("Insufficient data to read string");
    }
    std::string str(reinterpret_cast<const char*>(data), len);
    data += len;
    remaining -= len;
    return str;
}

// Parse GGUF file
bool parse_gguf_file(const uint8_t* file_data, size_t file_size, BitNetModel& model) {
    try {
        const uint8_t* data = file_data;
        size_t remaining = file_size;
        
        // Read header
        gguf_header header;
        if (remaining < sizeof(header)) {
            std::cerr << "File too small for GGUF header" << std::endl;
            return false;
        }
        
        std::memcpy(&header, data, sizeof(header));
        data += sizeof(header);
        remaining -= sizeof(header);
        
        // Check magic
        if (std::memcmp(header.magic, "GGUF", 4) != 0) {
            std::cerr << "Invalid GGUF magic" << std::endl;
            return false;
        }
        
        std::cout << "GGUF version: " << header.version << std::endl;
        std::cout << "Number of tensors: " << header.n_tensors << std::endl;
        std::cout << "Number of KV pairs: " << header.n_kv << std::endl;
        
        // Read KV pairs
        model.metadata.clear();
        for (uint64_t i = 0; i < header.n_kv; i++) {
            gguf_kv_pair kv;
            kv.key = read_string(data, remaining);
            kv.value_type = read_value<uint32_t>(data, remaining);
            
            // Read value based on type
            switch (kv.value_type) {
                case GGUF_TYPE_UINT32: {
                    uint32_t val = read_value<uint32_t>(data, remaining);
                    kv.value_data.resize(sizeof(val));
                    std::memcpy(kv.value_data.data(), &val, sizeof(val));
                    
                    // Extract important model parameters
                    if (kv.key == "llama.embedding_length" || kv.key == "llama.n_embd") {
                        model.n_embd = val;
                    } else if (kv.key == "llama.attention.head_count" || kv.key == "llama.n_head") {
                        model.n_head = val;
                    } else if (kv.key == "llama.block_count" || kv.key == "llama.n_layer") {
                        model.n_layer = val;
                    } else if (kv.key == "llama.context_length" || kv.key == "llama.n_ctx") {
                        model.n_ctx = val;
                    } else if (kv.key == "tokenizer.ggml.tokens" || kv.key == "llama.vocab_size") {
                        model.vocab_size = val;
                    }
                    break;
                }
                case GGUF_TYPE_STRING: {
                    std::string val = read_string(data, remaining);
                    kv.value_data.resize(val.size());
                    std::memcpy(kv.value_data.data(), val.data(), val.size());
                    break;
                }
                case GGUF_TYPE_FLOAT32: {
                    float val = read_value<float>(data, remaining);
                    kv.value_data.resize(sizeof(val));
                    std::memcpy(kv.value_data.data(), &val, sizeof(val));
                    break;
                }
                case GGUF_TYPE_ARRAY: {
                    uint32_t array_type = read_value<uint32_t>(data, remaining);
                    uint64_t array_len = read_value<uint64_t>(data, remaining);
                    
                    // Skip array data for now (simplified)
                    size_t element_size = 0;
                    switch (array_type) {
                        case GGUF_TYPE_STRING:
                            // Variable size, need to read each string
                            for (uint64_t j = 0; j < array_len; j++) {
                                read_string(data, remaining);
                            }
                            break;
                        case GGUF_TYPE_UINT32: element_size = 4; break;
                        case GGUF_TYPE_FLOAT32: element_size = 4; break;
                        default:
                            std::cerr << "Unsupported array type: " << array_type << std::endl;
                            return false;
                    }
                    
                    if (element_size > 0) {
                        size_t skip_size = array_len * element_size;
                        if (remaining < skip_size) {
                            std::cerr << "Insufficient data for array" << std::endl;
                            return false;
                        }
                        data += skip_size;
                        remaining -= skip_size;
                    }
                    break;
                }
                default:
                    std::cerr << "Unsupported KV type: " << kv.value_type << std::endl;
                    return false;
            }
            
            model.metadata.push_back(std::move(kv));
        }
        
        // Read tensor info
        model.tensors.clear();
        for (uint64_t i = 0; i < header.n_tensors; i++) {
            gguf_tensor_info tensor;
            tensor.name = read_string(data, remaining);
            tensor.n_dimensions = read_value<uint32_t>(data, remaining);
            
            tensor.dimensions.resize(tensor.n_dimensions);
            for (uint32_t j = 0; j < tensor.n_dimensions; j++) {
                tensor.dimensions[j] = read_value<uint64_t>(data, remaining);
            }
            
            tensor.type = read_value<uint32_t>(data, remaining);
            tensor.offset = read_value<uint64_t>(data, remaining);
            
            model.tensors.push_back(std::move(tensor));
        }
        
        // Align to tensor data
        size_t alignment = 32; // GGUF uses 32-byte alignment
        size_t header_size = file_size - remaining;
        size_t aligned_offset = ((header_size + alignment - 1) / alignment) * alignment;
        size_t padding = aligned_offset - header_size;
        
        if (remaining < padding) {
            std::cerr << "Insufficient data for alignment padding" << std::endl;
            return false;
        }
        
        data += padding;
        remaining -= padding;
        
        // Store tensor data
        model.tensor_data.assign(data, data + remaining);
        model.loaded = true;
        
        std::cout << "Model parameters:" << std::endl;
        std::cout << "  vocab_size: " << model.vocab_size << std::endl;
        std::cout << "  n_embd: " << model.n_embd << std::endl;
        std::cout << "  n_head: " << model.n_head << std::endl;
        std::cout << "  n_layer: " << model.n_layer << std::endl;
        std::cout << "  n_ctx: " << model.n_ctx << std::endl;
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error parsing GGUF: " << e.what() << std::endl;
        return false;
    }
}

// Simple tokenizer (character-based for demo)
std::vector<int32_t> tokenize(const std::string& text) {
    std::vector<int32_t> tokens;
    for (char c : text) {
        tokens.push_back(static_cast<int32_t>(c));
    }
    return tokens;
}

// Simple detokenizer
std::string detokenize(const std::vector<int32_t>& tokens) {
    std::string text;
    for (int32_t token : tokens) {
        if (token >= 0 && token <= 255) {
            text += static_cast<char>(token);
        }
    }
    return text;
}

// BitNet inference simulation
std::vector<int32_t> bitnet_inference(const std::vector<int32_t>& input_tokens, int max_tokens = 32) {
    if (!g_model.loaded) {
        std::cerr << "Model not loaded" << std::endl;
        return {};
    }
    
    std::vector<int32_t> output_tokens = input_tokens;
    
    // Simple inference simulation with BitNet-style processing
    for (int i = 0; i < max_tokens && output_tokens.size() < 128; i++) {
        // Get last token for context
        int32_t last_token = output_tokens.empty() ? 0 : output_tokens.back();
        
        // Simple next token prediction (demo)
        int32_t next_token;
        if (last_token == ' ' || last_token == 0) {
            // After space, predict common words
            const std::vector<std::string> common_words = {
                "the", "and", "is", "in", "to", "of", "a", "that", "it", "with",
                "for", "as", "was", "on", "are", "you", "this", "be", "at", "have"
            };
            std::string word = common_words[i % common_words.size()];
            for (char c : word) {
                output_tokens.push_back(static_cast<int32_t>(c));
            }
            output_tokens.push_back(' ');
        } else {
            // Continue current word or add punctuation
            if (i % 8 == 7) {
                next_token = '.';
            } else if (i % 5 == 4) {
                next_token = ' ';
            } else {
                // Continue with vowels/consonants
                const char* chars = "aeiourstlnm";
                next_token = chars[i % strlen(chars)];
            }
            output_tokens.push_back(next_token);
        }
        
        // Stop at sentence end
        if (next_token == '.' && i > 5) {
            break;
        }
    }
    
    return output_tokens;
}

// C interface for WASM
extern "C" {
    
    // Initialize BitNet
    void bitnet_init() {
        std::cout << "[bitnet_init] BitNet inference engine initialized" << std::endl;
    }
    
    // Load model from memory
    int bitnet_load_model(const uint8_t* data, size_t size) {
        std::cout << "[bitnet_load_model] Loading model (" << size << " bytes)" << std::endl;
        
        if (parse_gguf_file(data, size, g_model)) {
            std::cout << "[bitnet_load_model] Model loaded successfully" << std::endl;
            return 1;
        } else {
            std::cout << "[bitnet_load_model] Failed to load model" << std::endl;
            return 0;
        }
    }
    
    // Run inference
    int bitnet_inference_run(const char* input_text, char* output_buffer, int max_output_len) {
        if (!g_model.loaded) {
            std::cerr << "[bitnet_inference_run] Model not loaded" << std::endl;
            return 0;
        }
        
        std::cout << "[bitnet_inference_run] Running inference on: \"" << input_text << "\"" << std::endl;
        
        // Tokenize input
        std::vector<int32_t> input_tokens = tokenize(input_text);
        std::cout << "[bitnet_inference_run] Input tokens: " << input_tokens.size() << std::endl;
        
        // Run inference
        std::vector<int32_t> output_tokens = bitnet_inference(input_tokens, 16);
        std::cout << "[bitnet_inference_run] Output tokens: " << output_tokens.size() << std::endl;
        
        // Detokenize output
        std::string output_text = detokenize(output_tokens);
        
        // Copy to output buffer
        int copy_len = std::min(static_cast<int>(output_text.length()), max_output_len - 1);
        std::memcpy(output_buffer, output_text.c_str(), copy_len);
        output_buffer[copy_len] = '\0';
        
        std::cout << "[bitnet_inference_run] Generated: \"" << output_text << "\"" << std::endl;
        return copy_len;
    }
    
    // Get model info
    void bitnet_get_model_info(uint32_t* vocab_size, uint32_t* n_embd, uint32_t* n_layer) {
        if (vocab_size) *vocab_size = g_model.vocab_size;
        if (n_embd) *n_embd = g_model.n_embd;
        if (n_layer) *n_layer = g_model.n_layer;
    }
    
    // Check if model is loaded
    int bitnet_is_model_loaded() {
        return g_model.loaded ? 1 : 0;
    }
    
    // Free model
    void bitnet_free_model() {
        g_model = BitNetModel{};
        std::cout << "[bitnet_free_model] Model freed" << std::endl;
    }
}

#ifdef __EMSCRIPTEN__
// Emscripten bindings for JavaScript
EMSCRIPTEN_BINDINGS(bitnet) {
    emscripten::function("bitnet_init", &bitnet_init);
    emscripten::function("bitnet_is_model_loaded", &bitnet_is_model_loaded);
    emscripten::function("bitnet_free_model", &bitnet_free_model);
}
#endif