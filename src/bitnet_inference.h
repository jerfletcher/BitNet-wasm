#pragma once
#include <vector>
#include <string>
#include <cstdint>
#include <cstddef>

struct gguf_header {
    char magic[4];
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

struct BitNetModel {
    std::vector<gguf_kv_pair> metadata;
    std::vector<gguf_tensor_info> tensors;
    std::vector<uint8_t> tensor_data;
    bool loaded = false;
    uint32_t vocab_size = 0;
    uint32_t n_embd = 0;
    uint32_t n_head = 0;
    uint32_t n_layer = 0;
    uint32_t n_ctx = 0;
};

extern BitNetModel g_model;
bool parse_gguf_file(const uint8_t* file_data, size_t file_size, BitNetModel& model);
std::vector<int32_t> bitnet_inference(const std::vector<int32_t>& input_tokens, int max_tokens = 32);
std::vector<int32_t> tokenize(const std::string& text);
std::string detokenize(const std::vector<int32_t>& tokens);
