# Native GGUF-to-Flat Conversion Test

This document describes how to run and debug the native (non-WASM) test for converting a BitNet GGUF model to the flat binary format expected by the BitNet WASM module.

## How to Run the Native Test

1. **Build the test program:**

```bash
# From the BitNet-wasm root directory
# Make sure you have g++ and the headers in include/
g++ -Iinclude -o test_gguf_to_flat src/test_gguf_to_flat.cpp src/ggml-bitnet-lut.cpp
```

2. **Run the test with a GGUF model:**

```bash
./test_gguf_to_flat models/ggml-model-i2_s.gguf out.flat
```

- This will attempt to convert the GGUF model to a flat binary file (`out.flat`).
- The program prints detailed debug output about the GGUF parsing process.

## Where It Fails / Current Troubleshooting State

- The parser reads the GGUF header and starts parsing key-value (kv) pairs.
- It robustly skips known kv types, but if it encounters an unknown kv type, it breaks out of the kv loop and assumes the tensor section starts.
- In the current BitNet GGUF file, after about 24 kvs, a kv with an unknown type (e.g., type=2 for key `token_embd.weight`) is encountered. The parser then tries to parse the tensor section.
- However, the next bytes do not correspond to a valid tensor section (the name length is implausible, e.g., a huge number), so tensor parsing aborts.
- This suggests that the GGUF file contains extra kvs with custom/unknown types before the tensor section, and the parser's logic for skipping unknown types may not match the file's layout.

**Current debug output shows:**
- The parser prints something like:
  - `[gguf_to_flat] kv 24: key='token_embd.weight' ... type=2 ... Skipped unknown type 2 as len=2560 ...`
  - `[gguf_to_flat] Unknown or out-of-bounds kv type or value at kv 24, breaking to tensor section.`
  - `[gguf_to_flat] Unlikely tensor name_len=... aborting tensor parse.`
- No tensors are parsed, and the conversion fails with `tok_embeddings.weight not found or wrong type`.

## Next Steps for Troubleshooting

- The main issue is that the parser desynchronizes when it encounters unknown/custom kv types. The logic for skipping unknown types may not match the actual data layout in the BitNet GGUF file.
- To continue troubleshooting:
  1. Inspect the bytes at the kv/tensor boundary in the GGUF file to determine the correct way to skip unknown kv types.
  2. Compare with the Python GGUF parser or BitNet's own conversion scripts in `3rdparty/BitNet/utils/` to see how they handle these kvs.
  3. Adjust the C++ parser to either:
     - Correctly skip unknown/custom kv types (by reading the correct length and skipping), or
     - Use a more robust heuristic to detect the start of the tensor section.
  4. Add more debug prints for the raw bytes at the kv/tensor boundary to help diagnose the issue.

**You can pick up troubleshooting by editing `src/ggml-bitnet-lut.cpp` and re-running the test as above.**

---

_Last updated: 2025-06-22_
