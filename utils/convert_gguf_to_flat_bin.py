import struct
import numpy as np
import sys
import mmap

# This script converts a GGUF BitNet model to a flat binary buffer matching BitNetModelFlat/BitNetTransformerBlockFlat
# Usage: python convert_gguf_to_flat_bin.py input.gguf output.bin

# --- Helper: Read GGUF (very basic, for demo; use a real parser for production) ---
def read_gguf_weights(gguf_path):
    # This is a placeholder. You should use a real GGUF parser or llama.cpp's Python tools.
    # For now, we assume the GGUF file is a dict of numpy arrays (np.loadz or similar)
    # Replace this with your actual GGUF loader!
    import safetensors
    tensors = safetensors.safe_open(gguf_path, framework="np")
    weights = {k: tensors.get_tensor(k) for k in tensors.keys()}
    return weights

# --- Main conversion logic ---
def main():
    if len(sys.argv) != 3:
        print("Usage: python convert_gguf_to_flat_bin.py input.gguf output.bin")
        sys.exit(1)
    gguf_path, out_path = sys.argv[1], sys.argv[2]
    weights = read_gguf_weights(gguf_path)

    # Model hyperparams (replace with your actual model's values or parse from GGUF)
    dim = weights['tok_embeddings.weight'].shape[1]
    vocab_size = weights['tok_embeddings.weight'].shape[0]
    n_layers = len([k for k in weights if k.startswith('layers.') and k.endswith('.attention.wq.weight')])
    ffn_dim = weights['layers.0.feed_forward.w1.weight'].shape[0]

    # --- Flat buffer layout ---
    # 1. BitNetModelFlat struct (offsets, ints)
    # 2. All float/int8 arrays in order, as expected by C++
    # 3. BitNetTransformerBlockFlat structs for each layer

    # We'll build a list of (offset, data) for each array
    offset = struct.calcsize('I'*4 + 'i'*4)  # BitNetModelFlat size
    arrays = []

    # 1. Token embedding table (float32)
    token_embedding_table = weights['tok_embeddings.weight'].astype(np.float32).tobytes()
    token_embedding_table_offset = offset
    arrays.append(token_embedding_table)
    offset += len(token_embedding_table)

    # 2. Output projection (int8)
    output_proj = weights['output.weight'].astype(np.int8).tobytes()
    output_proj_offset = offset
    arrays.append(output_proj)
    offset += len(output_proj)

    # 3. Transformer layers
    layers_offset = offset
    block_structs = []
    for l in range(n_layers):
        # For each block, store offsets for all weights
        # (You may need to adapt these names to your GGUF keys)
        attn_norm_weight = weights[f'layers.{l}.attention_norm.weight'].astype(np.float32).tobytes()
        attn_norm_weight_offset = offset
        arrays.append(attn_norm_weight)
        offset += len(attn_norm_weight)

        attn_norm_bias = weights[f'layers.{l}.attention_norm.bias'].astype(np.float32).tobytes()
        attn_norm_bias_offset = offset
        arrays.append(attn_norm_bias)
        offset += len(attn_norm_bias)

        q_proj = weights[f'layers.{l}.attention.wq.weight'].astype(np.int8).tobytes()
        q_proj_offset = offset
        arrays.append(q_proj)
        offset += len(q_proj)

        k_proj = weights[f'layers.{l}.attention.wk.weight'].astype(np.int8).tobytes()
        k_proj_offset = offset
        arrays.append(k_proj)
        offset += len(k_proj)

        v_proj = weights[f'layers.{l}.attention.wv.weight'].astype(np.int8).tobytes()
        v_proj_offset = offset
        arrays.append(v_proj)
        offset += len(v_proj)

        o_proj = weights[f'layers.{l}.attention.wo.weight'].astype(np.int8).tobytes()
        o_proj_offset = offset
        arrays.append(o_proj)
        offset += len(o_proj)

        ffn_norm_weight = weights[f'layers.{l}.ffn_norm.weight'].astype(np.float32).tobytes()
        ffn_norm_weight_offset = offset
        arrays.append(ffn_norm_weight)
        offset += len(ffn_norm_weight)

        ffn_norm_bias = weights[f'layers.{l}.ffn_norm.bias'].astype(np.float32).tobytes()
        ffn_norm_bias_offset = offset
        arrays.append(ffn_norm_bias)
        offset += len(ffn_norm_bias)

        ffn_up_proj = weights[f'layers.{l}.feed_forward.w1.weight'].astype(np.int8).tobytes()
        ffn_up_proj_offset = offset
        arrays.append(ffn_up_proj)
        offset += len(ffn_up_proj)

        ffn_down_proj = weights[f'layers.{l}.feed_forward.w2.weight'].astype(np.int8).tobytes()
        ffn_down_proj_offset = offset
        arrays.append(ffn_down_proj)
        offset += len(ffn_down_proj)

        # Store the block struct
        block_structs.append((attn_norm_weight_offset, attn_norm_bias_offset, q_proj_offset, k_proj_offset, v_proj_offset, o_proj_offset, ffn_norm_weight_offset, ffn_norm_bias_offset, ffn_up_proj_offset, ffn_down_proj_offset))

    # Write BitNetModelFlat
    with open(out_path, 'wb') as f:
        # BitNetModelFlat: 4 offsets, 4 ints
        f.write(struct.pack('IIIIiiii',
            token_embedding_table_offset,
            output_proj_offset,
            layers_offset,
            0,  # reserved
            dim, vocab_size, n_layers, ffn_dim
        ))
        # Write all arrays
        for arr in arrays:
            f.write(arr)
        # Write all block structs
        for block in block_structs:
            f.write(struct.pack('IIIIIIIIII', *block))
    print(f"Wrote flat model to {out_path}")

if __name__ == '__main__':
    main()
