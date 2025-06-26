
// Ultra-minimal context configuration for WASM testing
export const minimalConfig = {
    n_ctx: 16,          // Absolute minimum context
    n_batch: 1,         // Single token batches
    n_threads: 1,       // Single thread
    temperature: 0.0,   // Deterministic output
    top_p: 1.0,
    top_k: 1,
    repeat_penalty: 1.0
};
