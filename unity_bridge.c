#include "llama.h"
#include <stdlib.h>
#include <string.h>

// Unity P/Invoke에서 안전하게 호출할 수 있는 단순화된 C 래퍼.
// 복잡한 구조체를 직접 마샬링하지 않고 opaque 포인터로 처리.

void unity_llama_backend_init(void) {
    llama_backend_init();
}

void unity_llama_backend_free(void) {
    llama_backend_free();
}

struct llama_model * unity_llama_model_load(const char * path, int n_gpu_layers) {
    struct llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = n_gpu_layers;
    return llama_model_load_from_file(path, params);
}

void unity_llama_model_free(struct llama_model * model) {
    llama_model_free(model);
}

struct llama_context * unity_llama_context_create(struct llama_model * model, int n_ctx, int n_batch) {
    struct llama_context_params params = llama_context_default_params();
    params.n_ctx = n_ctx;
    params.n_batch = n_batch;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    params.offload_kqv = true;
    return llama_init_from_model(model, params);
}

void unity_llama_context_free(struct llama_context * ctx) {
    llama_free(ctx);
}

const struct llama_vocab * unity_llama_get_vocab(const struct llama_model * model) {
    return llama_model_get_vocab(model);
}

int unity_llama_tokenize(const struct llama_vocab * vocab, const char * text, int * tokens, int n_max) {
    return llama_tokenize(vocab, text, strlen(text), tokens, n_max, true, true);
}

int unity_llama_decode_batch(struct llama_context * ctx, int * tokens, int n_tokens) {
    struct llama_batch batch = llama_batch_get_one(tokens, n_tokens);
    return llama_decode(ctx, batch);
}

int unity_llama_sample_greedy(struct llama_context * ctx, const struct llama_vocab * vocab) {
    // 마지막 토큰의 logits에서 argmax
    float * logits = llama_get_logits_ith(ctx, -1);
    int n_vocab = llama_vocab_n_tokens(vocab);
    int best = 0;
    float best_val = logits[0];
    for (int i = 1; i < n_vocab; i++) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best = i;
        }
    }
    return best;
}

int unity_llama_token_to_text(const struct llama_vocab * vocab, int token, char * buf, int buf_size) {
    return llama_token_to_piece(vocab, token, buf, buf_size, 0, true);
}

int unity_llama_vocab_eos(const struct llama_vocab * vocab) {
    return llama_vocab_eos(vocab);
}
