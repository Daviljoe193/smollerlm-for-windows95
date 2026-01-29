/* Inference for Llama-2 Transformer model in pure C */
/* Optimized for Windows 9x (Low RAM) & Pentium 3 (SSE) */
/* V9.2: Added 3DNow! support, because AMD got it out before Intel's SSE, and old is gold :3 */

#pragma GCC optimize("fast-math")

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdarg.h>
#include <signal.h>

#ifdef __SSE__
#include <xmmintrin.h>
#endif

#ifdef __3dNOW__
#include <mm3dnow.h>
#endif

// ----------------------------------------------------------------------------
// GLOBAL CONTROL
// ----------------------------------------------------------------------------
volatile int stop_generation = 0; 

// ----------------------------------------------------------------------------
// MEMORY ALLOCATOR (SSE ALIGNMENT)
// ----------------------------------------------------------------------------
void* malloc_aligned(size_t size) {
    void* ptr = malloc(size + 16);
    if (!ptr) return NULL;
    void* aligned = (void*)(((uintptr_t)ptr + 15) & ~0x0F);
    if (aligned == ptr) aligned = (void*)((uintptr_t)ptr + 16);
    *((void**)aligned - 1) = ptr;
    return aligned;
}

void* calloc_aligned(size_t num, size_t size) {
    size_t total = num * size;
    void* ptr = malloc_aligned(total);
    if (ptr) memset(ptr, 0, total);
    return ptr;
}

void free_aligned(void* ptr) {
    if (ptr) free(*((void**)ptr - 1));
}

// ----------------------------------------------------------------------------
// PLATFORM ADAPTERS
// ----------------------------------------------------------------------------

#if defined _WIN32
    #ifndef _WIN32_WINNT
    #define _WIN32_WINNT 0x0400
    #endif

    #include <windows.h>
    #include <conio.h>
    #include <io.h>

    // Win32 Colors
    #define COL_BG          0 
    #define COL_TEXT        (FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE) 
    #define COL_PROMPT      (FOREGROUND_GREEN | FOREGROUND_INTENSITY) 
    #define COL_PLACEHOLDER (FOREGROUND_INTENSITY) // Dark Gray
    #define COL_INFO        (FOREGROUND_BLUE | FOREGROUND_INTENSITY) 
    #define COL_BAR         (BACKGROUND_RED | BACKGROUND_GREEN | BACKGROUND_BLUE) 

    static inline int clock_gettime_impl(int clk_id, struct timespec *tp) {
        LARGE_INTEGER freq, count;
        if (QueryPerformanceFrequency(&freq)) {
            QueryPerformanceCounter(&count);
            tp->tv_sec = count.QuadPart / freq.QuadPart;
            tp->tv_nsec = (long)((count.QuadPart % freq.QuadPart) * 1000000000 / freq.QuadPart);
        } else {
            tp->tv_sec = time(NULL); tp->tv_nsec = 0;
        }
        return 0;
    }

    BOOL WINAPI CtrlHandler(DWORD fdwCtrlType) {
        if (fdwCtrlType == CTRL_C_EVENT) { stop_generation = 1; return TRUE; }
        return FALSE;
    }

#else
    #include <unistd.h>
    #include <sys/mman.h>
    #include <sys/time.h>
    #define mmap_impl mmap
    #define munmap_impl munmap
    #define malloc_aligned malloc
    #define calloc_aligned calloc
    #define free_aligned free
    static inline int clock_gettime_impl(int clk_id, struct timespec *tp) { return clock_gettime(clk_id, tp); }
#endif

long time_in_ms() {
    struct timespec time; clock_gettime_impl(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

// ----------------------------------------------------------------------------
// MODEL STRUCTS
// ----------------------------------------------------------------------------

typedef struct { int dim; int hidden_dim; int n_layers; int n_heads; int n_kv_heads; int vocab_size; int seq_len; int head_size; } Config;
typedef struct { int8_t* q; float* s; } QuantizedTensor;
typedef struct {
    QuantizedTensor token_embedding_table; float* rms_att_weight; float* rms_ffn_weight; 
    QuantizedTensor* wq; QuantizedTensor* wk; QuantizedTensor* wv; QuantizedTensor* wo;
    QuantizedTensor* w1; QuantizedTensor* w2; QuantizedTensor* w3;
    float* rms_final_weight; QuantizedTensor wcls;
} TransformerWeights;
typedef struct {
    float *x; float *xb; float *xb2; float *hb; float *hb2; float *q; float *k; float *v; float *att; float *logits; 
    float* key_cache; float* value_cache; float* rope_cos; float* rope_sin;
} RunState;
typedef struct {
    Config config; TransformerWeights weights; RunState state; 
    void* data; 
#if defined _WIN32
    HANDLE hFile; HANDLE hMapping;
#else
    int fd;
#endif
    size_t file_size; int group_size; 
} Transformer;

void free_run_state(RunState* s) {
    if (s->x) free_aligned(s->x); if (s->xb) free_aligned(s->xb); if (s->xb2) free_aligned(s->xb2);
    if (s->hb) free_aligned(s->hb); if (s->hb2) free_aligned(s->hb2); if (s->q) free_aligned(s->q);
    if (s->att) free_aligned(s->att); if (s->logits) free_aligned(s->logits);
    if (s->key_cache) free_aligned(s->key_cache); if (s->value_cache) free_aligned(s->value_cache);
    if (s->rope_cos) free_aligned(s->rope_cos); if (s->rope_sin) free_aligned(s->rope_sin);
}

void precompute_freqs(RunState* s, Config* p, int alloc_steps) {
    int head_size = p->head_size;
    for (int pos = 0; pos < alloc_steps; pos++) {
        for (int i = 0; i < head_size; i += 2) {
            float freq = 1.0f / powf(10000.0f, i / (float)head_size);
            float val = pos * freq;
            int idx = pos * (head_size / 2) + (i / 2);
            s->rope_cos[idx] = cosf(val); s->rope_sin[idx] = sinf(val);
        }
    }
}

void malloc_run_state(RunState* s, Config* p, int alloc_steps) {
    int q_dim = p->n_heads * p->head_size; int kv_dim = p->n_kv_heads * p->head_size;
    int xb_size = (q_dim > p->dim) ? q_dim : p->dim;
    if (alloc_steps <= 0) alloc_steps = p->seq_len;
    s->x = calloc_aligned(p->dim, sizeof(float)); s->xb = calloc_aligned(xb_size, sizeof(float)); s->xb2 = calloc_aligned(p->dim, sizeof(float));
    s->hb = calloc_aligned(p->hidden_dim, sizeof(float)); s->hb2 = calloc_aligned(p->hidden_dim, sizeof(float));
    s->q = calloc_aligned(q_dim, sizeof(float)); s->key_cache = calloc_aligned(p->n_layers * alloc_steps * kv_dim, sizeof(float));
    s->value_cache = calloc_aligned(p->n_layers * alloc_steps * kv_dim, sizeof(float));
    s->att = calloc_aligned(p->n_heads * alloc_steps, sizeof(float)); s->logits = calloc_aligned(p->vocab_size, sizeof(float));
    s->rope_cos = calloc_aligned(alloc_steps * (p->head_size / 2), sizeof(float)); s->rope_sin = calloc_aligned(alloc_steps * (p->head_size / 2), sizeof(float));
    if (!s->x || !s->key_cache || !s->rope_cos) { fprintf(stderr, "Fatal: RAM alloc failed.\n"); exit(1); }
    precompute_freqs(s, p, alloc_steps);
}

void init_quantized_tensor(QuantizedTensor* t, char** ptr_ref, int numel, int group_size) {
    char* ptr = *ptr_ref; t->q = (int8_t*)ptr; ptr += numel * sizeof(int8_t);
    t->s = (float*)ptr; ptr += (numel / group_size) * sizeof(float); *ptr_ref = ptr;
}

void load_weights(Transformer* t, int shared_weights) {
    Config* p = &t->config; TransformerWeights* w = &t->weights; char* ptr = (char*)t->data; ptr += 256; 
    w->rms_att_weight = (float*)ptr; ptr += p->n_layers * p->dim * sizeof(float);
    w->rms_ffn_weight = (float*)ptr; ptr += p->n_layers * p->dim * sizeof(float);
    w->rms_final_weight = (float*)ptr; ptr += p->dim * sizeof(float);
    w->wq = malloc(p->n_layers * sizeof(QuantizedTensor)); w->wk = malloc(p->n_layers * sizeof(QuantizedTensor));
    w->wv = malloc(p->n_layers * sizeof(QuantizedTensor)); w->wo = malloc(p->n_layers * sizeof(QuantizedTensor));
    w->w1 = malloc(p->n_layers * sizeof(QuantizedTensor)); w->w2 = malloc(p->n_layers * sizeof(QuantizedTensor));
    w->w3 = malloc(p->n_layers * sizeof(QuantizedTensor));
    unsigned long long dim = p->dim; unsigned long long att_dim = p->n_heads * p->head_size; 
    unsigned long long kv_dim = p->n_kv_heads * p->head_size; unsigned long long hidden_dim = p->hidden_dim;
    init_quantized_tensor(&w->token_embedding_table, &ptr, p->vocab_size * dim, t->group_size);
    for(int l=0; l<p->n_layers; l++) init_quantized_tensor(&w->wq[l], &ptr, dim * att_dim, t->group_size);
    for(int l=0; l<p->n_layers; l++) init_quantized_tensor(&w->wk[l], &ptr, dim * kv_dim, t->group_size);
    for(int l=0; l<p->n_layers; l++) init_quantized_tensor(&w->wv[l], &ptr, dim * kv_dim, t->group_size);
    for(int l=0; l<p->n_layers; l++) init_quantized_tensor(&w->wo[l], &ptr, att_dim * dim, t->group_size);
    for(int l=0; l<p->n_layers; l++) init_quantized_tensor(&w->w1[l], &ptr, dim * hidden_dim, t->group_size);
    for(int l=0; l<p->n_layers; l++) init_quantized_tensor(&w->w2[l], &ptr, hidden_dim * dim, t->group_size);
    for(int l=0; l<p->n_layers; l++) init_quantized_tensor(&w->w3[l], &ptr, dim * hidden_dim, t->group_size);
    if (shared_weights) w->wcls = w->token_embedding_table; else init_quantized_tensor(&w->wcls, &ptr, p->dim * p->vocab_size, t->group_size);
}

void build_transformer(Transformer *t, char* checkpoint_path, int steps) {
    FILE *file = fopen(checkpoint_path, "rb"); if (!file) { fprintf(stderr, "File not found: %s\n", checkpoint_path); exit(1); }
    uint32_t magic; fread(&magic, sizeof(uint32_t), 1, file);
    int version; fread(&version, sizeof(int), 1, file);
    fread(&t->config, sizeof(int) * 8, 1, file);
    uint8_t shared; fread(&shared, sizeof(uint8_t), 1, file);
    fread(&t->group_size, sizeof(int), 1, file); fclose(file);

#if defined _WIN32
    t->hFile = CreateFileA(checkpoint_path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (t->hFile == INVALID_HANDLE_VALUE) exit(1);
    t->file_size = GetFileSize(t->hFile, NULL);
    t->hMapping = CreateFileMapping(t->hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (t->hMapping == NULL) exit(1);
    t->data = MapViewOfFile(t->hMapping, FILE_MAP_READ, 0, 0, 0);
    if (t->data == NULL) exit(1);
#else
    t->fd = open(checkpoint_path, O_RDONLY);
    struct stat sb; fstat(t->fd, &sb); t->file_size = sb.st_size;
    t->data = mmap(NULL, t->file_size, PROT_READ, MAP_PRIVATE, t->fd, 0);
#endif
    load_weights(t, shared); malloc_run_state(&t->state, &t->config, steps);
}

void free_transformer(Transformer* t) {
    free(t->weights.wq); free(t->weights.wk); free(t->weights.wv); free(t->weights.wo);
    free(t->weights.w1); free(t->weights.w2); free(t->weights.w3);
#if defined _WIN32
    if (t->data) UnmapViewOfFile(t->data); if (t->hMapping) CloseHandle(t->hMapping); if (t->hFile != INVALID_HANDLE_VALUE) CloseHandle(t->hFile);
#else
    if (t->data != MAP_FAILED) munmap(t->data, t->file_size);
    if (t->fd != -1) close(t->fd);
#endif
    free_run_state(&t->state);
}

// ----------------------------------------------------------------------------
// SSE MATH
// ----------------------------------------------------------------------------
void rmsnorm(float* o, float* x, float* weight, int size) {
    #ifdef __SSE__
    float ss = 0.0f; int i = 0; __m128 sum_v = _mm_setzero_ps();
    for (; i <= size - 4; i += 4) { __m128 x_v = _mm_load_ps(&x[i]); sum_v = _mm_add_ps(sum_v, _mm_mul_ps(x_v, x_v)); }
    float temp[4]; _mm_storeu_ps(temp, sum_v); ss = temp[0] + temp[1] + temp[2] + temp[3];
    for (; i < size; i++) { ss += x[i] * x[i]; }
    ss /= size; ss += 1e-5f; __m128 ss_v = _mm_load_ss(&ss); ss_v = _mm_rsqrt_ss(ss_v); _mm_store_ss(&ss, ss_v);
    for (int j = 0; j < size; j++) { o[j] = weight[j] * (ss * x[j]); }
    
    #elif defined __3dNOW__
    float ss = 0.0f; 
    int i = 0; 
    __m64 sum_v = _m_from_int(0); 
    // MMX Sum
    for (; i <= size - 2; i += 2) { 
        __m64 x_v = *(__m64*)&x[i];       
        sum_v = _m_pfadd(sum_v, _m_pfmul(x_v, x_v));
    }
    sum_v = _m_pfacc(sum_v, sum_v); 
    float temp_ss; *(__m64*)&temp_ss = sum_v; ss = temp_ss;
    for (; i < size; i++) { ss += x[i] * x[i]; }
    
    _m_femms(); // CRITICAL: Clear MMX state before using FPU sqrtf
    
    ss /= size; ss += 1e-5f; ss = 1.0f / sqrtf(ss);

    // Back to MMX for multiplication
    __m64 ss_v = _m_from_float(ss); 
    ss_v = _m_punpckldq(ss_v, ss_v); 
    
    for (i = 0; i <= size - 2; i += 2) {
        __m64 w_v = *(__m64*)&weight[i];
        __m64 x_v = *(__m64*)&x[i];
        *(__m64*)&o[i] = _m_pfmul(w_v, _m_pfmul(ss_v, x_v));
    }
    _m_femms(); // CRITICAL: Clear MMX state before returning
    for (; i < size; i++) { o[i] = weight[i] * (ss * x[i]); }

    #else
    float ss = 0.0f; for (int j = 0; j < size; j++) { ss += x[j] * x[j]; } ss /= size; ss += 1e-5f; ss = 1.0f / sqrtf(ss); for (int j = 0; j < size; j++) { o[j] = weight[j] * (ss * x[j]); }
    #endif
}

void softmax(float* x, int size) {
    float max_val = x[0]; for (int i = 1; i < size; i++) { if (x[i] > max_val) max_val = x[i]; }
    float sum = 0.0f; for (int i = 0; i < size; i++) { x[i] = expf(x[i] - max_val); sum += x[i]; }
    for (int i = 0; i < size; i++) { x[i] /= sum; }
}

void matmul_q8(float* xout, float* x, QuantizedTensor* qt, int n, int d, int group_size) {
    #ifdef __SSE__
    for (int i = 0; i < d; i++) {
        int32_t in = i * n; float* s_ptr = &qt->s[in / group_size]; int8_t* w_ptr = &qt->q[in];
        __m128 sum_v = _mm_setzero_ps(); float temp_sum = 0.0f;
        for (int j = 0; j < n; j += group_size) {
            float scale = *s_ptr++; __m128 scale_v = _mm_load1_ps(&scale);
            int k = 0;
            for (; k <= group_size - 4; k += 4) {
                float w0=w_ptr[j+k], w1=w_ptr[j+k+1], w2=w_ptr[j+k+2], w3=w_ptr[j+k+3];
                sum_v = _mm_add_ps(sum_v, _mm_mul_ps(_mm_mul_ps(_mm_set_ps(w3,w2,w1,w0), scale_v), _mm_load_ps(&x[j+k])));
            }
            float temp[4]; _mm_storeu_ps(temp, sum_v); sum_v = _mm_setzero_ps(); temp_sum += temp[0]+temp[1]+temp[2]+temp[3];
            for (; k < group_size; k++) temp_sum += ((float)w_ptr[j+k] * scale) * x[j+k];
        }
        xout[i] = temp_sum;
    }
    #elif defined __3dNOW__
    // 3DNow! Implementation - FPU SAFE
    for (int i = 0; i < d; i++) {
        __m64 row_acc = _m_from_int(0); 
        int32_t in = i * n; 
        float* s_ptr = &qt->s[in / group_size]; 
        int8_t* w_ptr = &qt->q[in];

        for (int j = 0; j < n; j += group_size) {
            float scale = *s_ptr++;
            __m64 scale_v = _m_from_float(scale);
            scale_v = _m_punpckldq(scale_v, scale_v); 
            
            // Loop in steps of 2
            for (int k = 0; k < group_size; k += 2) {
                int32_t i0 = w_ptr[j+k];
                int32_t i1 = w_ptr[j+k+1];
                // Convert Int -> Float using MMX pipe (No FPU usage)
                __m64 w_v = _m_pi2fd(_mm_set_pi32(i1, i0)); 
                __m64 x_v = *(__m64*)&x[j+k];
                // acc += w * scale * x
                row_acc = _m_pfadd(row_acc, _m_pfmul(x_v, _m_pfmul(w_v, scale_v)));
            }
        }
        row_acc = _m_pfacc(row_acc, row_acc);
        float res; *(__m64*)&res = row_acc;
        xout[i] = res;
    }
    _m_femms(); // Clear state at end of layer
    #else
    for (int i = 0; i < d; i++) {
        float val = 0.0f; int32_t in = i * n; float* s_ptr = &qt->s[in / group_size]; int8_t* w_ptr = &qt->q[in];
        for (int j = 0; j < n; j += group_size) {
            float scale = *s_ptr++;
            for (int k = 0; k < group_size && (j+k) < n; k++) val += ((float)w_ptr[j+k] * scale) * x[j+k];
        }
        xout[i] = val;
    }
    #endif
}

float* forward(Transformer* t, int token, int pos, int stride_steps) {
    Config* p = &t->config; TransformerWeights* w = &t->weights; RunState* s = &t->state;
    float *x = s->x; int dim = p->dim; int kv_dim = p->n_kv_heads * p->head_size; int gs = t->group_size;

    int offset = token * dim;
    for (int i = 0; i < dim; i++) x[i] = (float)w->token_embedding_table.q[offset + i] * w->token_embedding_table.s[offset/gs + i/gs];

    for(unsigned long long l = 0; l < p->n_layers; l++) {
        rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);
        int loff = l * stride_steps * kv_dim; s->k = s->key_cache + loff + pos * kv_dim; s->v = s->value_cache + loff + pos * kv_dim;
        matmul_q8(s->q, s->xb, &w->wq[l], dim, p->n_heads * p->head_size, gs);
        matmul_q8(s->k, s->xb, &w->wk[l], dim, kv_dim, gs);
        matmul_q8(s->v, s->xb, &w->wv[l], dim, kv_dim, gs);

        #ifdef __3dNOW__
        _m_femms();
        #endif

        for (int i = 0; i < p->n_heads * p->head_size; i+=2) {
            int cidx = pos * (p->head_size / 2) + (i % p->head_size) / 2;
            float fcr = s->rope_cos[cidx], fci = s->rope_sin[cidx];
            float v0 = s->q[i], v1 = s->q[i+1]; s->q[i] = v0*fcr-v1*fci; s->q[i+1] = v0*fci+v1*fcr;
            if (i < kv_dim) { v0 = s->k[i]; v1 = s->k[i+1]; s->k[i] = v0*fcr-v1*fci; s->k[i+1] = v0*fci+v1*fcr; }
        }

        int h;
        for (h = 0; h < p->n_heads; h++) {
            float* q = s->q + h * p->head_size; float* att = s->att + h * stride_steps; 
            for (int t = 0; t <= pos; t++) {
                float* k = s->key_cache + loff + t * kv_dim + (h / (p->n_heads / p->n_kv_heads)) * p->head_size;
                float score = 0.0f; for (int i = 0; i < p->head_size; i++) { score += q[i] * k[i]; }
                att[t] = score / sqrtf(p->head_size);
            }
            softmax(att, pos + 1);
            float* xb = s->xb + h * p->head_size; memset(xb, 0, p->head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                float* v = s->value_cache + loff + t * kv_dim + (h / (p->n_heads / p->n_kv_heads)) * p->head_size;
                float a = att[t]; for (int i = 0; i < p->head_size; i++) { xb[i] += a * v[i]; }
            }
        }
        matmul_q8(s->xb2, s->xb, &w->wo[l], p->n_heads * p->head_size, dim, gs);
        for (int i = 0; i < dim; i++) { x[i] += s->xb2[i]; }
        rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);
        matmul_q8(s->hb, s->xb, &w->w1[l], dim, p->hidden_dim, gs);
        matmul_q8(s->hb2, s->xb, &w->w3[l], dim, p->hidden_dim, gs);

        #ifdef __3dNOW__
        _m_femms();
        #endif

        for (int i = 0; i < p->hidden_dim; i++) { float val = s->hb[i]; val = val / (1.0f + expf(-val)); s->hb[i] = val * s->hb2[i]; }
        matmul_q8(s->xb, s->hb, &w->w2[l], p->hidden_dim, dim, gs);
        for (int i = 0; i < dim; i++) { x[i] += s->xb[i]; }
    }
    rmsnorm(x, x, w->rms_final_weight, dim);
    matmul_q8(s->logits, x, &w->wcls, p->dim, p->vocab_size, gs);
    return s->logits;
}

// ----------------------------------------------------------------------------
// TOKENIZER & SAMPLER
// ----------------------------------------------------------------------------
typedef struct { char *str; int id; } TokenIndex;
typedef struct { char** vocab; float* vocab_scores; TokenIndex *sorted_vocab; int vocab_size; unsigned int max_token_length; unsigned char byte_pieces[512]; } Tokenizer;
typedef struct { float prob; int index; } ProbIndex;
typedef struct { int vocab_size; ProbIndex* probindex; float temperature; float topp; int topk; unsigned long long rng_state; } Sampler;

int compare_tokens(const void *a, const void *b) { return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str); }
char* decode(Tokenizer* t, int prev_token, int token) { char *piece = t->vocab[token]; if (prev_token == 1 && piece[0] == ' ') piece++; unsigned char byte_val; if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) piece = (char*)t->byte_pieces + byte_val * 2; return piece; }
void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    if (text == NULL) exit(1);
    if (!t->sorted_vocab) { t->sorted_vocab = malloc(t->vocab_size * sizeof(TokenIndex)); for (int i = 0; i < t->vocab_size; i++) { t->sorted_vocab[i].str = t->vocab[i]; t->sorted_vocab[i].id = i; } qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens); }
    char* str_buffer = malloc((t->max_token_length*2 +1 +2) * sizeof(char));
    size_t str_len = 0; *n_tokens = 0; if (bos) tokens[(*n_tokens)++] = 1;
    for (char *c = text; *c != '\0'; c++) {
        if ((*c & 0xC0) != 0x80) str_len = 0;
        str_buffer[str_len++] = *c; str_buffer[str_len] = '\0';
        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) continue;
        TokenIndex tok = { .str = str_buffer }; TokenIndex *res = bsearch(&tok, t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
        if (res) tokens[(*n_tokens)++] = res->id; else for (int i=0; i < str_len; i++) tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
        str_len = 0;
    }
    while (1) {
        float best_score = -1e10; int best_id = -1; int best_idx = -1;
        for (int i=0; i < (*n_tokens-1); i++) {
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            TokenIndex tok = { .str = str_buffer };  TokenIndex *res = bsearch(&tok, t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
            if (res && t->vocab_scores[res->id] > best_score) { best_score = t->vocab_scores[res->id]; best_id = res->id; best_idx = i; }
        }
        if (best_idx == -1) break;
        tokens[best_idx] = best_id; for (int i = best_idx+1; i < (*n_tokens-1); i++) tokens[i] = tokens[i+1]; (*n_tokens)--;
    }
    if (eos) tokens[(*n_tokens)++] = 2; free(str_buffer);
}
void build_tokenizer(Tokenizer* t, char* path, int vs) {
    t->vocab_size = vs; t->vocab = malloc(vs * sizeof(char*)); t->vocab_scores = malloc(vs * sizeof(float)); t->sorted_vocab = NULL;
    for (int i = 0; i < 256; i++) { t->byte_pieces[i*2] = (unsigned char)i; t->byte_pieces[i*2+1] = '\0'; }
    FILE *file = fopen(path, "rb"); if (!file) exit(1);
    fread(&t->max_token_length, sizeof(int), 1, file);
    for (int i = 0; i < vs; i++) { fread(t->vocab_scores + i, sizeof(float), 1, file); int len; fread(&len, sizeof(int), 1, file); t->vocab[i] = malloc(len + 1); fread(t->vocab[i], len, 1, file); t->vocab[i][len] = '\0'; }
    fclose(file);
}
int sample(Sampler* s, float* logits) {
    if (s->temperature == 0.0f) { int max_i=0; float max_p=logits[0]; for(int i=1;i<s->vocab_size;i++) if(logits[i]>max_p){max_i=i;max_p=logits[i];} return max_i; }
    for (int q=0; q<s->vocab_size; q++) logits[q] /= s->temperature; softmax(logits, s->vocab_size);
    if (s->topk > 0 && s->topk < s->vocab_size) {
        for (int i = 0; i < s->vocab_size; i++) { s->probindex[i].index = i; s->probindex[i].prob = logits[i]; }
        int compare(const void* a, const void* b) { ProbIndex* a_ = (ProbIndex*) a; ProbIndex* b_ = (ProbIndex*) b; if (a_->prob > b_->prob) return -1; if (a_->prob < b_->prob) return 1; return 0; }
        qsort(s->probindex, s->vocab_size, sizeof(ProbIndex), compare);
        float topk_sum = 0.0f; for (int i = 0; i < s->topk; i++) topk_sum += s->probindex[i].prob;
        float coin = (float)rand() / (float)RAND_MAX * topk_sum; float cdf = 0.0f;
        for (int i = 0; i < s->topk; i++) { cdf += s->probindex[i].prob; if (coin < cdf) return s->probindex[i].index; }
        return s->probindex[s->topk-1].index;
    }
    float coin = (float)rand() / (float)RAND_MAX;
    float cdf = 0.0f; for (int i = 0; i < s->vocab_size; i++) { cdf += logits[i]; if (coin < cdf) return i; }
    return s->vocab_size - 1;
}

// ----------------------------------------------------------------------------
// WINDOWS 95 TUI ENGINE (V9)
// ----------------------------------------------------------------------------
#if defined _WIN32

#define MAX_HIST 2000
#define SCREEN_W 100 // Safe Buffer

typedef struct {
    char lines[MAX_HIST][SCREEN_W]; 
    int line_lens[MAX_HIST];
    int count; 
    int view_offset; 
    char input[1024];
    int input_len;
    int cols, rows;
    HANDLE hOut;
    HANDLE hIn;
    int mode; // 0=Input, 1=Generating
} TUI;

TUI tui;

void tui_init() {
    tui.hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    tui.hIn = GetStdHandle(STD_INPUT_HANDLE);
    DWORD mode;
    GetConsoleMode(tui.hIn, &mode);
    SetConsoleMode(tui.hIn, mode | ENABLE_MOUSE_INPUT | ENABLE_WINDOW_INPUT);
    SetConsoleCtrlHandler(CtrlHandler, TRUE);
    CONSOLE_SCREEN_BUFFER_INFO csbi;
    GetConsoleScreenBufferInfo(tui.hOut, &csbi);
    tui.cols = csbi.srWindow.Right - csbi.srWindow.Left + 1;
    tui.rows = csbi.srWindow.Bottom - csbi.srWindow.Top + 1;
    if (tui.cols > SCREEN_W) tui.cols = SCREEN_W - 1;
    if (tui.cols < 20) tui.cols = 80; 
    tui.count = 0;
    tui.view_offset = 0;
    tui.input_len = 0;
    tui.input[0] = '\0';
    tui.mode = 0;
}

// Helper to force a new line entry in history
void tui_newline() {
    if (tui.count >= MAX_HIST) {
        memmove(&tui.lines[0], &tui.lines[1], (MAX_HIST-1) * SCREEN_W);
        memmove(&tui.line_lens[0], &tui.line_lens[1], (MAX_HIST-1) * sizeof(int));
        tui.count--;
    }
    tui.lines[tui.count][0] = '\0';
    tui.line_lens[tui.count] = 0;
    tui.count++;
}

// Wraps text into buffer. 
// Standard behavior: Append to current line. 
// If text has '\n', move to next line logic.
void tui_append(const char* txt) {
    if (tui.count == 0) tui_newline(); // Ensure at least one line exists

    const char* ptr = txt;
    while (*ptr) {
        int tail = tui.count - 1;
        int cur_len = tui.line_lens[tail];
        int avail = (tui.cols - 1) - cur_len;
        
        if (*ptr == '\n') {
            tui_newline();
            ptr++;
            continue;
        }

        if (avail > 0) {
            tui.lines[tail][cur_len] = *ptr;
            tui.lines[tail][cur_len+1] = '\0';
            tui.line_lens[tail]++;
            ptr++;
        } else {
            // Auto-wrap
            tui_newline();
            // Retry char
        }
    }
}

void tui_draw() {
    CHAR_INFO* buf = malloc(sizeof(CHAR_INFO) * tui.cols * tui.rows);
    if (!buf) return;
    for(int i=0; i<tui.cols*tui.rows; i++) { buf[i].Char.AsciiChar=' '; buf[i].Attributes=COL_BG; }

    // Start filling from row input_row (the active line) UPWARDS.
    // Determine where the active input line should be drawn.
    // If we have fewer lines than rows, we start at tui.count.
    // If we have more, we implicitly scroll, so input is at tui.rows - 1.
    
    int input_row = 0;
    if (tui.count < tui.rows - 1) {
        input_row = tui.count;
    } else {
        input_row = tui.rows - 1;
    }

    // Render History: 0 to tui.count-1
    // We map: history[last] -> input_row - 1
    int screen_y = input_row - 1;
    int hist_idx = tui.count - 1 - tui.view_offset;

    while (screen_y >= 0) {
        if (hist_idx >= 0 && hist_idx < tui.count) {
            char* l = tui.lines[hist_idx];
            int len = tui.line_lens[hist_idx];
            int attr = COL_TEXT;
            if (strncmp(l, ">>>", 3) == 0) attr = COL_PROMPT;
            else if (l[0] == '/') attr = COL_INFO; 

            for (int c=0; c<len; c++) {
                int idx = screen_y*tui.cols + c;
                buf[idx].Char.AsciiChar = l[c];
                buf[idx].Attributes = attr;
            }
        }
        screen_y--;
        hist_idx--;
    }

    // Render Active Input Line
    int base_idx = input_row * tui.cols;
    if (tui.mode == 0) {
        const char* prompt = ">>> ";
        for(int i=0; i<4; i++) {
            buf[base_idx+i].Char.AsciiChar = prompt[i];
            buf[base_idx+i].Attributes = COL_PROMPT;
        }
        if (tui.input_len == 0) {
            const char* ph = "Send a message (/? for help)";
            for (int i=0; ph[i]; i++) {
                buf[base_idx+4+i].Char.AsciiChar = ph[i];
                buf[base_idx+4+i].Attributes = COL_PLACEHOLDER; 
            }
        } else {
            for (int i=0; i<tui.input_len && (i+4 < tui.cols); i++) {
                buf[base_idx+4+i].Char.AsciiChar = tui.input[i];
                buf[base_idx+4+i].Attributes = COL_TEXT;
            }
        }
    } else {
        // Generating: The active input line is effectively the last line of history being generated
        // We actually want the 'input' cursor to chase the history in generation mode.
        // The render logic above handles buffer lines.
        // We just need to hide the ">>>" prompt when generating if it was already pushed to history.
    }

    // Scrollbar
    float scroll_pos = 1.0f;
    if (tui.count > tui.rows) {
        scroll_pos = 1.0f - ((float)tui.view_offset / (float)(tui.count - tui.rows + 1));
    }
    if (scroll_pos < 0) scroll_pos = 0; if (scroll_pos > 1) scroll_pos = 1;
    int knob = (int)((tui.rows - 1) * scroll_pos);
    if (knob >= tui.rows) knob = tui.rows - 1;
    for(int r=0; r<tui.rows; r++) {
        int idx = r*tui.cols + (tui.cols-1);
        buf[idx].Attributes = COL_BAR;
        buf[idx].Char.AsciiChar = (r==knob)?219:176;
    }

    COORD s = {tui.cols, tui.rows}; COORD c = {0, 0}; SMALL_RECT r = {0, 0, tui.cols-1, tui.rows-1};
    WriteConsoleOutput(tui.hOut, buf, s, c, &r);
    
    // Hardware Cursor
    COORD cur;
    if (tui.mode == 0) {
        cur.Y = input_row;
        cur.X = 4 + tui.input_len;
        if (tui.input_len==0) cur.X = 4;
    } else {
        // Generation: cursor ends at last line of history
        int last_hist_y = input_row; // Input row becomes next available line
        // But we want it at the END of the text we just printed
        // If we are scrolling (count > rows), the last line is at rows-1
        // If we are not scrolling, the last line is at count-1 + adjustment? 
        // Actually simpler: The loop above renders hist_idx 0 at input_row-1.
        // The most recent line (index tui.count-1) is drawn at input_row - 1 - view_offset
        int end_y = (tui.count < tui.rows) ? tui.count - 1 : tui.rows - 2; 
        // Logic fix:
        // If count < rows-1: InputRow = count. Last history line is at count-1.
        // If count >= rows-1: InputRow = rows-1. Last history history is at rows-2.
        int cursor_y = input_row - 1 - tui.view_offset;
        if (cursor_y >= 0 && cursor_y < tui.rows - 1) {
            cur.Y = cursor_y;
            cur.X = tui.line_lens[tui.count-1];
        } else {
             cur.Y = tui.rows-1; cur.X = 0; // Offscreen/scrolled
        }
    }
    SetConsoleCursorPosition(tui.hOut, cur);
    free(buf);
}
#endif

// ----------------------------------------------------------------------------
// GENERATION LOGIC (ONE-SHOT)
// ----------------------------------------------------------------------------

void generate(Transformer *t, Tokenizer *tok, Sampler *samp, char *prompt, int steps) {
    if (prompt == NULL) prompt = "";

    // Encode
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int));
    encode(tok, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) { fprintf(stderr, "Something is wrong, expected at least 1 prompt token\n"); exit(1); }

    // Start loop
    long start = 0; 
    int next;
    int token = prompt_tokens[0]; 
    int pos = 0;
    
    while (pos < steps) {
        if (stop_generation) { printf("\n^C Interrupted"); break; }

        float* logits = forward(t, token, pos, steps);

        if (pos < num_prompt_tokens - 1) {
            next = prompt_tokens[pos + 1];
        } else {
            next = sample(samp, logits);
        }
        pos++;

        if (next == 2) break; // EOS

        char* piece = decode(tok, token, next);
        // Print
        if (piece) { printf("%s", piece); fflush(stdout); }
        token = next;

        if (start == 0) start = time_in_ms();
    }
    printf("\n");

    if (pos > 1) {
        long end = time_in_ms();
        fprintf(stderr, "achieved tok/s: %f\n", (pos-1) / (double)(end-start)*1000);
    }
    free(prompt_tokens);
}

// ----------------------------------------------------------------------------
// CHAT LOGIC
// ----------------------------------------------------------------------------

void chat_loop(Transformer *t, Tokenizer *tok, Sampler *samp, int n_ctx, int use_tui, char* cli_system_prompt) {
    int id_im_start = 1; int id_im_end = 2; int id_nl = 198;    
    int id_system = 9690; int id_user = 4093; int id_ass1 = 520; int id_ass2 = 9531;   
    
    char* sys_prompt = "You are SmolLM, a helpful assistant.";
    if (cli_system_prompt != NULL) sys_prompt = cli_system_prompt;
    
    int* tokens = malloc(n_ctx * sizeof(int));
    int n_tok = 0, n_chunk = 0;
    int pos = 0;

    // Prefill
    tokens[n_tok++] = id_im_start; tokens[n_tok++] = id_system; tokens[n_tok++] = id_nl;
    encode(tok, sys_prompt, 0, 0, tokens+n_tok, &n_chunk); n_tok += n_chunk;
    tokens[n_tok++] = id_im_end; tokens[n_tok++] = id_nl;
    for(int i=0; i<n_tok; i++) forward(t, tokens[i], pos++, n_ctx);
    int initial_pos = pos; 

    #if defined _WIN32
    if (use_tui) { tui_init(); tui_draw(); } else 
    #endif
    { printf(">>> "); fflush(stdout); }

    char input_buf[1024];

    while(1) {
        #if defined _WIN32
        if (use_tui) {
            tui.mode = 0;
            tui.input[0] = '\0'; tui.input_len = 0; tui.view_offset = 0;
            // Interaction Loop
            INPUT_RECORD ir[32]; DWORD n;
            while(1) {
                tui_draw();
                ReadConsoleInput(tui.hIn, ir, 1, &n);
                if (ir[0].EventType == KEY_EVENT && ir[0].Event.KeyEvent.bKeyDown) {
                    char c = ir[0].Event.KeyEvent.uChar.AsciiChar;
                    WORD vk = ir[0].Event.KeyEvent.wVirtualKeyCode;
                    if (c == 13) break; 
                    else if (c == 8) { if (tui.input_len > 0) tui.input[--tui.input_len] = '\0'; }
                    else if (c >= 32) { if (tui.input_len < 1000) { tui.input[tui.input_len++] = c; tui.input[tui.input_len] = '\0'; } }
                    else {
                         if (vk == 33) { tui.view_offset += 5; } 
                         if (vk == 34) { tui.view_offset -= 5; if(tui.view_offset<0) tui.view_offset=0; } 
                    }
                }
            }
            strcpy(input_buf, tui.input);
            char hist[1050]; sprintf(hist, ">>> %s\n", input_buf); // Append with newline!
            tui_append(hist);
        } else 
        #endif
        { if(!fgets(input_buf, 1024, stdin)) break; size_t len=strlen(input_buf); if(len>0 && input_buf[len-1]=='\n') input_buf[len-1]=0; }

        if (input_buf[0] == '/') {
            if (strncmp(input_buf, "/bye", 4)==0) break;
            if (strncmp(input_buf, "/clear", 6)==0) {
                 pos = initial_pos; 
                 #if defined _WIN32
                 if (use_tui) { tui_append("Cleared session context\n"); } else
                 #endif
                 printf("Cleared session context\n");
                 continue; 
            }
            if (strncmp(input_buf, "/?", 2)==0 || strncmp(input_buf, "/help", 5)==0) {
                 #if defined _WIN32
                 if (use_tui) {
                     tui_append("Available Commands:\n");
                     tui_append("  /set parameter [temp|top_k] <val>\n");
                     tui_append("  /clear\n"); tui_append("  /bye\n");
                 } else
                 #endif
                 printf("Commands: /set, /clear, /bye\n");
                 continue;
            }
             if (strncmp(input_buf, "/set parameter temperature", 26)==0) { samp->temperature = atof(input_buf + 27); continue; }
             if (strncmp(input_buf, "/set parameter top_k", 20)==0) { samp->topk = atoi(input_buf + 21); continue; }
        }

        stop_generation = 0;
        int user_tokens[1024]; int n_user_tokens;
        int prompt_tokens[1024]; int n_prompt = 0;
        
        prompt_tokens[n_prompt++] = id_im_start; prompt_tokens[n_prompt++] = id_user; prompt_tokens[n_prompt++] = id_nl;
        encode(tok, input_buf, 0, 0, user_tokens, &n_user_tokens);
        for(int i=0; i<n_user_tokens; i++) prompt_tokens[n_prompt++] = user_tokens[i];
        prompt_tokens[n_prompt++] = id_im_end; prompt_tokens[n_prompt++] = id_nl;
        prompt_tokens[n_prompt++] = id_im_start; prompt_tokens[n_prompt++] = id_ass1; prompt_tokens[n_prompt++] = id_ass2; prompt_tokens[n_prompt++] = id_nl;

        for(int i=0; i<n_prompt; i++) forward(t, prompt_tokens[i], pos++, n_ctx);
        int token = prompt_tokens[n_prompt-1];

        // --- GENERATION ---
        #if defined _WIN32
        if (use_tui) tui.mode = 1; 
        #endif

        while (pos < n_ctx) {
            if (stop_generation) {
                #if defined _WIN32
                if(use_tui) tui_append("^C Interrupted\n"); else
                #endif
                printf("\n^C Interrupted");
                break;
            }
            float* logits = forward(t, token, pos, n_ctx);
            int next = sample(samp, logits);
            pos++;
            if (next == id_im_end || next == 2) break; 

            char* piece = decode(tok, token, next);
            #if defined _WIN32
            if (use_tui) {
                if(piece) {
                    tui_append(piece); 
                    tui_draw();
                }
                INPUT_RECORD ir; DWORD n;
                if (PeekConsoleInput(tui.hIn, &ir, 1, &n) && n > 0) {}
            } else 
            #endif
            { printf("%s", piece); fflush(stdout); }
            token = next;
        }
        
        #if defined _WIN32
        if (use_tui) {
             tui_append("\n"); // Fin. stream
             tui_draw();
        } else { printf("\n>>> "); }
        #else
        printf("\n>>> "); 
        #endif
        fflush(stdout);
    }
    free(tokens);
}

void error_usage() {
    fprintf(stderr, "Usage:   run <checkpoint> [options]\n");
    fprintf(stderr, "Example: run model.bin -n 256 -i \"Once upon a time\"\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
    fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
    fprintf(stderr, "  -m <string> mode: generate|chat, default: chat\n");
    fprintf(stderr, "  -y <string> (optional) system prompt in chat mode\n");
    exit(1);
}

int main(int argc, char *argv[]) {
    char *checkpoint = NULL; char *tokenizer = "tokenizer.bin";
    int steps = 512; float temp = 0.8f; float topp = 0.9f;
    char *prompt = NULL; char *mode = "chat"; char *sys_prompt = NULL;
    
    // 1. Generate a hardware-based default seed using QueryPerformanceCounter
    unsigned long long rng_seed = 0;
#if defined _WIN32
    LARGE_INTEGER qpc;
    QueryPerformanceCounter(&qpc);
    rng_seed = (unsigned long long)qpc.QuadPart;
#else
    rng_seed = (unsigned long long)time(NULL);
#endif

    if (argc >= 2) checkpoint = argv[1]; else { error_usage(); }
    
    for (int i = 2; i < argc; i+=2) {
        if (i + 1 >= argc || argv[i][0] != '-') break;
        if (argv[i][1] == 't') temp = atof(argv[i + 1]);
        else if (argv[i][1] == 'p') topp = atof(argv[i + 1]);
        else if (argv[i][1] == 'n') steps = atoi(argv[i + 1]);
        else if (argv[i][1] == 'z') tokenizer = argv[i + 1];
        else if (argv[i][1] == 'i') prompt = argv[i + 1];
        else if (argv[i][1] == 'm') mode = argv[i + 1];
        else if (argv[i][1] == 'y') sys_prompt = argv[i + 1];
        else if (argv[i][1] == 's') rng_seed = (unsigned long long)atol(argv[i + 1]);
        else error_usage();
    }
    
    // 3. IMPORTANT: Seed the standard C RNG because run-smol.c uses rand()
    srand((unsigned int)rng_seed); 

    Transformer transformer; build_transformer(&transformer, checkpoint, steps);
    Tokenizer tok; build_tokenizer(&tok, tokenizer, transformer.config.vocab_size);
    
    // 4. Update the sampler struct (mostly for book-keeping)
    Sampler samp; 
    samp.vocab_size = transformer.config.vocab_size; 
    samp.temperature = temp; 
    samp.topp = topp; 
    samp.topk = 40; 
    samp.rng_state = rng_seed; // Set this too, though your current sample() uses rand()
    samp.probindex = malloc(samp.vocab_size*sizeof(ProbIndex));
    
    if (strcmp(mode, "generate") == 0) {
        generate(&transformer, &tok, &samp, prompt, steps);
    } else {
        chat_loop(&transformer, &tok, &samp, steps, 1, sys_prompt);
    }
    
    free(samp.probindex); free_transformer(&transformer);
    return 0;
}
