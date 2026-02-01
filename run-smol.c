/* 
   Run-Smol Unified: LLM Inference for Legacy Hardware
   Supports: 
   1. Windows 95/98/XP on x86 (Scalar, MMX, SSE, 3DNow!)
   2. Mac OS X 10.4 on PowerPC G4 (AltiVec)
   
   Merged Version 12.0 - The "Pentium II" Update
*/

#pragma GCC optimize("fast-math")

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#include <stdarg.h>
#include <signal.h>

/* Handle stdint.h availability for legacy compilers (MSVC 6.0) */
#if defined(_MSC_VER) && (_MSC_VER < 1600)
    typedef signed char int8_t;
    typedef unsigned char uint8_t;
    typedef signed int int32_t;
    typedef unsigned int uint32_t;
    typedef signed __int64 int64_t;
    typedef unsigned __int64 uint64_t;
    typedef short int16_t;
#else
    #include <stdint.h>
#endif

/* ---------------------------------------------------------------------------- */
/* COMPILER COMPATIBILITY & INTRINSICS */
/* ---------------------------------------------------------------------------- */

/* C89 Compatibility for 'inline' */
#ifndef __cplusplus
    #ifdef __GNUC__
        #define inline __inline__
    #elif defined(_MSC_VER)
        #define inline __inline
    #else
        #define inline
    #endif
#endif

/* SSE Support */
#if defined(__SSE__)
    #include <xmmintrin.h>
#endif

/* 3DNow! Support */
#if defined(__3dNOW__)
    #include <mm3dnow.h>
#endif

/* MMX Support (Pentium MMX / Pentium II) */
#if defined(__MMX__)
    #include <mmintrin.h>
#endif

/* AltiVec / G4 Support */
#if defined(__ALTIVEC__)
    #include <altivec.h>
    #if !defined(__APPLE__)
        #undef bool
        #undef vector
        #undef pixel
    #endif
    
    static inline vector float vec_splats_poly(float x) {
        union { float f[4]; vector float v; } u;
        u.f[0] = x; u.f[1] = x; u.f[2] = x; u.f[3] = x;
        return u.v;
    }
    #define vec_splats vec_splats_poly

    static inline vector unsigned char vec_load_unaligned(unsigned char* ptr) {
        vector unsigned char v1 = vec_ld(0, ptr);
        vector unsigned char v2 = vec_ld(16, ptr);
        vector unsigned char mask = vec_lvsl(0, ptr);
        return vec_perm(v1, v2, mask);
    }
#endif

/* ---------------------------------------------------------------------------- */
/* ENDIANNESS CONTROL */
/* ---------------------------------------------------------------------------- */
#if defined(__G4__) || defined(__ppc__) || defined(__BIG_ENDIAN__)
    #define NEEDS_BSWAP 1
#endif

static inline uint32_t bswap32(uint32_t x) {
    return ((x & 0xFF000000u) >> 24) | ((x & 0x00FF0000u) >> 8) |
           ((x & 0x0000FF00u) << 8)  | ((x & 0x000000FFu) << 24);
}

static inline float bswap_float(float x) {
    union { float f; uint32_t i; } u;
    u.f = x;
    u.i = bswap32(u.i);
    return u.f;
}

/* ---------------------------------------------------------------------------- */
/* PLATFORM INCLUDES & DEFINES */
/* ---------------------------------------------------------------------------- */

#if defined _WIN32
    #ifndef _WIN32_WINNT
    #define _WIN32_WINNT 0x0400
    #endif
    #include <windows.h>
    #include <conio.h>
    #include <io.h>

    /* Win32 Colors */
    #define COL_BG          0 
    #define COL_TEXT        (FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE) 
    #define COL_PROMPT      (FOREGROUND_GREEN | FOREGROUND_INTENSITY) 
    #define COL_PLACEHOLDER (FOREGROUND_INTENSITY) 
    #define COL_INFO        (FOREGROUND_BLUE | FOREGROUND_INTENSITY) 
    #define COL_BAR         (BACKGROUND_RED | BACKGROUND_GREEN | BACKGROUND_BLUE) 
#else
    #include <unistd.h>
    #include <sys/mman.h>
    #include <sys/time.h>
    #include <sys/stat.h>
    #include <termios.h> 

    #define ANSI_COLOR_GREEN   "\x1b[1;32m"
    #define ANSI_COLOR_GRAY    "\x1b[90m" 
    #define ANSI_COLOR_BLUE    "\x1b[1;34m"
    #define ANSI_COLOR_RESET   "\x1b[0m"
    #define ANSI_CLEAR_SCREEN  "\033[2J\033[H"
    #define ANSI_CLEAR_LINE    "\033[K"
#endif

/* ---------------------------------------------------------------------------- */
/* GLOBAL CONTROL */
/* ---------------------------------------------------------------------------- */
volatile int stop_generation = 0; 

/* ---------------------------------------------------------------------------- */
/* MEMORY ALLOCATOR */
/* ---------------------------------------------------------------------------- */
void* malloc_aligned(size_t size) {
    void* ptr = malloc(size + 16);
    void* aligned;
    if (!ptr) return NULL;
    aligned = (void*)(((uintptr_t)ptr + 15) & ~0x0F);
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

/* ---------------------------------------------------------------------------- */
/* PLATFORM ADAPTERS (TIME & IO) */
/* ---------------------------------------------------------------------------- */

#if defined _WIN32
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
    #ifdef __APPLE__
    static inline int clock_gettime_impl(int clk_id, struct timespec *tp) {
        struct timeval tv;
        gettimeofday(&tv, NULL);
        tp->tv_sec = tv.tv_sec;
        tp->tv_nsec = tv.tv_usec * 1000;
        return 0;
    }
    #else
    static inline int clock_gettime_impl(int clk_id, struct timespec *tp) { return clock_gettime(clk_id, tp); }
    #endif
    void handle_sigint(int sig) { stop_generation = 1; }
#endif

long time_in_ms() {
    struct timespec time; clock_gettime_impl(0, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

/* ---------------------------------------------------------------------------- */
/* MODEL STRUCTS */
/* ---------------------------------------------------------------------------- */

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
    int pos, i;
    for (pos = 0; pos < alloc_steps; pos++) {
        for (i = 0; i < head_size; i += 2) {
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
    t->s = (float*)ptr; 
    #if defined(NEEDS_BSWAP)
    {
        int gs = (group_size > 0) ? group_size : 32;
        int n_scales = numel / gs;
        int i;
        for(i=0; i<n_scales; i++) t->s[i] = bswap_float(t->s[i]);
        ptr += n_scales * sizeof(float);
    }
    #else
    ptr += (numel / group_size) * sizeof(float);
    #endif
    *ptr_ref = ptr;
}

void load_weights(Transformer* t, int shared_weights) {
    Config* p = &t->config; TransformerWeights* w = &t->weights; char* ptr = (char*)t->data; 
    unsigned long long dim = p->dim; unsigned long long att_dim = p->n_heads * p->head_size; 
    unsigned long long kv_dim = p->n_kv_heads * p->head_size; unsigned long long hidden_dim = p->hidden_dim;
    int l;
    float *fptr; int num_f;
    #if defined(NEEDS_BSWAP)
    int z;
    #endif

    ptr += 256; /* Skip header */
    
    w->rms_att_weight = (float*)ptr; ptr += p->n_layers * dim * sizeof(float);
    w->rms_ffn_weight = (float*)ptr; ptr += p->n_layers * dim * sizeof(float);
    w->rms_final_weight = (float*)ptr; ptr += dim * sizeof(float);

    #if defined(NEEDS_BSWAP)
        fptr = w->rms_att_weight; num_f = p->n_layers * p->dim; for(z=0;z<num_f;z++) fptr[z] = bswap_float(fptr[z]);
        fptr = w->rms_ffn_weight; num_f = p->n_layers * p->dim; for(z=0;z<num_f;z++) fptr[z] = bswap_float(fptr[z]);
        fptr = w->rms_final_weight; num_f = p->dim; for(z=0;z<num_f;z++) fptr[z] = bswap_float(fptr[z]);
    #endif

    w->wq = malloc(p->n_layers * sizeof(QuantizedTensor)); w->wk = malloc(p->n_layers * sizeof(QuantizedTensor));
    w->wv = malloc(p->n_layers * sizeof(QuantizedTensor)); w->wo = malloc(p->n_layers * sizeof(QuantizedTensor));
    w->w1 = malloc(p->n_layers * sizeof(QuantizedTensor)); w->w2 = malloc(p->n_layers * sizeof(QuantizedTensor));
    w->w3 = malloc(p->n_layers * sizeof(QuantizedTensor));

    init_quantized_tensor(&w->token_embedding_table, &ptr, p->vocab_size * dim, t->group_size);
    for(l=0; l<p->n_layers; l++) init_quantized_tensor(&w->wq[l], &ptr, dim * att_dim, t->group_size);
    for(l=0; l<p->n_layers; l++) init_quantized_tensor(&w->wk[l], &ptr, dim * kv_dim, t->group_size);
    for(l=0; l<p->n_layers; l++) init_quantized_tensor(&w->wv[l], &ptr, dim * kv_dim, t->group_size);
    for(l=0; l<p->n_layers; l++) init_quantized_tensor(&w->wo[l], &ptr, att_dim * dim, t->group_size);
    for(l=0; l<p->n_layers; l++) init_quantized_tensor(&w->w1[l], &ptr, dim * hidden_dim, t->group_size);
    for(l=0; l<p->n_layers; l++) init_quantized_tensor(&w->w2[l], &ptr, hidden_dim * dim, t->group_size);
    for(l=0; l<p->n_layers; l++) init_quantized_tensor(&w->w3[l], &ptr, dim * hidden_dim, t->group_size);
    if (shared_weights) w->wcls = w->token_embedding_table; else init_quantized_tensor(&w->wcls, &ptr, p->dim * p->vocab_size, t->group_size);
}

void build_transformer(Transformer *t, char* checkpoint_path, int steps) {
    FILE *file = fopen(checkpoint_path, "rb"); if (!file) { fprintf(stderr, "File not found: %s\n", checkpoint_path); exit(1); }
    uint32_t magic; fread(&magic, sizeof(uint32_t), 1, file);
    int version; fread(&version, sizeof(int), 1, file);
    
    fread(&t->config, sizeof(int) * 8, 1, file);
    #if defined(NEEDS_BSWAP)
    {
        int *pConf = (int*)&t->config;
        int i; for(i=0; i<8; i++) pConf[i] = bswap32(pConf[i]);
    }
    #endif

    if (t->config.head_size == 0) t->config.head_size = t->config.dim / t->config.n_heads;

    uint8_t shared; fread(&shared, sizeof(uint8_t), 1, file);
    fread(&t->group_size, sizeof(int), 1, file); 
    
    #if defined(NEEDS_BSWAP)
    if (t->group_size > 10000 || t->group_size < 0) t->group_size = bswap32(t->group_size);
    #endif
    
    fclose(file);

#if defined _WIN32
    t->hFile = CreateFileA(checkpoint_path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (t->hFile == INVALID_HANDLE_VALUE) exit(1);
    t->file_size = GetFileSize(t->hFile, NULL);
    t->hMapping = CreateFileMapping(t->hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (t->hMapping == NULL) exit(1);
    t->data = MapViewOfFile(t->hMapping, FILE_MAP_READ, 0, 0, 0);
    if (t->data == NULL) exit(1);
#else
    #if defined(__APPLE__) || defined(NEEDS_BSWAP)
        {
            FILE* f; struct stat sb; 
            if (stat(checkpoint_path, &sb) == -1) exit(1);
            t->file_size = sb.st_size;
            t->data = malloc_aligned(t->file_size);
            if (!t->data) { fprintf(stderr, "Malloc failed.\n"); exit(1); }
            f = fopen(checkpoint_path, "rb");
            fread(t->data, 1, t->file_size, f);
            fclose(f);
            t->fd = -1;
        }
    #else
        t->fd = open(checkpoint_path, O_RDONLY);
        struct stat sb; fstat(t->fd, &sb); t->file_size = sb.st_size;
        t->data = mmap(NULL, t->file_size, PROT_READ, MAP_PRIVATE, t->fd, 0);
    #endif
#endif
    load_weights(t, shared); malloc_run_state(&t->state, &t->config, steps);
}

void free_transformer(Transformer* t) {
    free(t->weights.wq); free(t->weights.wk); free(t->weights.wv); free(t->weights.wo);
    free(t->weights.w1); free(t->weights.w2); free(t->weights.w3);
#if defined _WIN32
    if (t->data) UnmapViewOfFile(t->data); if (t->hMapping) CloseHandle(t->hMapping); if (t->hFile != INVALID_HANDLE_VALUE) CloseHandle(t->hFile);
#else
    if (t->fd == -1 && t->data) free_aligned(t->data);
    else if (t->data != MAP_FAILED) munmap(t->data, t->file_size);
    if (t->fd != -1) close(t->fd);
#endif
    free_run_state(&t->state);
}

/* ---------------------------------------------------------------------------- */
/* MATH KERNELS */
/* ---------------------------------------------------------------------------- */

void rmsnorm(float* o, float* x, float* weight, int size) {
    #if defined(__ALTIVEC__)
        float ss = 0.0f; int i = 0;
        vector float sum_v = vec_splats(0.0f);
        vector float zero_v = vec_splats(0.0f);
        for (; i <= size - 4; i += 4) {
            vector float xv = vec_ld(0, &x[i]);
            sum_v = vec_madd(xv, xv, sum_v);
        }
        {
            float temp[4] __attribute__((aligned(16)));
            union { vector float v; float f[4]; } u;
            vec_ste(sum_v, 0, temp); vec_ste(sum_v, 4, &temp[1]);
            vec_ste(sum_v, 8, &temp[2]); vec_ste(sum_v, 12, &temp[3]);
            u.v = sum_v; ss = u.f[0] + u.f[1] + u.f[2] + u.f[3];
        }
        for (; i < size; i++) { ss += x[i] * x[i]; }
        ss /= size; ss += 1e-5f; ss = 1.0f / sqrtf(ss);
        {
            vector float ss_v = vec_splats(ss);
            i = 0;
            for (; i <= size - 4; i += 4) {
                vector float xv = vec_ld(0, &x[i]);
                vector unsigned char w_raw = vec_load_unaligned((unsigned char*)&weight[i]);
                vector float wv = (vector float)w_raw;
                vector float res = vec_madd(wv, vec_madd(xv, ss_v, zero_v), zero_v);
                vec_st(res, 0, &o[i]);
            }
        }
        for (; i < size; i++) { o[i] = weight[i] * (ss * x[i]); }
    
    #elif defined(__3dNOW__)
        float ss = 0.0f; int i = 0; 
        __m64 sum_v = _m_from_int(0); 
        for (; i <= size - 2; i += 2) { 
            __m64 x_v = *(__m64*)&x[i];       
            sum_v = _m_pfadd(sum_v, _m_pfmul(x_v, x_v));
        }
        sum_v = _m_pfacc(sum_v, sum_v); float temp_ss; *(__m64*)&temp_ss = sum_v; ss = temp_ss; _m_femms();
        for (; i < size; i++) { ss += x[i] * x[i]; }
        ss /= size; ss += 1e-5f; ss = 1.0f / sqrtf(ss);
        __m64 ss_v = _m_from_float(ss); ss_v = _m_punpckldq(ss_v, ss_v); 
        for (i = 0; i <= size - 2; i += 2) {
            __m64 w_v = *(__m64*)&weight[i]; __m64 x_v = *(__m64*)&x[i];
            *(__m64*)&o[i] = _m_pfmul(w_v, _m_pfmul(ss_v, x_v));
        }
        _m_femms(); for (; i < size; i++) { o[i] = weight[i] * (ss * x[i]); }

    #elif defined(__SSE__)
        float ss = 0.0f; int i = 0; __m128 sum_v = _mm_setzero_ps();
        for (; i <= size - 4; i += 4) { __m128 x_v = _mm_load_ps(&x[i]); sum_v = _mm_add_ps(sum_v, _mm_mul_ps(x_v, x_v)); }
        float temp[4]; _mm_storeu_ps(temp, sum_v); ss = temp[0] + temp[1] + temp[2] + temp[3];
        for (; i < size; i++) { ss += x[i] * x[i]; }
        ss /= size; ss += 1e-5f; __m128 ss_v = _mm_load_ss(&ss); ss_v = _mm_rsqrt_ss(ss_v); _mm_store_ss(&ss, ss_v);
        for (int j = 0; j < size; j++) { o[j] = weight[j] * (ss * x[j]); }
    #else
        float ss = 0.0f; int j; for (j = 0; j < size; j++) { ss += x[j] * x[j]; } ss /= size; ss += 1e-5f; ss = 1.0f / sqrtf(ss); for (j = 0; j < size; j++) { o[j] = weight[j] * (ss * x[j]); }
    #endif
}

void softmax(float* x, int size) {
    float max_val = x[0]; float sum = 0.0f; int i;
    for (i = 1; i < size; i++) { if (x[i] > max_val) max_val = x[i]; }
    for (i = 0; i < size; i++) { x[i] = expf(x[i] - max_val); sum += x[i]; }
    for (i = 0; i < size; i++) { x[i] /= sum; }
}

void matmul_q8(float* xout, float* x, QuantizedTensor* qt, int n, int d, int group_size) {
    #if defined(__ALTIVEC__)
    if (group_size % 4 == 0 && group_size >= 16) {
        int i; 
        vector float zero_v = vec_splats(0.0f);
        for (i = 0; i < d; i++) {
            int32_t in = i * n; float* s_ptr = &qt->s[in / group_size]; int8_t* w_ptr = &qt->q[in];
            vector float v_sum = vec_splats(0.0f);
            int j = 0;
            for (; j < n; j += group_size) {
                float scale = *s_ptr++; vector float v_scale = vec_splats(scale);
                int k;
                for (k = 0; k < group_size; k += 16) {
                    vector unsigned char raw_w = vec_load_unaligned((unsigned char*)&w_ptr[j+k]);
                    vector signed char v_w = (vector signed char)raw_w;
                    vector signed short v_w_h = vec_unpackh(v_w); vector signed short v_w_l = vec_unpackl(v_w);
                    vector signed int v_w_0 = vec_unpackh(v_w_h); vector signed int v_w_1 = vec_unpackl(v_w_h);
                    vector signed int v_w_2 = vec_unpackh(v_w_l); vector signed int v_w_3 = vec_unpackl(v_w_l);
                    vector float vf_w_0 = vec_ctf(v_w_0, 0); vector float vf_w_1 = vec_ctf(v_w_1, 0);
                    vector float vf_w_2 = vec_ctf(v_w_2, 0); vector float vf_w_3 = vec_ctf(v_w_3, 0);
                    vector float vf_x_0 = vec_ld(0, &x[j+k]); vector float vf_x_1 = vec_ld(16, &x[j+k]);
                    vector float vf_x_2 = vec_ld(32, &x[j+k]); vector float vf_x_3 = vec_ld(48, &x[j+k]);
                    vf_w_0 = vec_madd(vf_w_0, v_scale, zero_v); vf_w_1 = vec_madd(vf_w_1, v_scale, zero_v);
                    vf_w_2 = vec_madd(vf_w_2, v_scale, zero_v); vf_w_3 = vec_madd(vf_w_3, v_scale, zero_v);
                    v_sum = vec_madd(vf_w_0, vf_x_0, v_sum); v_sum = vec_madd(vf_w_1, vf_x_1, v_sum);
                    v_sum = vec_madd(vf_w_2, vf_x_2, v_sum); v_sum = vec_madd(vf_w_3, vf_x_3, v_sum);
                }
            }
            { union { vector float v; float f[4]; } u; u.v = v_sum; xout[i] = u.f[0] + u.f[1] + u.f[2] + u.f[3]; }
        }
        return;
    }

    #elif defined(__SSE__)
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
    return;

    #elif defined(__3dNOW__)
    for (int i = 0; i < d; i++) {
        __m64 row_acc = _m_from_int(0); int32_t in = i * n; float* s_ptr = &qt->s[in / group_size]; int8_t* w_ptr = &qt->q[in];
        for (int j = 0; j < n; j += group_size) {
            float scale = *s_ptr++; __m64 scale_v = _m_from_float(scale); scale_v = _m_punpckldq(scale_v, scale_v); 
            for (int k = 0; k < group_size; k += 2) {
                int32_t i0 = w_ptr[j+k]; int32_t i1 = w_ptr[j+k+1];
                __m64 w_v = _m_pi2fd(_mm_set_pi32(i1, i0)); __m64 x_v = *(__m64*)&x[j+k];
                row_acc = _m_pfadd(row_acc, _m_pfmul(x_v, _m_pfmul(w_v, scale_v)));
            }
        }
        row_acc = _m_pfacc(row_acc, row_acc); float res; *(__m64*)&res = row_acc; xout[i] = res;
    }
    _m_femms();
    return;

    #elif defined(__MMX__)
    /* 
       Dynamic Fixed-Point Quantization for MMX - High Precision
       Quantizes float inputs (x) to full-range int16 to minimize divergence from FPU output.
    */
    {
        int16_t x_quant[4096]; 
        float x_scales[512];   
        int32_t blk_sums[512]; 
        int j, k;

        // 1. FPU Phase: Pre-quantize input
        // Optimization: Use ~15-bit range (32000) instead of 7-bit (127) for much higher precision.
        for(j = 0; j < n; j += group_size) {
            float max_val = 1e-9f; // Prevent divide-by-zero
            int blk_limit = (j + group_size > n) ? n - j : group_size;
            
            // Find max absolute value in block
            for(k = 0; k < blk_limit; k++) {
                float v = x[j+k]; if(v < 0) v = -v; 
                if(v > max_val) max_val = v;
            }
            
            // Use 32000 to leave headroom for rounding overshoot
            float scale = 32000.0f / max_val; 
            float inv_scale = max_val / 32000.0f;
            
            x_scales[j / group_size] = inv_scale;

            for(k = 0; k < blk_limit; k++) {
                // Rounding (v + 0.5) is critical for accuracy vs simple truncation
                float val = x[j+k] * scale;
                x_quant[j+k] = (int16_t)(val + (val >= 0.0f ? 0.5f : -0.5f));
            }
        }

        // 2. Row Loop
        for (int i = 0; i < d; i++) {
            int32_t in = i * n;
            int8_t* w_row_start = &qt->q[in];
            int blk_idx = 0;

            // MMX Phase: Integer Dot Products
            for (j = 0; j < n; j += group_size) {
                int8_t* w_ptr = w_row_start + j;
                int blk_limit = (j + group_size > n) ? n - j : group_size;
                
                __m64 sum_v = _mm_setzero_si64();
                __m64 zero_v = _mm_setzero_si64();

                for (k = 0; k <= blk_limit - 4; k += 4) {
                    // Load weights (32-bit), unpack to 16-bit
                    int32_t w_raw = *(int32_t*)&w_ptr[k];
                    __m64 w_v = _mm_cvtsi32_si64(w_raw);
                    __m64 mask = _mm_cmpgt_pi8(zero_v, w_v);
                    __m64 w_16 = _mm_unpacklo_pi8(w_v, mask); 
                    
                    // Load input (64-bit, 4x int16)
                    __m64 x_v = *(__m64*)&x_quant[j+k];

                    // Multiply-Add: 16-bit * 16-bit -> 32-bit accumulation
                    sum_v = _mm_add_pi32(sum_v, _mm_madd_pi16(w_16, x_v));
                }

                // Horizontal sum
                __m64 high = _mm_srli_si64(sum_v, 32);
                sum_v = _mm_add_pi32(sum_v, high);
                int32_t blk_sum = _mm_cvtsi64_si32(sum_v);

                // Scalar cleanup
                for (; k < blk_limit; k++) {
                    blk_sum += (int32_t)w_ptr[k] * (int32_t)x_quant[j+k];
                }
                
                blk_sums[blk_idx++] = blk_sum;
            }

            // Clear MMX state before FPU operations
            _mm_empty();

            // FPU Phase: Accumulate with scales
            float val = 0.0f;
            float* s_ptr = &qt->s[in / group_size];
            for (int b = 0; b < blk_idx; b++) {
                 // Reconstruct: sum * weight_scale * activation_scale
                 val += (float)blk_sums[b] * s_ptr[b] * x_scales[b];
            }
            xout[i] = val;
        }
    }
    return;
    #endif

    /* Scalar Fallback */
    {
        int i;
        for (i = 0; i < d; i++) {
            float val = 0.0f; int32_t in = i * n; float* s_ptr = &qt->s[in / group_size]; int8_t* w_ptr = &qt->q[in];
            int j;
            for (j = 0; j < n; j += group_size) {
                float scale = *s_ptr++;
                int k;
                for (k = 0; k < group_size && (j+k) < n; k++) val += ((float)w_ptr[j+k] * scale) * x[j+k];
            }
            xout[i] = val;
        }
    }
}

float* forward(Transformer* t, int token, int pos, int stride_steps) {
    Config* p = &t->config; TransformerWeights* w = &t->weights; RunState* s = &t->state;
    float *x = s->x; int dim = p->dim; int kv_dim = p->n_kv_heads * p->head_size; int gs = t->group_size;
    int offset = token * dim;
    unsigned long long l; int i, h;

    for (i = 0; i < dim; i++) x[i] = (float)w->token_embedding_table.q[offset + i] * w->token_embedding_table.s[offset/gs + i/gs];

    for(l = 0; l < p->n_layers; l++) {
        int loff;
        rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);
        loff = l * stride_steps * kv_dim; s->k = s->key_cache + loff + pos * kv_dim; s->v = s->value_cache + loff + pos * kv_dim;
        matmul_q8(s->q, s->xb, &w->wq[l], dim, p->n_heads * p->head_size, gs);
        matmul_q8(s->k, s->xb, &w->wk[l], dim, kv_dim, gs);
        matmul_q8(s->v, s->xb, &w->wv[l], dim, kv_dim, gs);
        
        #if defined(__3dNOW__)
        _m_femms();
        #endif

        for (i = 0; i < p->n_heads * p->head_size; i+=2) {
            int cidx = pos * (p->head_size / 2) + (i % p->head_size) / 2;
            float fcr = s->rope_cos[cidx], fci = s->rope_sin[cidx];
            float v0 = s->q[i], v1 = s->q[i+1]; s->q[i] = v0*fcr-v1*fci; s->q[i+1] = v0*fci+v1*fcr;
            if (i < kv_dim) { v0 = s->k[i]; v1 = s->k[i+1]; s->k[i] = v0*fcr-v1*fci; s->k[i+1] = v0*fci+v1*fcr; }
        }

        for (h = 0; h < p->n_heads; h++) {
            float* q = s->q + h * p->head_size; float* att = s->att + h * stride_steps; 
            int t_step;
            for (t_step = 0; t_step <= pos; t_step++) {
                float* k = s->key_cache + loff + t_step * kv_dim + (h / (p->n_heads / p->n_kv_heads)) * p->head_size;
                float score = 0.0f; int k_idx;
                for (k_idx = 0; k_idx < p->head_size; k_idx++) { score += q[k_idx] * k[k_idx]; }
                att[t_step] = score / sqrtf(p->head_size);
            }
            softmax(att, pos + 1);
            {
                float* xb = s->xb + h * p->head_size; memset(xb, 0, p->head_size * sizeof(float));
                for (t_step = 0; t_step <= pos; t_step++) {
                    float* v = s->value_cache + loff + t_step * kv_dim + (h / (p->n_heads / p->n_kv_heads)) * p->head_size;
                    float a = att[t_step]; int v_idx;
                    for (v_idx = 0; v_idx < p->head_size; v_idx++) { xb[v_idx] += a * v[v_idx]; }
                }
            }
        }
        matmul_q8(s->xb2, s->xb, &w->wo[l], p->n_heads * p->head_size, dim, gs);
        for (i = 0; i < dim; i++) { x[i] += s->xb2[i]; }
        rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);
        matmul_q8(s->hb, s->xb, &w->w1[l], dim, p->hidden_dim, gs);
        matmul_q8(s->hb2, s->xb, &w->w3[l], dim, p->hidden_dim, gs);
        
        #if defined(__3dNOW__)
        _m_femms();
        #endif
        
        for (i = 0; i < p->hidden_dim; i++) { float val = s->hb[i]; val = val / (1.0f + expf(-val)); s->hb[i] = val * s->hb2[i]; }
        matmul_q8(s->xb, s->hb, &w->w2[l], p->hidden_dim, dim, gs);
        for (i = 0; i < dim; i++) { x[i] += s->xb[i]; }
    }
    rmsnorm(x, x, w->rms_final_weight, dim);
    matmul_q8(s->logits, x, &w->wcls, p->dim, p->vocab_size, gs);
    return s->logits;
}

/* ---------------------------------------------------------------------------- */
/* TOKENIZER & SAMPLER */
/* ---------------------------------------------------------------------------- */
typedef struct { char *str; int id; } TokenIndex;
typedef struct { char** vocab; float* vocab_scores; TokenIndex *sorted_vocab; int vocab_size; unsigned int max_token_length; unsigned char byte_pieces[512]; } Tokenizer;
typedef struct { float prob; int index; } ProbIndex;
typedef struct { int vocab_size; ProbIndex* probindex; float temperature; float topp; int topk; unsigned long long rng_state; } Sampler;

int compare_tokens(const void *a, const void *b) { return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str); }
char* decode(Tokenizer* t, int prev_token, int token) { 
    char *piece = t->vocab[token]; unsigned char byte_val;
    if (prev_token == 1 && piece[0] == ' ') piece++; 
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) piece = (char*)t->byte_pieces + byte_val * 2; 
    return piece; 
}

void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    char* str_buffer; size_t str_len = 0; char *c; int i;
    if (text == NULL) exit(1);
    if (!t->sorted_vocab) { 
        t->sorted_vocab = malloc(t->vocab_size * sizeof(TokenIndex)); 
        for (i = 0; i < t->vocab_size; i++) { t->sorted_vocab[i].str = t->vocab[i]; t->sorted_vocab[i].id = i; } 
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens); 
    }
    str_buffer = malloc((t->max_token_length*2 +1 +2) * sizeof(char));
    *n_tokens = 0; if (bos) tokens[(*n_tokens)++] = 1;
    
    for (c = text; *c != '\0'; c++) {
        if ((*c & 0xC0) != 0x80) str_len = 0;
        str_buffer[str_len++] = *c; str_buffer[str_len] = '\0';
        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) continue;
        {
            TokenIndex tok = { 0 }; TokenIndex *res; tok.str = str_buffer;
            res = bsearch(&tok, t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
            if (res) tokens[(*n_tokens)++] = res->id; 
            else for (i=0; i < str_len; i++) tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
        }
        str_len = 0;
    }
    while (1) {
        float best_score = -1e10; int best_id = -1; int best_idx = -1;
        for (i=0; i < (*n_tokens-1); i++) {
            TokenIndex tok = { 0 }; TokenIndex *res;
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            tok.str = str_buffer;
            res = bsearch(&tok, t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
            if (res && t->vocab_scores[res->id] > best_score) { best_score = t->vocab_scores[res->id]; best_id = res->id; best_idx = i; }
        }
        if (best_idx == -1) break;
        tokens[best_idx] = best_id; for (i = best_idx+1; i < (*n_tokens-1); i++) tokens[i] = tokens[i+1]; (*n_tokens)--;
    }
    if (eos) tokens[(*n_tokens)++] = 2; free(str_buffer);
}

void build_tokenizer(Tokenizer* t, char* path, int vs) {
    int i; FILE *file;
    t->vocab_size = vs; t->vocab = malloc(vs * sizeof(char*)); t->vocab_scores = malloc(vs * sizeof(float)); t->sorted_vocab = NULL;
    for (i = 0; i < 256; i++) { t->byte_pieces[i*2] = (unsigned char)i; t->byte_pieces[i*2+1] = '\0'; }
    file = fopen(path, "rb"); if (!file) exit(1);
    fread(&t->max_token_length, sizeof(int), 1, file);
    #if defined(NEEDS_BSWAP)
    t->max_token_length = bswap32(t->max_token_length);
    #endif
    for (i = 0; i < vs; i++) { 
        fread(t->vocab_scores + i, sizeof(float), 1, file); 
        #if defined(NEEDS_BSWAP)
        t->vocab_scores[i] = bswap_float(t->vocab_scores[i]);
        #endif
        int len; if (fread(&len, sizeof(int), 1, file) != 1) { break; }
        #if defined(NEEDS_BSWAP)
        len = bswap32(len);
        #endif
        t->vocab[i] = malloc(len + 1); fread(t->vocab[i], len, 1, file); t->vocab[i][len] = '\0'; 
    }
    fclose(file);
}

static int compare_probindex(const void* a, const void* b) { 
    ProbIndex* a_ = (ProbIndex*) a; ProbIndex* b_ = (ProbIndex*) b; 
    if (a_->prob > b_->prob) return -1; if (a_->prob < b_->prob) return 1; return 0; 
}

int sample(Sampler* s, float* logits) {
    int i; int q;
    if (s->temperature == 0.0f) { int max_i=0; float max_p=logits[0]; for(i=1;i<s->vocab_size;i++) if(logits[i]>max_p){max_i=i;max_p=logits[i];} return max_i; }
    for (q=0; q<s->vocab_size; q++) logits[q] /= s->temperature; 
    softmax(logits, s->vocab_size);
    if (s->topk > 0 && s->topk < s->vocab_size) {
        float topk_sum = 0.0f; float coin; float cdf = 0.0f;
        for (i = 0; i < s->vocab_size; i++) { s->probindex[i].index = i; s->probindex[i].prob = logits[i]; }
        qsort(s->probindex, s->vocab_size, sizeof(ProbIndex), compare_probindex);
        for (i = 0; i < s->topk; i++) topk_sum += s->probindex[i].prob;
        coin = (float)rand() / (float)RAND_MAX * topk_sum; 
        for (i = 0; i < s->topk; i++) { cdf += s->probindex[i].prob; if (coin < cdf) return s->probindex[i].index; }
        return s->probindex[s->topk-1].index;
    } else {
        float coin = (float)rand() / (float)RAND_MAX;
        float cdf = 0.0f; for (i = 0; i < s->vocab_size; i++) { cdf += logits[i]; if (coin < cdf) return i; }
        return s->vocab_size - 1;
    }
}

/* ---------------------------------------------------------------------------- */
/* TUI ENGINE (Unified: Windows 95 API & POSIX ANSI) */
/* ---------------------------------------------------------------------------- */

#if defined _WIN32
    #define MAX_HIST 2000
    #define SCREEN_W 100
    typedef struct {
        char lines[MAX_HIST][SCREEN_W]; int line_lens[MAX_HIST];
        int count; int view_offset; char input[1024]; int input_len;
        int cols, rows; HANDLE hOut; HANDLE hIn; int mode; 
    } TUI;
    TUI tui;

    void tui_init() {
        tui.hOut = GetStdHandle(STD_OUTPUT_HANDLE); tui.hIn = GetStdHandle(STD_INPUT_HANDLE);
        DWORD mode; GetConsoleMode(tui.hIn, &mode); SetConsoleMode(tui.hIn, mode | ENABLE_MOUSE_INPUT | ENABLE_WINDOW_INPUT);
        SetConsoleCtrlHandler(CtrlHandler, TRUE);
        CONSOLE_SCREEN_BUFFER_INFO csbi; GetConsoleScreenBufferInfo(tui.hOut, &csbi);
        tui.cols = csbi.srWindow.Right - csbi.srWindow.Left + 1; tui.rows = csbi.srWindow.Bottom - csbi.srWindow.Top + 1;
        if (tui.cols > SCREEN_W) tui.cols = SCREEN_W - 1; if (tui.cols < 20) tui.cols = 80; 
        tui.count = 0; tui.view_offset = 0; tui.input_len = 0; tui.input[0] = '\0'; tui.mode = 0;
    }
    void tui_newline() {
        if (tui.count >= MAX_HIST) {
            memmove(&tui.lines[0], &tui.lines[1], (MAX_HIST-1) * SCREEN_W); memmove(&tui.line_lens[0], &tui.line_lens[1], (MAX_HIST-1) * sizeof(int)); tui.count--;
        }
        tui.lines[tui.count][0] = '\0'; tui.line_lens[tui.count] = 0; tui.count++;
    }
    void tui_append(const char* txt) {
        const char* ptr = txt; if (tui.count == 0) tui_newline(); 
        while (*ptr) {
            int tail = tui.count - 1; int cur_len = tui.line_lens[tail]; int avail = (tui.cols - 1) - cur_len;
            if (*ptr == '\n') { tui_newline(); ptr++; continue; }
            if (avail > 0) { tui.lines[tail][cur_len] = *ptr; tui.lines[tail][cur_len+1] = '\0'; tui.line_lens[tail]++; ptr++; } else { tui_newline(); }
        }
    }
    void tui_draw() {
        int i, c, r;
        int input_row, screen_y, hist_idx, base_idx, knob;
        float scroll_pos;
        CHAR_INFO* buf;
        COORD s, c_coord, cur; SMALL_RECT sr;
        buf = malloc(sizeof(CHAR_INFO) * tui.cols * tui.rows);
        if (!buf) return;
        for(i=0; i<tui.cols*tui.rows; i++) { buf[i].Char.AsciiChar=' '; buf[i].Attributes=COL_BG; }
        if (tui.count < tui.rows - 1) input_row = tui.count; else input_row = tui.rows - 1;
        screen_y = input_row - 1; hist_idx = tui.count - 1 - tui.view_offset;
        while (screen_y >= 0) {
            if (hist_idx >= 0 && hist_idx < tui.count) {
                char* l = tui.lines[hist_idx]; int len = tui.line_lens[hist_idx]; int attr = COL_TEXT;
                if (strncmp(l, ">>>", 3) == 0) attr = COL_PROMPT; else if (l[0] == '/') attr = COL_INFO; 
                for (c=0; c<len; c++) { int idx = screen_y*tui.cols + c; buf[idx].Char.AsciiChar = l[c]; buf[idx].Attributes = attr; }
            }
            screen_y--; hist_idx--;
        }
        base_idx = input_row * tui.cols;
        if (tui.mode == 0) {
            const char* prompt = ">>> "; for(i=0; i<4; i++) { buf[base_idx+i].Char.AsciiChar = prompt[i]; buf[base_idx+i].Attributes = COL_PROMPT; }
            if (tui.input_len == 0) { const char* ph = "Send a message (/? for help)"; for (i=0; ph[i]; i++) { buf[base_idx+4+i].Char.AsciiChar = ph[i]; buf[base_idx+4+i].Attributes = COL_PLACEHOLDER; } } 
            else { for (i=0; i<tui.input_len && (i+4 < tui.cols); i++) { buf[base_idx+4+i].Char.AsciiChar = tui.input[i]; buf[base_idx+4+i].Attributes = COL_TEXT; } }
        }
        scroll_pos = 1.0f; if (tui.count > tui.rows) { scroll_pos = 1.0f - ((float)tui.view_offset / (float)(tui.count - tui.rows + 1)); }
        if (scroll_pos < 0) scroll_pos = 0; if (scroll_pos > 1) scroll_pos = 1; 
        knob = (int)((tui.rows - 1) * scroll_pos); if (knob >= tui.rows) knob = tui.rows - 1;
        for(r=0; r<tui.rows; r++) { int idx = r*tui.cols + (tui.cols-1); buf[idx].Attributes = COL_BAR; buf[idx].Char.AsciiChar = (r==knob)?219:176; }
        s.X = tui.cols; s.Y = tui.rows; c_coord.X = 0; c_coord.Y = 0; 
        sr.Left = 0; sr.Top = 0; sr.Right = tui.cols-1; sr.Bottom = tui.rows-1;
        WriteConsoleOutput(tui.hOut, buf, s, c_coord, &sr);
        if (tui.mode == 0) { cur.Y = input_row; cur.X = 4 + tui.input_len; if (tui.input_len==0) cur.X = 4; } 
        else { 
            int cursor_y = input_row - 1 - tui.view_offset; 
            if (cursor_y >= 0 && cursor_y < tui.rows - 1) { cur.Y = cursor_y; cur.X = tui.line_lens[tui.count-1]; } 
            else { cur.Y = tui.rows-1; cur.X = 0; } 
        }
        SetConsoleCursorPosition(tui.hOut, cur); free(buf);
    }
#else
    /* POSIX / Mac OS X TUI Implementation using ANSI Escape Codes */
    struct termios orig_termios;
    void disable_raw_mode() { tcsetattr(STDIN_FILENO, TCSAFLUSH, &orig_termios); }
    void enable_raw_mode() {
        struct termios raw;
        tcgetattr(STDIN_FILENO, &orig_termios);
        atexit(disable_raw_mode);
        raw = orig_termios;
        raw.c_lflag &= ~(ECHO | ICANON);
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);
    }
    void tui_init() {
        signal(SIGINT, handle_sigint);
        printf("%s", ANSI_CLEAR_SCREEN);
    }
    void tui_draw() { } 
#endif

/* ---------------------------------------------------------------------------- */
/* GENERATION & CHAT LOGIC */
/* ---------------------------------------------------------------------------- */

void generate(Transformer *t, Tokenizer *tok, Sampler *samp, char *prompt, int steps) {
    int num_prompt_tokens = 0; int* prompt_tokens; long start = 0; int next; int token; int pos = 0;
    float* logits;
    if (prompt == NULL) prompt = "";
    prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int));
    encode(tok, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    token = prompt_tokens[0]; 
    while (pos < steps) {
        if (stop_generation) { printf("\n^C Interrupted"); break; }
        logits = forward(t, token, pos, steps);
        if (pos < num_prompt_tokens - 1) next = prompt_tokens[pos + 1]; else next = sample(samp, logits);
        pos++;
        if (next == 2) break; 
        { char* piece = decode(tok, token, next); if (piece) { printf("%s", piece); fflush(stdout); } }
        token = next;
        if (start == 0) start = time_in_ms();
    }
    printf("\n");
    if (pos > 1) { long end = time_in_ms(); fprintf(stderr, "achieved tok/s: %f\n", (pos-1) / (double)(end-start)*1000); }
    free(prompt_tokens);
}

void chat_loop(Transformer *t, Tokenizer *tok, Sampler *samp, int n_ctx, int use_tui, char* cli_system_prompt) {
    int id_im_start = 1; int id_im_end = 2; int id_nl = 198; int id_system = 9690; int id_user = 4093; int id_ass1 = 520; int id_ass2 = 9531;   
    char* sys_prompt = "You are SmolLM, a helpful assistant.";
    int* tokens; int n_tok = 0, n_chunk = 0; int pos = 0; int initial_pos; int i; char input_buf[1024];
    int user_tokens[1024]; int n_user_tokens; int prompt_tokens[1024]; int n_prompt; int token;
    
    if (cli_system_prompt != NULL) sys_prompt = cli_system_prompt;
    tokens = malloc(n_ctx * sizeof(int));
    tokens[n_tok++] = id_im_start; tokens[n_tok++] = id_system; tokens[n_tok++] = id_nl;
    encode(tok, sys_prompt, 0, 0, tokens+n_tok, &n_chunk); n_tok += n_chunk;
    tokens[n_tok++] = id_im_end; tokens[n_tok++] = id_nl;
    for(i=0; i<n_tok; i++) forward(t, tokens[i], pos++, n_ctx);
    initial_pos = pos; 

    if (use_tui) { tui_init(); } else { printf(">>> "); fflush(stdout); }

    while(1) {
        #if defined _WIN32
        if (use_tui) {
            tui.mode = 0; tui.input[0] = '\0'; tui.input_len = 0; tui.view_offset = 0;
            {
                INPUT_RECORD ir[32]; DWORD n;
                while(1) {
                    tui_draw(); ReadConsoleInput(tui.hIn, ir, 1, &n);
                    if (ir[0].EventType == KEY_EVENT && ir[0].Event.KeyEvent.bKeyDown) {
                        char c = ir[0].Event.KeyEvent.uChar.AsciiChar; WORD vk = ir[0].Event.KeyEvent.wVirtualKeyCode;
                        if (c == 13) break; 
                        else if (c == 8) { if (tui.input_len > 0) tui.input[--tui.input_len] = '\0'; }
                        else if (c >= 32) { if (tui.input_len < 1000) { tui.input[tui.input_len++] = c; tui.input[tui.input_len] = '\0'; } }
                        else { 
                            if (vk == 33) { tui.view_offset += 5; } 
                            if (vk == 34) { tui.view_offset -= 5; if(tui.view_offset<0) tui.view_offset=0; } 
                        }
                    }
                }
            }
            strcpy(input_buf, tui.input); 
            { char hist[1050]; sprintf(hist, ">>> %s\n", input_buf); tui_append(hist); }
        } else 
        #else
        /* POSIX Raw Input Mode to replicate TUI feel */
        if (use_tui) {
            char c; int idx = 0; char* ph = "Send a message (/? for help)";
            printf("%s>>> %s", ANSI_COLOR_GREEN, ANSI_COLOR_RESET);
            /* Initially print placeholder */
            printf("%s%s%s", ANSI_COLOR_GRAY, ph, ANSI_COLOR_RESET);
            printf("\033[%dD", (int)strlen(ph)); fflush(stdout);
            
            enable_raw_mode();
            while(read(STDIN_FILENO, &c, 1) == 1) {
                if (c == 3) { stop_generation = 1; disable_raw_mode(); exit(0); }
                
                if (c == 127 || c == 8) { /* Backspace */
                    if (idx > 0) { 
                        idx--; printf("\b \b"); 
                        if (idx == 0) { 
                            /* Buffer empty, restore placeholder */
                            printf("%s%s%s", ANSI_COLOR_GRAY, ph, ANSI_COLOR_RESET);
                            printf("\033[%dD", (int)strlen(ph));
                        }
                        fflush(stdout); 
                    }
                } else if (c == '\n' || c == '\r') {
                    if (idx == 0) printf("%s", ANSI_CLEAR_LINE); /* Clear placeholder */
                    input_buf[idx] = 0; printf("\n"); break;
                } else if (c >= 32 && idx < 1023) {
                    if (idx == 0) printf("%s", ANSI_CLEAR_LINE); /* Clear placeholder on first char */
                    input_buf[idx++] = c; printf("%c", c); fflush(stdout);
                }
            }
            disable_raw_mode();
        } else
        #endif
        { if(!fgets(input_buf, 1024, stdin)) break; { size_t len=strlen(input_buf); if(len>0 && input_buf[len-1]=='\n') input_buf[len-1]=0; } }

        if (input_buf[0] == '/') {
            if (strncmp(input_buf, "/bye", 4)==0) break;
            if (strncmp(input_buf, "/clear", 6)==0) { pos = initial_pos; 
                 #if defined _WIN32
                 if (use_tui) { tui_append("Cleared session context\n"); } else
                 #else
                 printf("%sCleared session context%s\n", ANSI_COLOR_BLUE, ANSI_COLOR_RESET); 
                 #endif
                 if (!use_tui) printf("Cleared session context\n"); continue; 
            }
            if (strncmp(input_buf, "/?", 2)==0 || strncmp(input_buf, "/help", 5)==0) {
                 #if defined _WIN32
                 if (use_tui) {
                     tui_append("Available Commands:\n"); tui_append("  /set parameter [temp|top_k] <val>\n");
                     tui_append("  /clear\n"); tui_append("  /bye\n");
                 } else
                 #endif
                 {
                    #if !defined _WIN32
                    printf("%sAvailable Commands:%s\n", ANSI_COLOR_BLUE, ANSI_COLOR_RESET);
                    printf("  /set parameter [temp|top_k] <val>\n");
                    printf("  /clear\n");
                    printf("  /bye\n");
                    #else
                    printf("Commands: /set, /clear, /bye\n");
                    #endif
                 }
                 continue;
            }
             if (strncmp(input_buf, "/set parameter temperature", 26)==0) { samp->temperature = atof(input_buf + 27); continue; }
             if (strncmp(input_buf, "/set parameter temp", 19)==0) { samp->temperature = atof(input_buf + 20); continue; }
             if (strncmp(input_buf, "/set parameter top_k", 20)==0) { samp->topk = atoi(input_buf + 21); continue; }
        }

        stop_generation = 0;
        n_prompt = 0;
        prompt_tokens[n_prompt++] = id_im_start; prompt_tokens[n_prompt++] = id_user; prompt_tokens[n_prompt++] = id_nl;
        encode(tok, input_buf, 0, 0, user_tokens, &n_user_tokens);
        for(i=0; i<n_user_tokens; i++) prompt_tokens[n_prompt++] = user_tokens[i];
        prompt_tokens[n_prompt++] = id_im_end; prompt_tokens[n_prompt++] = id_nl;
        prompt_tokens[n_prompt++] = id_im_start; prompt_tokens[n_prompt++] = id_ass1; prompt_tokens[n_prompt++] = id_ass2; prompt_tokens[n_prompt++] = id_nl;
        for(i=0; i<n_prompt; i++) forward(t, prompt_tokens[i], pos++, n_ctx);
        token = prompt_tokens[n_prompt-1];

        #if defined _WIN32
        if (use_tui) tui.mode = 1; 
        #endif

        while (pos < n_ctx) {
            float* logits; int next; char* piece;
            if (stop_generation) {
                #if defined _WIN32
                if(use_tui) tui_append("^C Interrupted\n"); else
                #endif
                printf("\n^C Interrupted"); break;
            }
            logits = forward(t, token, pos, n_ctx);
            next = sample(samp, logits);
            pos++;
            if (next == id_im_end || next == 2) break; 
            piece = decode(tok, token, next);
            #if defined _WIN32
            if (use_tui) { 
                if(piece) { tui_append(piece); tui_draw(); } 
                { INPUT_RECORD ir; DWORD n; if (PeekConsoleInput(tui.hIn, &ir, 1, &n) && n > 0) {} } 
            } else 
            #endif
            { printf("%s", piece); fflush(stdout); }
            token = next;
        }
        #if defined _WIN32
        if (use_tui) { tui_append("\n"); tui_draw(); } else { printf("\n>>> "); }
        #else
        printf("\n"); 
        #endif
        fflush(stdout);
    }
    free(tokens);
}

int main(int argc, char *argv[]) {
    char *checkpoint = NULL; char *tokenizer = "tokenizer.bin";
    int steps = 512; float temp = 0.8f; float topp = 0.9f;
    char *prompt = NULL; char *mode = "chat"; char *sys_prompt = NULL;
    int i; 
    unsigned long long rng_seed = 0;

#if defined _WIN32
    LARGE_INTEGER qpc; QueryPerformanceCounter(&qpc); rng_seed = (unsigned long long)qpc.QuadPart;
#else
    rng_seed = (unsigned long long)time(NULL);
#endif

    if (argc >= 2) checkpoint = argv[1]; else { fprintf(stderr, "Usage: run <model.bin> [options]\n"); exit(1); }
    
    for (i = 2; i < argc; i+=2) {
        if (i + 1 >= argc || argv[i][0] != '-') break;
        if (argv[i][1] == 't') temp = atof(argv[i + 1]);
        else if (argv[i][1] == 'p') topp = atof(argv[i + 1]);
        else if (argv[i][1] == 'n') steps = atoi(argv[i + 1]);
        else if (argv[i][1] == 'z') tokenizer = argv[i + 1];
        else if (argv[i][1] == 'i') prompt = argv[i + 1];
        else if (argv[i][1] == 'm') mode = argv[i + 1];
        else if (argv[i][1] == 'y') sys_prompt = argv[i + 1];
        else if (argv[i][1] == 's') rng_seed = (unsigned long long)atol(argv[i + 1]);
    }
    
    srand((unsigned int)rng_seed); 

    {
        Transformer transformer; Tokenizer tok; Sampler samp; 
        build_transformer(&transformer, checkpoint, steps);
        build_tokenizer(&tok, tokenizer, transformer.config.vocab_size);
        
        samp.vocab_size = transformer.config.vocab_size; 
        samp.temperature = temp; samp.topp = topp; samp.topk = 40; samp.rng_state = rng_seed;
        samp.probindex = malloc(samp.vocab_size*sizeof(ProbIndex));
        
        if (strcmp(mode, "generate") == 0) generate(&transformer, &tok, &samp, prompt, steps);
        else chat_loop(&transformer, &tok, &samp, steps, 1, sys_prompt);
        
        free(samp.probindex); free_transformer(&transformer);
    }
    return 0;
}
