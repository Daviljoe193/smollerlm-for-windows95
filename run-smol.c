/* 
   Run-Smol: Atari Falcon030 Arena Edition (Fixed)
   
   Architecture:
   - Arena Memory Management: Prevents fragmentation on 14MB systems.
   - Distributed Ring Topology: Master -> Slave 1 -> Slave 2 -> Slave 3 -> Master.
   - 32-bit Compatibility: Enforced file header types.
   - Fault Tolerance: Clamps errors instead of dropping packets to prevent Ring Deadlock.
   - ALIGNMENT FIX: Manually aligns buffers to prevent Bus Errors.
   - LOADER FIX: Corrected weight loading order to match export script.
   
   Compile with: m68k-atari-mint-gcc runssmol.c -o smol30.tos -O3 -m68030 -m68881 -mhard-float -fomit-frame-pointer -funroll-loops -ffast-math -lm
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <osbind.h>

/* Stack size definition for Mint/TOS */
long _stksize = 65536; 

#define COMM_DEV 3  /* 3 = MIDI/Serial (Check your specific TOS mapping) */

/* ---------------------------------------------------------------------------- */
/* MEMORY ARENA                                                                 */
/* ---------------------------------------------------------------------------- */

typedef struct {
    uint8_t* base;
    size_t size;
    size_t offset;
} Arena;

void arena_init(Arena* a, size_t size) {
    /* Align request to 16 bytes */
    size = (size + 15) & ~15;
    printf("Arena Alloc: %ld KB... ", size/1024);
    
    /* Attempt allocation */
    a->base = malloc(size + 16);
    if (!a->base) {
        printf("FAILED! Not enough contiguous RAM.\n");
        (void)Cconin(); 
        exit(1);
    }
    
    /* Align memory base pointer for faster FPU access */
    uint32_t raw = (uint32_t)a->base;
    uint32_t aligned = (raw + 15) & ~15;
    a->base = (uint8_t*)aligned;
    a->size = size;
    a->offset = 0;
    
    /* Store original pointer behind for potential free (not used here) */
    uint32_t* ret = (uint32_t*)a->base;
    *(ret - 1) = raw;
    
    printf("OK @ $%p\n", a->base);
}

void* arena_alloc(Arena* a, size_t size, int zero) {
    /* Align all sub-allocations to 4 bytes */
    size_t aligned_size = (size + 3) & ~3;
    
    if (a->offset + aligned_size > a->size) {
        printf("\nArena Overflow! Needed %ld, have %ld left.\n", (long)size, (long)(a->size - a->offset));
        (void)Cconin(); 
        exit(1);
    }
    
    void* ptr = a->base + a->offset;
    if (zero) memset(ptr, 0, aligned_size);
    a->offset += aligned_size;
    return ptr;
}

/* Fallback for non-critical allocations (Tokenizer, etc) */
void* malloc_safe(size_t size) {
    void* p = malloc(size);
    if (!p) { 
        printf("\nOOM on malloc(%ld)\n", (long)size); 
        (void)Cconin(); 
        exit(1); 
    }
    return p;
}

/* ---------------------------------------------------------------------------- */
/* MATH HELPERS (Big Endian / Alignment)                                        */
/* ---------------------------------------------------------------------------- */

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

static inline float safe_load_float(float* ptr) {
    uint32_t addr = (uint32_t)ptr;
    if (addr & 3) { 
        uint32_t tmp; 
        memcpy(&tmp, ptr, 4); 
        return *(float*)&tmp; 
    } 
    return *ptr;
}

/* ---------------------------------------------------------------------------- */
/* HARDWARE INTERFACE                                                           */
/* ---------------------------------------------------------------------------- */

uint16_t* v_screen_base = NULL;
int term_width = 80; /* Default width */
int term_col = 0;    /* Current column for wrapping */

void detect_resolution() {
    /* 0 = ST Low (320x200), others usually 80 col */
    if (Getrez() == 0) term_width = 40; else term_width = 80; 
}

void init_video() { 
    v_screen_base = (uint16_t*)Physbase(); 
}

void update_status_led(uint16_t color) {
    int i, j;
    if (!v_screen_base) return;
    /* Draw a 4x4 pixel block in top left for status */
    for(i=0; i<4; i++) {
        for(j=0; j<4; j++) {
            v_screen_base[j + (i * 320)] = color; 
        }
    }
}

void print_token_wrapped(char* txt) {
    char* c = txt;
    while (*c) {
        if (*c == '\n') { putchar('\n'); term_col = 0; }
        else {
            putchar(*c); term_col++;
            if (term_col >= term_width) { putchar('\n'); term_col = 0; }
        }
        c++;
    }
    fflush(stdout);
}

int serial_has_data() { 
    return (Bconstat(COMM_DEV) != 0); 
}

void serial_putc(uint8_t c) { 
    Bconout(COMM_DEV, c); 
}

uint8_t serial_getc_yielding() {
    int loops = 0;
    while(!serial_has_data()) { 
        loops++;
        /* Blink heartbeat */
        if ((loops & 0xFFF) == 0) update_status_led(0x001F); 
        if ((loops & 0xFFF) == 0x800) update_status_led(0x0000); 
        
        }
    update_status_led(0x07E0); /* Green on Rx */
    return (uint8_t)(Bconin(COMM_DEV) & 0xFF);
}

void serial_read_buffer(void* buf, size_t size) {
    uint8_t* p = (uint8_t*)buf; 
    size_t i; 
    uint8_t sync = 0;
    
    /* Wait for Sync Byte 0x42 */
    while(sync != 0x42) { 
        sync = serial_getc_yielding(); 
    }
    
    /* Read Payload - no byte filtering; 0xFF is valid data */
    for(i=0; i<size; i++) { 
        p[i] = serial_getc_yielding(); 
    }
    update_status_led(0x0000);
}

void serial_send_buffer(void* buf, size_t size) {
    uint8_t* p = (uint8_t*)buf; 
    size_t i;
    update_status_led(0xF800); /* Red on Tx */
    serial_putc(0x42);         /* Sync Byte */
    for(i=0; i<size; i++) serial_putc(p[i]);
    update_status_led(0x0000);
}

/* ---------------------------------------------------------------------------- */
/* MODEL STRUCTURES                                                             */
/* ---------------------------------------------------------------------------- */

typedef struct { 
    int32_t dim; 
    int32_t hidden_dim; 
    int32_t n_layers; 
    int32_t n_heads; 
    int32_t n_kv_heads; 
    int32_t vocab_size; 
    int32_t seq_len; 
    int32_t head_size; 
} Config;

typedef struct { 
    int8_t* q; 
    float* s; 
} QuantizedTensor;

typedef struct { 
    QuantizedTensor token_embedding_table; 
    float* rms_att_weight; 
    float* rms_ffn_weight; 
    QuantizedTensor* wq; 
    QuantizedTensor* wk; 
    QuantizedTensor* wv; 
    QuantizedTensor* wo; 
    QuantizedTensor* w1; 
    QuantizedTensor* w2; 
    QuantizedTensor* w3; 
    float* rms_final_weight; 
    QuantizedTensor wcls; 
} TransformerWeights;

typedef struct { 
    float *x; 
    float *xb; 
    float *xb2; 
    float *hb; 
    float *hb2; 
    float *q; 
    float *k; 
    float *v; 
    float *att; 
    float *logits; 
    float* key_cache; 
    float* value_cache; 
    float* rope_cos; 
    float* rope_sin; 
} RunState;

typedef struct { 
    Config config; 
    TransformerWeights weights; 
    RunState state; 
    int32_t group_size; 
    int node_id; 
    int total_nodes; 
    int layer_start; 
    int layer_end; 
} Transformer;

/* Serial Packet Header */
typedef struct { 
    float temp; 
    float topp; 
    float topk; 
    int32_t pos; 
} PacketHeader;

/* ---------------------------------------------------------------------------- */
/* LOADING LOGIC                                                                */
/* ---------------------------------------------------------------------------- */

/* Arena-aware tensor loader */
void load_tensor_arena(QuantizedTensor* t, FILE* f, int numel, int group_size, int should_load, Arena* arena) {
    int gs = (group_size > 0) ? group_size : 32;
    size_t q_bytes = numel * sizeof(int8_t);
    size_t s_bytes = (numel / gs) * sizeof(float);
    int i;
    
    if (should_load) {
        /* Allocate from Arena */
        void* mem = arena_alloc(arena, q_bytes + s_bytes, 0);
        
        t->q = (int8_t*)mem;
        t->s = (float*)((uint8_t*)mem + q_bytes);
        
        fread(t->q, 1, q_bytes, f);
        fread(t->s, 1, s_bytes, f);
        
        int n_scales = numel / gs;
        for(i=0; i<n_scales; i++) t->s[i] = bswap_float(t->s[i]);
    } else {
        /* Skip if not owned by this node */
        fseek(f, q_bytes + s_bytes, SEEK_CUR);
        t->q = NULL; 
        t->s = NULL;
    }
}

void build_transformer(Transformer *t, char* path, int node_id, int total_nodes, int context_limit) {
    FILE *file = fopen(path, "rb"); 
    uint32_t magic; int32_t version; uint8_t shared; int32_t *pConf; int i, l;
    
    if (!file) { printf("Error: %s\n", path); (void)Cconin(); exit(1); }
    t->node_id = node_id; t->total_nodes = total_nodes;

    /* Read Header */
    fread(&magic, 4, 1, file); 
    fread(&version, 4, 1, file);
    fread(&t->config, sizeof(Config), 1, file);
    
    /* Endian Swap Config */
    pConf = (int32_t*)&t->config; 
    for(i=0; i<8; i++) pConf[i] = bswap32(pConf[i]);
    
    if (t->config.head_size == 0) t->config.head_size = t->config.dim / t->config.n_heads;
    
    /* Override Sequence Length */
    if (context_limit > 0) t->config.seq_len = context_limit;

    fread(&shared, 1, 1, file); 
    fread(&t->group_size, 4, 1, file); 
    t->group_size = bswap32(t->group_size);
    if (t->group_size <= 0) t->group_size=32; 

    /* Define Layer Ownership */
    if (total_nodes == 4 && t->config.n_layers == 30) {
        if (node_id == 1) { t->layer_start=0; t->layer_end=0; }
        else if (node_id == 2) { t->layer_start=0; t->layer_end=12; }
        else if (node_id == 3) { t->layer_start=12; t->layer_end=24; }
        else { t->layer_start=24; t->layer_end=30; }
    } else {
        int lpn = t->config.n_layers / total_nodes;
        t->layer_start = (node_id-1)*lpn; 
        t->layer_end = t->layer_start+lpn;
    }

    printf("Node %d: Layers %d-%d (Ctx %d)\n", node_id, t->layer_start, t->layer_end, t->config.seq_len);
    
    int pad = 256 - ftell(file); 
    if (pad > 0) fseek(file, pad, SEEK_CUR);

    /* --- PHASE 1: CALCULATE RAM REQUIREMENT --- */
    size_t weight_ram_needed = 0;
    int dim = t->config.dim; 
    int att_dim = t->config.n_heads * t->config.head_size;
    int kv_dim = t->config.n_kv_heads * t->config.head_size;
    int hidden = t->config.hidden_dim;
    int gs = t->group_size;

    size_t sz_att = (dim * att_dim) + ((dim * att_dim)/gs)*4;
    size_t sz_kv = (dim * kv_dim) + ((dim * kv_dim)/gs)*4;
    size_t sz_ffn = (dim * hidden) + ((dim * hidden)/gs)*4;
    size_t sz_ffn2 = (hidden * dim) + ((hidden * dim)/gs)*4;
    
    /* Norms are loaded by everyone for simplicity */
    size_t sz_rms = t->config.n_layers * dim * 4 * 2 + dim * 4;
    
    weight_ram_needed += sz_rms + 1024; /* Padding */

    /* Layer Weights */
    int my_layers = t->layer_end - t->layer_start;
    weight_ram_needed += my_layers * (sz_att*2 + sz_kv*2 + sz_ffn*2 + sz_ffn2);

    /* Embeddings (Node 1 or Node 4) */
    int load_embed = (node_id == 1) || (node_id == total_nodes && shared);
    int load_head = (node_id == total_nodes);
    
    size_t sz_emb = (t->config.vocab_size * dim) + ((t->config.vocab_size * dim)/gs)*4;
    if (load_embed) weight_ram_needed += sz_emb;
    if (load_head && !shared) weight_ram_needed += sz_emb;

    /* --- PHASE 2: ALLOCATE WEIGHT ARENA --- */
    Arena w_arena;
    arena_init(&w_arena, weight_ram_needed);

    /* --- PHASE 3: LOAD WEIGHTS --- */
    TransformerWeights* w = &t->weights;
    
    /* Load Norms */
    w->rms_att_weight = arena_alloc(&w_arena, t->config.n_layers * dim * 4, 0);
    fread(w->rms_att_weight, 4, t->config.n_layers * dim, file);
    w->rms_ffn_weight = arena_alloc(&w_arena, t->config.n_layers * dim * 4, 0);
    fread(w->rms_ffn_weight, 4, t->config.n_layers * dim, file);
    w->rms_final_weight = arena_alloc(&w_arena, dim * 4, 0);
    fread(w->rms_final_weight, 4, dim, file);

    /* Swap Norms */
    for(i=0; i<t->config.n_layers*dim; i++) w->rms_att_weight[i] = bswap_float(w->rms_att_weight[i]);
    for(i=0; i<t->config.n_layers*dim; i++) w->rms_ffn_weight[i] = bswap_float(w->rms_ffn_weight[i]);
    for(i=0; i<dim; i++) w->rms_final_weight[i] = bswap_float(w->rms_final_weight[i]);

    /* Alloc pointers for layer structs (small, safe to malloc) */
    w->wq = malloc_safe(t->config.n_layers * sizeof(QuantizedTensor));
    w->wk = malloc_safe(t->config.n_layers * sizeof(QuantizedTensor));
    w->wv = malloc_safe(t->config.n_layers * sizeof(QuantizedTensor));
    w->wo = malloc_safe(t->config.n_layers * sizeof(QuantizedTensor));
    w->w1 = malloc_safe(t->config.n_layers * sizeof(QuantizedTensor));
    w->w2 = malloc_safe(t->config.n_layers * sizeof(QuantizedTensor));
    w->w3 = malloc_safe(t->config.n_layers * sizeof(QuantizedTensor));

    /* 
     * FIXED LOADING ORDER:
     * The export script writes weights grouped by type across all layers:
     * [Embed, All WQ, All WK, All WV, All WO, All W1, All W2, All W3]
     * 
     * The previous code tried to load per-layer (WQ[0], WK[0]...), which caused
     * massive desynchronization. We now load sequentially by type.
     */

    if(load_embed) printf("Embed...");
    load_tensor_arena(&w->token_embedding_table, file, t->config.vocab_size * dim, gs, load_embed, &w_arena);

    // WQ
    for(l=0; l<t->config.n_layers; l++) {
        int is_mine = (l >= t->layer_start && l < t->layer_end);
        load_tensor_arena(&w->wq[l], file, dim * att_dim, gs, is_mine, &w_arena);
    }

    // WK
    for(l=0; l<t->config.n_layers; l++) {
        int is_mine = (l >= t->layer_start && l < t->layer_end);
        load_tensor_arena(&w->wk[l], file, dim * kv_dim, gs, is_mine, &w_arena);
    }

    // WV
    for(l=0; l<t->config.n_layers; l++) {
        int is_mine = (l >= t->layer_start && l < t->layer_end);
        load_tensor_arena(&w->wv[l], file, dim * kv_dim, gs, is_mine, &w_arena);
    }

    // WO
    for(l=0; l<t->config.n_layers; l++) {
        int is_mine = (l >= t->layer_start && l < t->layer_end);
        load_tensor_arena(&w->wo[l], file, att_dim * dim, gs, is_mine, &w_arena);
    }

    // W1
    for(l=0; l<t->config.n_layers; l++) {
        int is_mine = (l >= t->layer_start && l < t->layer_end);
        load_tensor_arena(&w->w1[l], file, dim * hidden, gs, is_mine, &w_arena);
    }

    // W2
    for(l=0; l<t->config.n_layers; l++) {
        int is_mine = (l >= t->layer_start && l < t->layer_end);
        load_tensor_arena(&w->w2[l], file, hidden * dim, gs, is_mine, &w_arena);
    }

    // W3
    for(l=0; l<t->config.n_layers; l++) {
        int is_mine = (l >= t->layer_start && l < t->layer_end);
        load_tensor_arena(&w->w3[l], file, dim * hidden, gs, is_mine, &w_arena);
        if (is_mine && (l%2)==0) { printf("."); fflush(stdout); }
    }
    
    if (shared) w->wcls = w->token_embedding_table;
    else load_tensor_arena(&w->wcls, file, t->config.vocab_size * dim, gs, load_head, &w_arena);
    
    fclose(file);
    printf(" Done.\n");

    /* --- PHASE 4: ALLOCATE STATE (KV CACHE) --- */
    RunState* s = &t->state;
    if (my_layers > 0 || load_head) {
        int q_dim = t->config.n_heads * t->config.head_size;
        int xb_size = (q_dim > dim) ? q_dim : dim;
        
        size_t kv_cache_size = (size_t)my_layers * t->config.seq_len * kv_dim * 4;
        size_t logits_size = (load_head) ? t->config.vocab_size * 4 : 0;
        
        /* Calculate activation buffers size */
        size_t act_size = (dim*4) + (xb_size*4) + (dim*4) + (hidden*4)*2 + (q_dim*4) + (t->config.n_heads * t->config.seq_len * 4) + (t->config.seq_len * t->config.head_size * 4);
        
        size_t total_state = kv_cache_size + kv_cache_size + logits_size + act_size + 8192; // Extra padding
        
        Arena s_arena;
        arena_init(&s_arena, total_state);

        s->x = arena_alloc(&s_arena, dim*4, 0);
        s->xb = arena_alloc(&s_arena, xb_size*4, 0);
        s->xb2 = arena_alloc(&s_arena, dim*4, 0);
        s->hb = arena_alloc(&s_arena, hidden*4, 0);
        s->hb2 = arena_alloc(&s_arena, hidden*4, 0);
        s->q = arena_alloc(&s_arena, q_dim*4, 0);
        
        /* KV CACHE MUST BE ZEROED TO PREVENT GARBAGE ATTENTION */
        s->key_cache = arena_alloc(&s_arena, kv_cache_size, 1);
        s->value_cache = arena_alloc(&s_arena, kv_cache_size, 1);
        
        s->att = arena_alloc(&s_arena, t->config.n_heads * t->config.seq_len * 4, 0);
        s->rope_cos = arena_alloc(&s_arena, t->config.seq_len * (t->config.head_size/2) * 4, 0);
        s->rope_sin = arena_alloc(&s_arena, t->config.seq_len * (t->config.head_size/2) * 4, 0);
        
        if (load_head) s->logits = arena_alloc(&s_arena, logits_size, 0);

        /* Precompute RoPE */
        int pos, j;
        for (pos = 0; pos < t->config.seq_len; pos++) {
            for (j = 0; j < t->config.head_size; j += 2) {
                float val = pos * (1.0f / (float)pow(100000.0, (double)j / (double)t->config.head_size));
                int idx = pos * (t->config.head_size / 2) + (j / 2);
                s->rope_cos[idx] = (float)cos(val); 
                s->rope_sin[idx] = (float)sin(val);
            }
        }
    } else {
        /* Node 1 dummy state */
        s->x = malloc_safe(dim * 4);
    }
}

/* ---------------------------------------------------------------------------- */
/* COMPUTE KERNELS                                                              */
/* ---------------------------------------------------------------------------- */

void rmsnorm(float* o, float* x, float* weight, int size) {
    float ss = 0.0f; 
    int j; 
    for (j = 0; j < size; j++) ss += x[j] * x[j];
    ss /= size; 
    ss += 1e-5f; 
    ss = 1.0f / sqrt(ss);
    for (j = 0; j < size; j++) o[j] = weight[j] * (ss * x[j]); 
}

void softmax(float* x, int size) {
    float max_val = x[0]; 
    float sum = 0.0f; 
    int i;
    for (i = 1; i < size; i++) if (x[i] > max_val) max_val = x[i];
    for (i = 0; i < size; i++) { 
        x[i] = (float)exp(x[i] - max_val); 
        sum += x[i]; 
    }
    float inv = 1.0f / sum; 
    for (i = 0; i < size; i++) x[i] *= inv;
}

void matmul_q8(float* xout, float* x, QuantizedTensor* qt, int n, int d, int group_size) {
    int i, j;
    for (i = 0; i < d; i++) {
        float val = 0.0f; 
        int32_t in = i * n; 
        float* s_ptr = &qt->s[in / group_size]; 
        int8_t* w_ptr = &qt->q[in];
        
        for (j = 0; j < n; j += group_size) {
            float scale = safe_load_float(s_ptr++); 
            float temp = 0.0f; 
            int k;
            /* Unrolling loop slightly for 68030 */
            for (k = 0; k < group_size; k++) temp += ((float)*w_ptr++ * x[j+k]);
            val += temp * scale;
        }
        xout[i] = val;
    }
}

/* ---------------------------------------------------------------------------- */
/* TOKENIZER & SAMPLER                                                          */
/* ---------------------------------------------------------------------------- */

typedef struct { char *str; int id; } TokenIndex;
typedef struct { 
    char** vocab; 
    float* vocab_scores; 
    TokenIndex *sorted_vocab; 
    int vocab_size; 
    int max_token_length; 
    unsigned char byte_pieces[512]; 
} Tokenizer;

typedef struct { float prob; int index; } ProbIndex;
typedef struct { float temperature; float topp; int topk; unsigned long long rng_state; } Sampler;

int compare_tokens(const void *a, const void *b) { 
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str); 
}

char* decode(Tokenizer* t, int prev_token, int token) { 
    char *piece = t->vocab[token]; 
    int b;
    if (prev_token == 1 && piece[0] == ' ') piece++; 
    if (sscanf(piece, "<0x%02X>", &b) == 1) piece = (char*)t->byte_pieces + b * 2; 
    return piece; 
}

void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    char* str_buffer; size_t str_len = 0; char *c; int i;
    if (text == NULL) text = "";
    
    if (!t->sorted_vocab) { 
        t->sorted_vocab = malloc_safe(t->vocab_size * sizeof(TokenIndex)); 
        for (i = 0; i < t->vocab_size; i++) { 
            t->sorted_vocab[i].str = t->vocab[i]; 
            t->sorted_vocab[i].id = i; 
        } 
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens); 
    }
    
    str_buffer = malloc_safe(t->max_token_length*2 +3);
    *n_tokens = 0; 
    if (bos) tokens[(*n_tokens)++] = 1;
    
    for (c = text; *c != '\0'; c++) {
        if ((*c & 0xC0) != 0x80) str_len = 0;
        str_buffer[str_len++] = *c; str_buffer[str_len] = '\0';
        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) continue;
        
        TokenIndex tok = { 0 }; TokenIndex *res; tok.str = str_buffer;
        res = bsearch(&tok, t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
        
        if (res) tokens[(*n_tokens)++] = res->id; 
        else for (i=0; i < str_len; i++) tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
        
        str_len = 0;
    }
    
    while (1) {
        float best_score = -1e10; int best_id = -1; int best_idx = -1;
        for (i=0; i < (*n_tokens-1); i++) {
            TokenIndex tok = { 0 }; TokenIndex *res;
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            tok.str = str_buffer;
            res = bsearch(&tok, t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
            if (res && t->vocab_scores[res->id] > best_score) { 
                best_score = t->vocab_scores[res->id]; 
                best_id = res->id; 
                best_idx = i; 
            }
        }
        if (best_idx == -1) break;
        tokens[best_idx] = best_id; 
        for (i = best_idx+1; i < (*n_tokens-1); i++) tokens[i] = tokens[i+1]; 
        (*n_tokens)--;
    }
    
    if (eos) tokens[(*n_tokens)++] = 2; 
    free(str_buffer);
}

void build_tokenizer(Tokenizer* t, char* path, int vs) {
    int i; FILE *file; char* arena; char* arena_ptr; size_t arena_size = 0;
    t->vocab_size = vs; 
    t->vocab = malloc_safe(vs * sizeof(char*)); 
    t->vocab_scores = malloc_safe(vs * sizeof(float)); 
    t->sorted_vocab = NULL;
    
    for (i = 0; i < 256; i++) { 
        t->byte_pieces[i*2] = (unsigned char)i; 
        t->byte_pieces[i*2+1] = '\0'; 
    }
    
    file = fopen(path, "rb"); 
    if (!file) { printf("Error: %s\n", path); (void)Cconin(); exit(1); }
    
    fread(&t->max_token_length, 4, 1, file); 
    t->max_token_length = bswap32(t->max_token_length);
    
    /* Calc size for Arena */
    long pos = ftell(file);
    for (i = 0; i < vs; i++) { 
        int len; 
        fseek(file, 4, SEEK_CUR); /* skip score */
        fread(&len, 4, 1, file); 
        len = bswap32(len); 
        arena_size += len + 1; 
        fseek(file, len, SEEK_CUR); 
    }
    fseek(file, pos, SEEK_SET);
    
    printf("Tok Arena: %ld KB...", arena_size/1024);
    arena = malloc_safe(arena_size); 
    arena_ptr = arena;
    
    for (i = 0; i < vs; i++) { 
        int len; 
        fread(t->vocab_scores + i, 4, 1, file); 
        t->vocab_scores[i] = bswap_float(t->vocab_scores[i]); 
        
        fread(&len, 4, 1, file); 
        len = bswap32(len); 
        
        t->vocab[i] = arena_ptr; 
        fread(t->vocab[i], len, 1, file); 
        t->vocab[i][len] = '\0'; 
        arena_ptr += (len + 1); 
    }
    fclose(file); 
    printf(" Done.\n");
}

/* ---------------------------------------------------------------------------- */
/* SAMPLE & SLAVE/MASTER LOOPS                                                  */
/* ---------------------------------------------------------------------------- */

static int compare_probindex(const void* a, const void* b) { 
    ProbIndex* a_=(ProbIndex*)a; 
    ProbIndex* b_=(ProbIndex*)b; 
    if(a_->prob > b_->prob) return -1; 
    if(a_->prob < b_->prob) return 1; 
    return 0; 
}

int sample_node4(float* logits, int vocab_size, float temp, float topp, int topk) {
    int i; 
    
    /* Greedy */
    if (temp == 0.0f) { 
        int best = 0; 
        float max = logits[0]; 
        for(i=1; i<vocab_size; i++) if(logits[i] > max) { max = logits[i]; best = i; } 
        return best; 
    }
    
    for (i=0; i<vocab_size; i++) logits[i] /= temp; 
    softmax(logits, vocab_size);
    
    if (topk > 0 && topk < vocab_size) {
        static ProbIndex* probindex = NULL;
        static int pi_size = 0;
        float topk_sum = 0.0f; 
        float coin;

        if (!probindex || pi_size != vocab_size) {
            if (probindex) free(probindex);
            probindex = malloc_safe(vocab_size * sizeof(ProbIndex));
            pi_size = vocab_size;
        }
        float cdf = 0.0f;
        
        for (i = 0; i < vocab_size; i++) { 
            probindex[i].index = i; 
            probindex[i].prob = logits[i]; 
        }
        qsort(probindex, vocab_size, sizeof(ProbIndex), compare_probindex);
        
        if (topp > 0 && topp < 1.0f) { 
            float cumsum = 0.0f; 
            int k; 
            for (k=0; k < topk; k++) { 
                cumsum += probindex[k].prob; 
                if (cumsum >= topp) { 
                    topk = k + 1; 
                    break; 
                } 
            } 
        }
        
        for (i = 0; i < topk; i++) topk_sum += probindex[i].prob;
        coin = (float)rand() / (float)RAND_MAX * topk_sum; 
        
        for (i = 0; i < topk; i++) { 
            cdf += probindex[i].prob; 
            if (coin < cdf) { 
                return probindex[i].index; 
            } 
        }
        
        return probindex[topk-1].index;
    }
    
    /* Fallback Sampling */
    float coin = (float)rand() / (float)RAND_MAX; 
    float cdf = 0.0f;
    for (i = 0; i < vocab_size; i++) { 
        cdf += logits[i]; 
        if (coin < cdf) return i; 
    }
    return vocab_size - 1;
}

int forward_cluster_master(Transformer* t, int token, int pos, Sampler* samp) {
    Config* p = &t->config; 
    TransformerWeights* w = &t->weights;
    int dim = p->dim; 
    int gs = t->group_size; 
    int i;
    
    /* Prepare Packet */
    PacketHeader header; 
    uint32_t incoming;
    
    header.temp = samp->temperature; 
    header.topp = samp->topp; 
    header.topk = (float)samp->topk;
    header.pos = bswap32((uint32_t)pos);

    size_t packet_size = sizeof(PacketHeader) + dim * 4;
    
    /* FIX: Manual 16-byte alignment to prevent Bus Errors when casting to structs */
    uint8_t* raw_payload = malloc_safe(packet_size + 16);
    uint8_t* payload = (uint8_t*)(((uint32_t)raw_payload + 15) & ~15);
    
    memcpy(payload, &header, sizeof(PacketHeader));
    
    /* Embed Token */
    int offset = token * dim; 
    float* data_ptr = (float*)(payload + sizeof(PacketHeader));
    
    for (i = 0; i < dim; i++) { 
        float s_val = safe_load_float(&w->token_embedding_table.s[offset/gs + i/gs]); 
        int32_t q_val = (int32_t)w->token_embedding_table.q[offset + i]; 
        data_ptr[i] = (float)q_val * s_val; 
    }
    
    /* Send to Ring */
    serial_send_buffer(payload, packet_size); 
    free(raw_payload); /* Free the original pointer */
    
    /* Wait for result */
    serial_read_buffer(&incoming, 4);
    return (int)bswap32(incoming);
}

void slave_loop(Transformer* t) {
    Config* p = &t->config; 
    RunState* s = &t->state;
    int dim = p->dim; 
    int kv_dim = p->n_kv_heads * p->head_size; 
    int gs = t->group_size;
    int l, i, h;
    
    size_t packet_size = sizeof(PacketHeader) + dim * 4;
    
    /* FIX: Manual 16-byte alignment */
    uint8_t* raw_payload = malloc_safe(packet_size + 16);
    uint8_t* payload = (uint8_t*)(((uint32_t)raw_payload + 15) & ~15);
    
    PacketHeader* header = (PacketHeader*)payload; 
    float* data_ptr = (float*)(payload + sizeof(PacketHeader));
    
    printf("Node %d Listening (GS=%d)...\n", t->node_id, gs);
    
    while(1) {
        serial_read_buffer(payload, packet_size);
        memcpy(s->x, data_ptr, dim * 4);
        int pos = (int)bswap32(header->pos);

        /* FIX 2: Bounds Check -> Clamp -> Forward to prevent Ring Deadlock. 
           If pos is garbage, we set it to 0 so we don't crash the bus,
           but we continue processing so the ring completes its cycle. */
        if (pos >= p->seq_len || pos < 0) {
            update_status_led(0xF800); /* Flash Red */
            pos = 0; /* Clamp to safe value to prevent crash */
        }

        for(l = t->layer_start; l < t->layer_end; l++) {
            if (pos % 10 == 0) update_status_led(0x001F);
            
            rmsnorm(s->xb, s->x, t->weights.rms_att_weight + l*dim, dim);
            
            /* Calculate offsets */
            int local_layer_idx = l - t->layer_start; 
            long loff = (long)local_layer_idx * p->seq_len * kv_dim;
            
            s->k = s->key_cache + loff + pos * kv_dim; 
            s->v = s->value_cache + loff + pos * kv_dim;
            
            matmul_q8(s->q, s->xb, &t->weights.wq[l], dim, p->n_heads * p->head_size, gs);
            matmul_q8(s->k, s->xb, &t->weights.wk[l], dim, kv_dim, gs);
            matmul_q8(s->v, s->xb, &t->weights.wv[l], dim, kv_dim, gs);
            
            /* RoPE */
            for (i = 0; i < p->n_heads * p->head_size; i+=2) {
                int cidx = pos * (p->head_size / 2) + (i % p->head_size) / 2;
                float fcr = s->rope_cos[cidx], fci = s->rope_sin[cidx];
                float v0 = s->q[i], v1 = s->q[i+1]; 
                s->q[i] = v0*fcr-v1*fci; s->q[i+1] = v0*fci+v1*fcr;
                if (i < kv_dim) { 
                    v0 = s->k[i]; v1 = s->k[i+1]; 
                    s->k[i] = v0*fcr-v1*fci; s->k[i+1] = v0*fci+v1*fcr; 
                }
            }
            
            /* Attention */
            for (h = 0; h < p->n_heads; h++) {
                float* q = s->q + h * p->head_size; 
                float* att = s->att + h * p->seq_len; 
                int t_step;
                
                for (t_step = 0; t_step <= pos; t_step++) {
                    float* k = s->key_cache + loff + t_step * kv_dim + (h / (p->n_heads / p->n_kv_heads)) * p->head_size;
                    float score = 0.0f; 
                    int k_idx; 
                    for (k_idx = 0; k_idx < p->head_size; k_idx++) score += q[k_idx] * k[k_idx];
                    att[t_step] = score / sqrt(p->head_size);
                }
                
                softmax(att, pos + 1);
                
                float* xb = s->xb + h * p->head_size; 
                int z; for(z=0;z<p->head_size;z++) xb[z]=0.0f;
                
                for (t_step = 0; t_step <= pos; t_step++) {
                    float* v = s->value_cache + loff + t_step * kv_dim + (h / (p->n_heads / p->n_kv_heads)) * p->head_size;
                    float a = att[t_step]; 
                    int v_idx; 
                    for (v_idx = 0; v_idx < p->head_size; v_idx++) xb[v_idx] += a * v[v_idx];
                }
            }
            
            matmul_q8(s->xb2, s->xb, &t->weights.wo[l], p->n_heads * p->head_size, dim, gs);
            for (i = 0; i < dim; i++) s->x[i] += s->xb2[i];
            
            rmsnorm(s->xb, s->x, t->weights.rms_ffn_weight + l*dim, dim);
            matmul_q8(s->hb, s->xb, &t->weights.w1[l], dim, p->hidden_dim, gs);
            matmul_q8(s->hb2, s->xb, &t->weights.w3[l], dim, p->hidden_dim, gs);
            
            for (i = 0; i < p->hidden_dim; i++) { 
                float val = s->hb[i]; 
                val = val / (1.0f + (float)exp(-val)); 
                s->hb[i] = val * s->hb2[i]; 
            }
            
            matmul_q8(s->xb, s->hb, &t->weights.w2[l], p->hidden_dim, dim, gs);
            for (i = 0; i < dim; i++) s->x[i] += s->xb[i];
        }

        if (t->node_id == t->total_nodes) {
            rmsnorm(s->x, s->x, t->weights.rms_final_weight, dim);
            matmul_q8(s->logits, s->x, &t->weights.wcls, p->dim, p->vocab_size, gs);
            
            int chosen_token = sample_node4(s->logits, p->vocab_size, header->temp, header->topp, (int)header->topk);
            
            uint32_t result_id = bswap32((uint32_t)chosen_token);
            serial_send_buffer(&result_id, 4);
        } else {
            memcpy(data_ptr, s->x, dim * 4);
            serial_send_buffer(payload, packet_size);
        }
        update_status_led(0x0000);
    }
}

void chat_loop(Transformer *t, Tokenizer *tok, Sampler *samp, int n_ctx, char *sys_prompt_cfg) {
    int id_im_start = 1, id_im_end = 2, id_nl = 198, id_system = 9690, id_user = 4093, id_ass1 = 520, id_ass2 = 9531;   
    int* tokens; int n_tok = 0, n_chunk = 0; int pos = 0; int initial_pos; int i; char input_buf[1024];
    int user_tokens[1024]; int n_user_tokens; int prompt_tokens[1024]; int n_prompt; int token;
    
    tokens = malloc_safe(n_ctx * sizeof(int));
    
    printf("Pre-filling System Prompt... "); fflush(stdout);
    tokens[n_tok++] = id_im_start; tokens[n_tok++] = id_system; tokens[n_tok++] = id_nl;
    encode(tok, sys_prompt_cfg, 0, 0, tokens+n_tok, &n_chunk); n_tok += n_chunk;
    tokens[n_tok++] = id_im_end; tokens[n_tok++] = id_nl;
    
    for(i=0; i<n_tok; i++) forward_cluster_master(t, tokens[i], pos++, samp);
    
    printf("Done.\n"); initial_pos = pos; 
    printf(">>> "); fflush(stdout);
    term_col = 4; /* Reset column counter for wrapping */
    
    while(1) {
        if(!fgets(input_buf, 1024, stdin)) break; 
        { size_t len=strlen(input_buf); if(len>0 && input_buf[len-1]=='\n') input_buf[len-1]=0; }
        term_col = 0;
        
        if (input_buf[0] == '/') {
            if (strncmp(input_buf, "/bye", 4)==0) break;
            if (strncmp(input_buf, "/clear", 6)==0) { 
                pos = initial_pos; 
                printf("Context reset locally.\n>>> "); 
                term_col = 4; fflush(stdout);
                continue; 
            }
            if (strncmp(input_buf, "/set temp", 9)==0) { 
                samp->temperature = atof(input_buf + 10); 
                printf("Temp: %f\n>>> ", samp->temperature); 
                continue; 
            }
            if (strncmp(input_buf, "/set top_k", 10)==0) { 
                samp->topk = atoi(input_buf + 11); 
                printf("TopK: %d\n>>> ", samp->topk); 
                continue; 
            }
            printf("Commands: /set temp X, /set top_k X, /clear, /bye\n>>> "); continue;
        }
        
        n_prompt = 0;
        prompt_tokens[n_prompt++] = id_im_start; prompt_tokens[n_prompt++] = id_user; prompt_tokens[n_prompt++] = id_nl;
        encode(tok, input_buf, 0, 0, user_tokens, &n_user_tokens);
        for(i=0; i<n_user_tokens; i++) prompt_tokens[n_prompt++] = user_tokens[i];
        prompt_tokens[n_prompt++] = id_im_end; prompt_tokens[n_prompt++] = id_nl;
        prompt_tokens[n_prompt++] = id_im_start; prompt_tokens[n_prompt++] = id_ass1; prompt_tokens[n_prompt++] = id_ass2; prompt_tokens[n_prompt++] = id_nl;
        
        if (pos + n_prompt + 1 >= n_ctx) {
            print_token_wrapped("[Context full, clearing...]\n");
            pos = initial_pos;
        }
        for(i=0; i<n_prompt; i++) forward_cluster_master(t, prompt_tokens[i], pos++, samp);
        token = prompt_tokens[n_prompt-1];
        
        while (pos < n_ctx) {
            int next; char* piece; 
            next = forward_cluster_master(t, token, pos, samp); 
            pos++;
            
            if (next == id_im_end || next == 2) break; 

            /* Check for CTRL+C cleanly between tokens */
            if (Bconstat(2)) { long k = Bconin(2); if ((k & 0xFF) == 3) { printf("\n^C"); break; } }

            piece = decode(tok, token, next); 
            print_token_wrapped(piece);
            token = next; 

        }
        printf("\n>>> "); fflush(stdout);
        term_col = 4;
    }
    free(tokens);
}

void load_config_txt(char *cfg_path, int *node, int *total, char *model, char *tok, int *steps, float *temp, float *topp, int *topk, char *sys_prompt) {
    FILE *f = fopen(cfg_path, "r");
    char line[256];
    if (!f) { printf("Error: %s not found.\n", cfg_path); (void)Cconin(); exit(1); }
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "node=", 5) == 0) *node = atoi(line + 5);
        if (strncmp(line, "total=", 6) == 0) *total = atoi(line + 6);
        if (strncmp(line, "steps=", 6) == 0) *steps = atoi(line + 6);
        if (strncmp(line, "top_k=", 6) == 0) *topk = atoi(line + 6);
        if (strncmp(line, "temp=", 5) == 0) *temp = atof(line + 5);
        if (strncmp(line, "top_p=", 6) == 0) *topp = atof(line + 6);
        if (strncmp(line, "model=", 6) == 0) { char *p = line+6; size_t l=strlen(p); if(l>0 && p[l-1]=='\n') p[l-1]=0; strcpy(model, p); }
        if (strncmp(line, "tokenizer=", 10) == 0) { char *p = line+10; size_t l=strlen(p); if(l>0 && p[l-1]=='\n') p[l-1]=0; strcpy(tok, p); }
        if (strncmp(line, "system_prompt=", 14) == 0) { char *p = line+14; size_t l=strlen(p); if(l>0 && p[l-1]=='\n') p[l-1]=0; strcpy(sys_prompt, p); }
    }
    fclose(f);
}

void test_ring_connection(int node_id, int total_nodes) {
    uint8_t token = 0; 
    /* Flush buffer */
    while(serial_has_data()) (void)Bconin(COMM_DEV);
    
    if (node_id == 1) {
        printf("MASTER: Press SPACE to test Ring.\n");
        while(1) { if (Bconstat(2) && (Bconin(2) & 0xFF) == ' ') break; }
        
        printf("Pinging..."); 
        serial_putc(0xAA);
        token = serial_getc_yielding();
        
        if (token == 0xAA) { 
            printf(" OK! Sending GO.\n"); 
            serial_putc(0x55); 
            token = serial_getc_yielding(); 
        } else { 
            printf(" FAIL (Got 0x%02X)\n", token); 
            (void)Cconin(); 
            exit(1); 
        }
    } else {
        printf("SLAVE: Listening...");
        while(1) {
            token = serial_getc_yielding();
            if (token == 0xAA) { serial_putc(0xAA); }
            else if (token == 0x55) { serial_putc(0x55); printf(" GO!\n"); return; }
        }
    }
}

int main(void) {
    char model_path[64] = "SMOLLM.BIN"; 
    char tok_path[64] = "TOKEN.BIN"; 
    char sys_prompt[256] = "You are SmolLM, a helpful assistant.";
    int steps = 256; 
    int node_id = 1; 
    int total_nodes = 1; 
    float temp = 0.8f; 
    float topp = 0.9f; 
    int topk = 40;
    
    init_video(); 
    detect_resolution(); /* Auto-detect ST Low vs High */
    /* Configure Serial: 19200 or 38400 baud recommended */
    /* Rsconf( ... ); */
    
    srand(time(NULL)); 
    printf("\033E"); /* Clear Screen */
    
    load_config_txt("SMOL.CFG", &node_id, &total_nodes, model_path, tok_path, &steps, &temp, &topp, &topk, sys_prompt);
    
    printf("Node %d / %d | %s | T=%.1f\n", node_id, total_nodes, model_path, temp);
    
    test_ring_connection(node_id, total_nodes);
    
    {
        Transformer t; 
        Tokenizer tok; 
        Sampler samp;
        
        build_transformer(&t, model_path, node_id, total_nodes, steps);
        
        if (node_id == 1) { 
            build_tokenizer(&tok, tok_path, t.config.vocab_size); 
            samp.temperature = temp; 
            samp.topp = topp; 
            samp.topk = topk; 
            chat_loop(&t, &tok, &samp, steps, sys_prompt); 
        } else { 
            slave_loop(&t); 
        }
    }
    return 0;
}
