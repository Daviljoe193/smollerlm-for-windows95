/* 
   Run-Smol: Mac OS 9.x (Unified AltiVec Edition v3.7 Fixed)
   Target: PowerMac G4 "Yikes!" / "Sawtooth" or newer
   Compiler: CodeWarrior 5
*/

#include <Memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <stdarg.h>
#include <signal.h>
#include <ctype.h>

/* Macintosh Toolbox Includes */
#include <SIOUX.h>
#include <console.h>
#include <Events.h>
#include <Timer.h>
#include <StandardFile.h> 
#include <Gestalt.h>
#include <Resources.h>
#include <MacTypes.h>
#include <Quickdraw.h>
#include <ToolUtils.h>
#include <Folders.h>
#include <Files.h>
#include <Script.h>
#include <Dialogs.h>
#include <Controls.h>
#include <Menus.h>
#include <TextUtils.h>

#include <altivec.h>

#define kPrefDialogID 128
#define kAlertID      129
#define P2CStr(s) p2cstr((unsigned char *)(s))
#define C2PStr(s) c2pstr((char *)(s))

/* ---------------------------------------------------------------------------- */
/* PREFS & GLOBALS */
/* ---------------------------------------------------------------------------- */

#define PREF_CREATOR 'Smol'
#define PREF_TYPE    'Pref'
#define PREF_FNAME   "\pRunSmol.pref"

typedef struct {
    FSSpec lastModel;
    int hasLastModel;
    float temperature;
    float topp;
    int topk;
    int steps;
    int mode; /* 0=Chat, 1=Gen */
    char prompt[1024]; 
    int disablePrompt; 
    int useRandomSeed;
    unsigned long fixedSeed;
} AppPrefs;

AppPrefs gPrefs;
volatile int stop_generation = 0; 

/* Globals to preserve the Application Folder location */
short gAppVol;
long gAppDir;

/* Global handle to keep track of the Temporary Memory block */
Handle gModelHandle = NULL;

/* ---------------------------------------------------------------------------- */
/* MATH & HELPERS */
/* ---------------------------------------------------------------------------- */

typedef signed char int8_t;
typedef unsigned char uint8_t;
typedef signed long int32_t;
typedef unsigned long uint32_t;
typedef unsigned long uintptr_t;

#undef bool
#undef vector
#undef pixel

static inline vector float vec_splats_poly(float x) {
    union { float f[4]; vector float v; } u; u.f[0]=x; u.f[1]=x; u.f[2]=x; u.f[3]=x; return u.v;
}
#define vec_splats vec_splats_poly

static inline vector unsigned char vec_load_unaligned(unsigned char* ptr) {
    vector unsigned char v1 = vec_ld(0, ptr);
    vector unsigned char v2 = vec_ld(16, ptr);
    vector unsigned char mask = vec_lvsl(0, ptr);
    return vec_perm(v1, v2, mask);
}

static inline uint32_t bswap32(uint32_t x) {
    return ((x&0xFF000000u)>>24)|((x&0x00FF0000u)>>8)|((x&0x0000FF00u)<<8)|((x&0x000000FFu)<<24);
}
static inline float bswap_float(float x) {
    union { float f; uint32_t i; } u; u.f = x; u.i = bswap32(u.i); return u.f;
}

void* malloc_aligned(size_t size) {
    void* ptr = malloc(size + 16); void* aligned; if (!ptr) return NULL;
    aligned = (void*)(((uintptr_t)ptr + 15) & ~0x0F);
    if (aligned == ptr) aligned = (void*)((uintptr_t)ptr + 16); *((void**)aligned - 1) = ptr; return aligned;
}
void* calloc_aligned(size_t num, size_t size) {
    size_t total = num * size; void* ptr = malloc_aligned(total); if (ptr) memset(ptr, 0, total); return ptr;
}
void free_aligned(void* ptr) { if (ptr) free(*((void**)ptr - 1)); }

void HandleSigInt(int sig) { (void)sig; stop_generation = 1; }

void DumpDebugLog(const char* event, const char* extra, int val) {
    FILE* fp = fopen("RunSmol_Log.txt", "a");
    if(fp) { fprintf(fp, "Event: %s | Info: %s | Val: %d | Clock: %lu\n", event, extra, val, (unsigned long)clock()); fclose(fp); }
}

void YieldToOS(void); /* ADD THIS PROTOTYPE HERE */

void ShowFatalError(const char* msg) {
    DumpDebugLog("Fatal Error", msg, 0);
    printf("\nFATAL ERROR: %s\nPress Cmd-Q to quit.", msg);
    while(1) YieldToOS();
}

void CheckForAltiVec(void) {
    long response;
    OSErr err = Gestalt(gestaltPowerPCProcessorFeatures, &response);
    if (err != noErr || !(response & (1 << gestaltPowerPCHasVectorInstructions))) {
        ShowFatalError("PowerPC G4 (AltiVec) CPU Required.");
    }
}

void YieldToOS(void) { EventRecord ev; WaitNextEvent(everyEvent, &ev, 0, NULL); }

long time_in_ms(void) {
    UnsignedWide uw; Microseconds(&uw); return (long)((uw.hi * 4294967.296) + (uw.lo / 1000));
}

void ClearConsole(void) {
    printf("\f"); fflush(stdout); 
}

/* ---------------------------------------------------------------------------- */
/* PREFERENCES */
/* ---------------------------------------------------------------------------- */

void SetDefaults(void) {
    gPrefs.hasLastModel = 0; gPrefs.temperature = 0.8f; gPrefs.topp = 0.9f; gPrefs.topk = 40;
    gPrefs.steps = 512; gPrefs.mode = 0; 
    strcpy(gPrefs.prompt, "You are SmolLM, a helpful assistant.");
    gPrefs.disablePrompt = 0; gPrefs.useRandomSeed = 1; gPrefs.fixedSeed = 0;
}

void LoadPreferences(void) {
    short vRefNum, refNum; long dirID, count = sizeof(AppPrefs);
    short saveVRef; long saveDir; HGetVol(NULL, &saveVRef, &saveDir); /* Save Vol */
    
    if (FindFolder(kOnSystemDisk, kPreferencesFolderType, kDontCreateFolder, &vRefNum, &dirID) != noErr) { SetDefaults(); return; }
    HSetVol(NULL, vRefNum, dirID);
    memset(&gPrefs, 0, sizeof(AppPrefs)); /* FIX: Clear memory first */
    if (FSOpen(PREF_FNAME, vRefNum, &refNum) == noErr) { 
        FSRead(refNum, &count, &gPrefs); FSClose(refNum); 
        
        /* LOGIC FIX: Check for corrupt/old struct alignment */
        if (count != sizeof(AppPrefs) || gPrefs.steps <= 0 || gPrefs.steps > 16384) {
            SetDefaults();
            /* Restore volume BEFORE logging to ensure log file writes to App folder, not Prefs folder */
            HSetVol(NULL, saveVRef, saveDir);
            DumpDebugLog("Prefs", "Corruption Detected - Resetting Defaults", 0);
        } else {
             HSetVol(NULL, saveVRef, saveDir); /* Restore Vol */
        }
        
        if (gPrefs.topk == 0) gPrefs.topk = 40;
    } else {
        HSetVol(NULL, saveVRef, saveDir); /* Restore Vol */
        SetDefaults();
    }
}

void SavePreferences(void) {
    short vRefNum, refNum; long dirID, count = sizeof(AppPrefs);
    if (FindFolder(kOnSystemDisk, kPreferencesFolderType, kCreateFolder, &vRefNum, &dirID) != noErr) return;
    HSetVol(NULL, vRefNum, dirID); FSDelete(PREF_FNAME, vRefNum);
    if (Create(PREF_FNAME, vRefNum, PREF_CREATOR, PREF_TYPE) == noErr) {
        if (FSOpen(PREF_FNAME, vRefNum, &refNum) == noErr) { FSWrite(refNum, &count, &gPrefs); FSClose(refNum); }
    }
}

int SelectModelFile(void) {
    StandardFileReply reply; SFTypeList types = { 'BINA', 'TEXT', 0, 0 }; 
    StandardGetFile(NULL, -1, types, &reply);
    if (reply.sfGood) { gPrefs.lastModel = reply.sfFile; gPrefs.hasLastModel = 1; return 1; }
    return 0;
}

/* ---------------------------------------------------------------------------- */
/* GUI LOGIC */
/* ---------------------------------------------------------------------------- */

/* Helpers for accessing DITL items */
void SetDialogText(DialogPtr d, int item, const char* str) {
    short type; Handle h; Rect r; unsigned char temp[1024];
    strncpy((char*)temp, str, 1023); C2PStr((char*)temp);
    GetDialogItem(d, item, &type, &h, &r); SetDialogItemText(h, temp);
}
void GetDialogText(DialogPtr d, int item, char* out, int max) {
    short type; Handle h; Rect r; unsigned char p[1024];
    GetDialogItem(d, item, &type, &h, &r); GetDialogItemText(h, p); P2CStr(p);
    strncpy(out, (char*)p, max); out[max-1]=0;
}
void SetDialogVal(DialogPtr d, int item, int val) {
    short type; Handle h; Rect r; GetDialogItem(d, item, &type, &h, &r); SetControlValue((ControlHandle)h, val);
}
int GetDialogVal(DialogPtr d, int item) {
    short type; Handle h; Rect r; GetDialogItem(d, item, &type, &h, &r); return GetControlValue((ControlHandle)h);
}

void SyncDialog(DialogPtr d) {
    char buf[256];
    
    if (gPrefs.hasLastModel) {
        char n[256];
        memcpy(n, gPrefs.lastModel.name, 64);
        P2CStr((unsigned char*)n);
        SetDialogText(d, 4, n);
    } else SetDialogText(d, 4, "No Model Selected");

    SetDialogVal(d, 5, gPrefs.mode == 0); SetDialogVal(d, 6, gPrefs.mode == 1);
    
    if (!gPrefs.disablePrompt) SetDialogText(d, 8, gPrefs.prompt);
    else SetDialogText(d, 8, "");
    
    SetDialogVal(d, 9, gPrefs.disablePrompt);
    
    sprintf(buf, "%.2f", gPrefs.temperature); SetDialogText(d, 10, buf);
    sprintf(buf, "%.2f", gPrefs.topp); SetDialogText(d, 11, buf);
    sprintf(buf, "%d", gPrefs.topk); SetDialogText(d, 12, buf);
    
    SetDialogVal(d, 14, gPrefs.useRandomSeed);
    if (!gPrefs.useRandomSeed) { sprintf(buf, "%lu", gPrefs.fixedSeed); SetDialogText(d, 13, buf); }
    else SetDialogText(d, 13, "");
    
    /* Manual Force Refresh of Static Text Labels to fix ghosting */
    SetDialogText(d, 15, "System Prompt (Chat) / Initial Text (Gen):");
    SetDialogText(d, 16, "Temp:");
    SetDialogText(d, 17, "Top P:");
    SetDialogText(d, 18, "Top K:");
    SetDialogText(d, 19, "Note: Requires G4 (AltiVec) and ~64MB RAM.");
}

void ScrapeDialog(DialogPtr d) {
    char buf[256];
    gPrefs.mode = GetDialogVal(d, 6) ? 1 : 0;
    gPrefs.disablePrompt = GetDialogVal(d, 9);
    if (!gPrefs.disablePrompt) GetDialogText(d, 8, gPrefs.prompt, 1023);
    
    GetDialogText(d, 10, buf, 63); gPrefs.temperature = atof(buf);
    GetDialogText(d, 11, buf, 63); gPrefs.topp = atof(buf);
    GetDialogText(d, 12, buf, 63); gPrefs.topk = atoi(buf);
    
    gPrefs.useRandomSeed = GetDialogVal(d, 14);
    if (!gPrefs.useRandomSeed) {
        GetDialogText(d, 13, buf, 63); gPrefs.fixedSeed = strtoul(buf, NULL, 10);
    }
}

pascal Boolean DialogFilter(DialogPtr dlg, EventRecord *event, short *itemHit) {
    (void)itemHit; /* Silence warning */
    
    if ((event->what == keyDown) || (event->what == autoKey)) {
        char key = (char)(event->message & charCodeMask);
        if ((key == 13) || (key == 3)) { /* Return or Enter */
            /* Check if the PROMPT box (item 8, index 7) is focused */
            if (((DialogPeek)dlg)->editField == (8 - 1)) {
                /* Transform Return into a textual NewLine (ASCII 10) */
                event->message &= ~charCodeMask;
                event->message |= 10; 
            }
        }
    }
    return false; 
}

void DoConfigDialog(void) {
    DialogPtr d; short item;
    d = GetNewDialog(kPrefDialogID, NULL, (WindowPtr)-1);
    if (!d) ExitToShell();
    
    SyncDialog(d);
    ShowWindow(d);
    
    SelectDialogItemText(d, 8, 0, 32767);
    
    while(1) {
        ModalDialog(NewModalFilterProc(DialogFilter), &item);
        
        if (item == 1) { /* Start */
            ScrapeDialog(d);
            if (!gPrefs.hasLastModel) { NoteAlert(kAlertID,NULL); if(SelectModelFile()) SyncDialog(d); continue; }
            SavePreferences(); break;
        }
        else if (item == 2) { DisposeDialog(d); ExitToShell(); } /* Quit */
        else if (item == 3) { SelectModelFile(); SyncDialog(d); }
        else if (item == 5) { SetDialogVal(d,5,1); SetDialogVal(d,6,0); }
        else if (item == 6) { SetDialogVal(d,5,0); SetDialogVal(d,6,1); }
        else if (item == 7) { 
            FSSpec s = gPrefs.lastModel; int h = gPrefs.hasLastModel;
            SetDefaults(); gPrefs.lastModel = s; gPrefs.hasLastModel = h;
            SyncDialog(d);
        }
        else if (item == 9) { /* Disable Prompt Toggle */
            int v = !GetDialogVal(d, 9); SetDialogVal(d, 9, v);
            if (v) SetDialogText(d, 8, ""); else SetDialogText(d, 8, gPrefs.prompt);
        }
        else if (item == 14) { /* Random Toggle */
            int v = !GetDialogVal(d, 14); SetDialogVal(d, 14, v);
            if (v) SetDialogText(d, 13, "");
        }
    }
    DisposeDialog(d);
}

/* ---------------------------------------------------------------------------- */
/* MODEL ENGINE */
/* ---------------------------------------------------------------------------- */

typedef struct { int dim; int hidden_dim; int n_layers; int n_heads; int n_kv_heads; int vocab_size; int seq_len; int head_size; } Config;
typedef struct { int8_t* q; float* s; } QuantizedTensor;
typedef struct { QuantizedTensor token_embedding_table; float* rms_att_weight; float* rms_ffn_weight; QuantizedTensor *wq, *wk, *wv, *wo, *w1, *w2, *w3; float* rms_final_weight; QuantizedTensor wcls; } TransformerWeights;
typedef struct { float *x, *xb, *xb2, *hb, *hb2, *q, *k, *v, *att, *logits; float *key_cache, *value_cache, *rope_cos, *rope_sin; } RunState;
typedef struct { Config config; TransformerWeights weights; RunState state; void* data; size_t file_size; int group_size; } Transformer;

void free_run_state(RunState* s) { if(s->x) free_aligned(s->x); if(s->xb) free_aligned(s->xb); if(s->xb2) free_aligned(s->xb2); if(s->hb) free_aligned(s->hb); if(s->hb2) free_aligned(s->hb2); if(s->q) free_aligned(s->q); if(s->att) free_aligned(s->att); if(s->logits) free_aligned(s->logits); if(s->key_cache) free_aligned(s->key_cache); if(s->value_cache) free_aligned(s->value_cache); if(s->rope_cos) free_aligned(s->rope_cos); if(s->rope_sin) free_aligned(s->rope_sin); }

void free_transformer(Transformer* t) { 
    free(t->weights.wq); free(t->weights.wk); free(t->weights.wv); free(t->weights.wo); 
    free(t->weights.w1); free(t->weights.w2); free(t->weights.w3); 
    
    /* Handle cleanup */
    if (gModelHandle != NULL) {
        DisposeHandle(gModelHandle);
        gModelHandle = NULL;
    } else if (t->data) {
        free_aligned(t->data); 
    }
    
    free_run_state(&t->state); 
}

void precompute_freqs(RunState* s, Config* p, int alloc_steps) {
    int head_size = p->head_size; int pos, i;
    for (pos = 0; pos < alloc_steps; pos++) {
        for (i = 0; i < head_size; i += 2) {
            float freq = 1.0f / pow(10000.0f, i / (float)head_size);
            float val = pos * freq; int idx = pos * (head_size / 2) + (i / 2);
            s->rope_cos[idx] = cos(val); s->rope_sin[idx] = sin(val);
        }
    }
}

void malloc_run_state(RunState* s, Config* p, int alloc_steps) {
    int q_dim = p->n_heads * p->head_size; int kv_dim = p->n_kv_heads * p->head_size;
    int xb_size = (q_dim > p->dim) ? q_dim : p->dim;
    
    if (alloc_steps <= 0) alloc_steps = p->seq_len;
    DumpDebugLog("RunState", "Allocating Steps", alloc_steps);

    s->x = (float*)calloc_aligned(p->dim, sizeof(float)); s->xb = (float*)calloc_aligned(xb_size, sizeof(float)); s->xb2 = (float*)calloc_aligned(p->dim, sizeof(float));
    s->hb = (float*)calloc_aligned(p->hidden_dim, sizeof(float)); s->hb2 = (float*)calloc_aligned(p->hidden_dim, sizeof(float));
    s->q = (float*)calloc_aligned(q_dim, sizeof(float)); s->key_cache = (float*)calloc_aligned(p->n_layers * alloc_steps * kv_dim, sizeof(float));
    s->value_cache = (float*)calloc_aligned(p->n_layers * alloc_steps * kv_dim, sizeof(float));
    s->att = (float*)calloc_aligned(p->n_heads * alloc_steps, sizeof(float)); s->logits = (float*)calloc_aligned(p->vocab_size, sizeof(float));
    s->rope_cos = (float*)calloc_aligned(alloc_steps * (p->head_size / 2), sizeof(float)); s->rope_sin = (float*)calloc_aligned(alloc_steps * (p->head_size / 2), sizeof(float));
    
    if (!s->x || !s->key_cache || !s->rope_cos) { 
        ShowFatalError("Memory allocation failed (OOM). Increase App Memory."); 
    }
    precompute_freqs(s, p, alloc_steps);
}

void init_quantized_tensor(QuantizedTensor* t, char** ptr_ref, int numel, int group_size) {
    char* ptr = *ptr_ref; t->q = (int8_t*)ptr; ptr += numel * sizeof(int8_t);
    t->s = (float*)ptr; 
    { int gs = (group_size > 0) ? group_size : 32; int n_scales = numel / gs; int i;
      for(i=0; i<n_scales; i++) t->s[i] = bswap_float(t->s[i]);
      ptr += n_scales * sizeof(float); }
    *ptr_ref = ptr;
}

void load_weights(Transformer* t, int shared_weights) {
    Config* p = &t->config; TransformerWeights* w = &t->weights; char* ptr = (char*)t->data; 
    unsigned long long dim = p->dim; unsigned long long att_dim = p->n_heads * p->head_size; 
    unsigned long long kv_dim = p->n_kv_heads * p->head_size; unsigned long long hidden_dim = p->hidden_dim;
    int l; float *fptr; int num_f; int z;
    ptr += 256; w->rms_att_weight = (float*)ptr; ptr += p->n_layers * dim * sizeof(float);
    w->rms_ffn_weight = (float*)ptr; ptr += p->n_layers * dim * sizeof(float);
    w->rms_final_weight = (float*)ptr; ptr += dim * sizeof(float);
    fptr = w->rms_att_weight; num_f = p->n_layers * p->dim; for(z=0;z<num_f;z++) fptr[z] = bswap_float(fptr[z]);
    fptr = w->rms_ffn_weight; num_f = p->n_layers * p->dim; for(z=0;z<num_f;z++) fptr[z] = bswap_float(fptr[z]);
    fptr = w->rms_final_weight; num_f = p->dim; for(z=0;z<num_f;z++) fptr[z] = bswap_float(fptr[z]);
    w->wq = (QuantizedTensor*)malloc(p->n_layers * sizeof(QuantizedTensor)); w->wk = (QuantizedTensor*)malloc(p->n_layers * sizeof(QuantizedTensor));
    w->wv = (QuantizedTensor*)malloc(p->n_layers * sizeof(QuantizedTensor)); w->wo = (QuantizedTensor*)malloc(p->n_layers * sizeof(QuantizedTensor));
    w->w1 = (QuantizedTensor*)malloc(p->n_layers * sizeof(QuantizedTensor)); w->w2 = (QuantizedTensor*)malloc(p->n_layers * sizeof(QuantizedTensor));
    w->w3 = (QuantizedTensor*)malloc(p->n_layers * sizeof(QuantizedTensor));
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

void build_transformer(Transformer *t, int steps) {
    FILE *file; uint32_t magic; int version; uint8_t shared; int *pConf; int i; char fname_buf[256];
    OSErr err;
//    Size actualSize;
    
    DumpDebugLog("BuildTransformer", "Starting", gPrefs.steps);

    if (HSetVol(NULL, gPrefs.lastModel.vRefNum, gPrefs.lastModel.parID) != noErr) { 
        StopAlert(kAlertStopAlert, "\pError: Model file inaccessible."); ExitToShell(); 
    }
    
    memcpy(fname_buf, gPrefs.lastModel.name, 256); P2CStr((unsigned char*)fname_buf);
    file = fopen(fname_buf, "rb"); 
    if (!file) ExitToShell();
    
    fseek(file, 0, SEEK_END); t->file_size = ftell(file); fseek(file, 0, SEEK_SET);

    fread(&magic, sizeof(uint32_t), 1, file); fread(&version, sizeof(int), 1, file);
    fread(&t->config, sizeof(int) * 8, 1, file);
    pConf = (int*)&t->config; for(i=0; i<8; i++) pConf[i] = bswap32(pConf[i]);
    if (t->config.head_size == 0) t->config.head_size = t->config.dim / t->config.n_heads;
    fread(&shared, sizeof(uint8_t), 1, file); fread(&t->group_size, sizeof(int), 1, file); 
    if (t->group_size > 10000 || t->group_size < 0) t->group_size = bswap32(t->group_size);
    
    InitCursor(); SetCursor(*GetCursor(watchCursor));
    
    /* --------------------------------------------------------- */
    /* MEMORY STRATEGY CHANGE: Use Temporary Memory for Model Data */
    /* --------------------------------------------------------- */
    DumpDebugLog("Allocating Temp RAM", "Size", t->file_size);
    
    /* Attempt to allocate in System Heap (Temporary Memory) */
    gModelHandle = TempNewHandle(t->file_size + 16, &err);
    
    if (err != noErr || gModelHandle == NULL) {
        /* Fallback: Try standard App Heap if Temp fails */
        DumpDebugLog("TempAlloc Failed", "Trying App Heap", err);
        t->data = malloc_aligned(t->file_size);
        gModelHandle = NULL; /* Flag that we used malloc */
    } else {
        /* Lock the handle high to prevent fragmentation, then dereference */
        HLockHi(gModelHandle);
        t->data = (void*)(((uintptr_t)(*gModelHandle) + 15) & ~0x0F); // Align manually just in case
    }
    
    if (!t->data) { 
        DumpDebugLog("Error", "All Allocations Failed - OOM", t->file_size);
        StopAlert(kAlertStopAlert, "\pOut of RAM. Increase App Memory Partition or System RAM."); 
        ExitToShell(); 
    }
    /* --------------------------------------------------------- */
    
    fseek(file, 0, SEEK_SET); fread(t->data, 1, t->file_size, file); fclose(file);
    
    HSetVol(NULL, gAppVol, gAppDir);
    
    InitCursor(); load_weights(t, shared); 
    malloc_run_state(&t->state, &t->config, steps);
}

/* ---------------------------------------------------------------------------- */
/* KERNELS */
/* ---------------------------------------------------------------------------- */

void rmsnorm(float* o, float* x, float* weight, int size) {
    float ss = 0.0f; int i = 0; 
    vector float sum_v = vec_splats(0.0f); 
    vector float zero_v = vec_splats(0.0f);
    
    /* CodeWarrior Stack Alignment Safety: */
    /* We cannot trust the stack to be 16-byte aligned for the union. */
    /* We store to a vector-aligned buffer first. */
    vector float buf;
    float* fptr = (float*)&buf;

    for (; i <= size - 4; i += 4) { 
        vector float xv = vec_ld(0, &x[i]); 
        sum_v = vec_madd(xv, xv, sum_v); 
    }
    
    /* Extract sum safely */
    buf = sum_v;
    ss = fptr[0] + fptr[1] + fptr[2] + fptr[3];

    for (; i < size; i++) { ss += x[i] * x[i]; }
    ss /= size; ss += 1e-5f; ss = 1.0f / sqrt(ss);

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
}

void softmax(float* x, int size) {
    float max_val = x[0]; float sum = 0.0f; int i;
    for (i = 1; i < size; i++) { if (x[i] > max_val) max_val = x[i]; }
    for (i = 0; i < size; i++) { x[i] = exp(x[i] - max_val); sum += x[i]; }
    for (i = 0; i < size; i++) { x[i] /= sum; }
}

void matmul_q8(float* xout, float* x, QuantizedTensor* qt, int n, int d, int group_size) {
    if (group_size % 4 == 0 && group_size >= 16) {
        int i; 
        vector float zero_v = vec_splats(0.0f);
        
        /* Re-use a single aligned buffer for extraction to save stack churn */
        vector float res_buf;
        float* res_ptr = (float*)&res_buf;

        for (i = 0; i < d; i++) {
            int32_t in = i * n; 
            float* s_ptr = &qt->s[in / group_size]; 
            int8_t* w_ptr = &qt->q[in];
            vector float v_sum = vec_splats(0.0f); 
            int j = 0;
            
            for (; j < n; j += group_size) {
                float scale = *s_ptr++; 
                vector float v_scale = vec_splats(scale); 
                int k;
                for (k = 0; k < group_size; k += 16) {
                    vector unsigned char raw_w = vec_load_unaligned((unsigned char*)&w_ptr[j+k]);
                    /* Unpack chars to shorts, then shorts to ints, then convert to floats */
                    vector signed short w_h = vec_unpackh((vector signed char)raw_w);
                    vector signed short w_l = vec_unpackl((vector signed char)raw_w);
                    
                    vector float vf_w_0 = vec_ctf(vec_unpackh(w_h), 0);
                    vector float vf_w_1 = vec_ctf(vec_unpackl(w_h), 0);
                    vector float vf_w_2 = vec_ctf(vec_unpackh(w_l), 0);
                    vector float vf_w_3 = vec_ctf(vec_unpackl(w_l), 0);
                    
                    /* Dequantize */
                    vf_w_0 = vec_madd(vf_w_0, v_scale, zero_v); 
                    vf_w_1 = vec_madd(vf_w_1, v_scale, zero_v);
                    vf_w_2 = vec_madd(vf_w_2, v_scale, zero_v); 
                    vf_w_3 = vec_madd(vf_w_3, v_scale, zero_v);
                    
                    /* Accumulate */
                    v_sum = vec_madd(vf_w_0, vec_ld(0, &x[j+k]), v_sum);
                    v_sum = vec_madd(vf_w_1, vec_ld(16, &x[j+k]), v_sum);
                    v_sum = vec_madd(vf_w_2, vec_ld(32, &x[j+k]), v_sum);
                    v_sum = vec_madd(vf_w_3, vec_ld(48, &x[j+k]), v_sum);
                }
            }
            /* Safe extraction */
            res_buf = v_sum;
            xout[i] = res_ptr[0] + res_ptr[1] + res_ptr[2] + res_ptr[3];
        }
    } else {
        /* Fallback for odd sizes */
        int i; 
        for (i = 0; i < d; i++) {
            float val = 0.0f; 
            int32_t in = i * n; 
            float* s_ptr = &qt->s[in / group_size]; 
            int8_t* w_ptr = &qt->q[in];
            int j; 
            for (j = 0; j < n; j += group_size) {
                float scale = *s_ptr++; 
                int k; 
                for (k = 0; k < group_size && (j+k) < n; k++) 
                    val += ((float)w_ptr[j+k] * scale) * x[j+k];
            } 
            xout[i] = val;
        }
    }
}

float* forward(Transformer* t, int token, int pos, int stride_steps) {
    Config* p = &t->config; TransformerWeights* w = &t->weights; RunState* s = &t->state;
    float *x = s->x; int dim = p->dim; int kv_dim = p->n_kv_heads * p->head_size; int gs = t->group_size;
    int offset = token * dim; unsigned long long l; int i, h; int loff;
    for (i = 0; i < dim; i++) x[i] = (float)w->token_embedding_table.q[offset + i] * w->token_embedding_table.s[offset/gs + i/gs];
    for(l = 0; l < p->n_layers; l++) {
        rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);
        loff = l * stride_steps * kv_dim; s->k = s->key_cache + loff + pos * kv_dim; s->v = s->value_cache + loff + pos * kv_dim;
        matmul_q8(s->q, s->xb, &w->wq[l], dim, p->n_heads * p->head_size, gs);
        matmul_q8(s->k, s->xb, &w->wk[l], dim, kv_dim, gs);
        matmul_q8(s->v, s->xb, &w->wv[l], dim, kv_dim, gs);
        for (i = 0; i < p->n_heads * p->head_size; i+=2) {
            int cidx = pos * (p->head_size / 2) + (i % p->head_size) / 2;
            float fcr = s->rope_cos[cidx], fci = s->rope_sin[cidx];
            float v0 = s->q[i], v1 = s->q[i+1]; s->q[i] = v0*fcr-v1*fci; s->q[i+1] = v0*fci+v1*fcr;
            if (i < kv_dim) { v0 = s->k[i]; v1 = s->k[i+1]; s->k[i] = v0*fcr-v1*fci; s->k[i+1] = v0*fci+v1*fcr; }
        }
        for (h = 0; h < p->n_heads; h++) {
            float* q = s->q + h * p->head_size; float* att = s->att + h * stride_steps; int t_step;
            for (t_step = 0; t_step <= pos; t_step++) {
                float* k = s->key_cache + loff + t_step * kv_dim + (h / (p->n_heads / p->n_kv_heads)) * p->head_size;
                float score = 0.0f; int k_idx;
                for (k_idx = 0; k_idx < p->head_size; k_idx++) { score += q[k_idx] * k[k_idx]; }
                att[t_step] = score / sqrt(p->head_size);
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
        for (i = 0; i < p->hidden_dim; i++) { float val = s->hb[i]; val = val / (1.0f + exp(-val)); s->hb[i] = val * s->hb2[i]; }
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
static int compare_probindex(const void* a, const void* b) { ProbIndex* a_=(ProbIndex*)a; ProbIndex* b_=(ProbIndex*)b; if(a_->prob>b_->prob)return -1; if(a_->prob<b_->prob)return 1; return 0; }

char* decode(Tokenizer* t, int prev_token, int token) { 
    char *piece = t->vocab[token]; int b;
    if (prev_token == 1 && piece[0] == ' ') piece++; 
    if (sscanf(piece, "<0x%02X>", &b) == 1) piece = (char*)t->byte_pieces + b * 2; 
    return piece; 
}
void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    char* str_buffer; size_t str_len = 0; char *c; int i;
    if (text == NULL) return;
    if (!t->sorted_vocab) { 
        t->sorted_vocab = (TokenIndex*)malloc(t->vocab_size * sizeof(TokenIndex)); 
        for (i = 0; i < t->vocab_size; i++) { t->sorted_vocab[i].str = t->vocab[i]; t->sorted_vocab[i].id = i; } 
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens); 
    }
    str_buffer = (char*)malloc((t->max_token_length*2 +3) * sizeof(char));
    *n_tokens = 0; if (bos) tokens[(*n_tokens)++] = 1;
    for (c = text; *c != '\0'; c++) {
        if ((*c & 0xC0) != 0x80) str_len = 0;
        str_buffer[str_len++] = *c; str_buffer[str_len] = '\0';
        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) continue;
        {
            TokenIndex tok = { 0 }; TokenIndex *res; tok.str = str_buffer;
            res = (TokenIndex*)bsearch(&tok, t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
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
            res = (TokenIndex*)bsearch(&tok, t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
            if (res && t->vocab_scores[res->id] > best_score) { best_score = t->vocab_scores[res->id]; best_id = res->id; best_idx = i; }
        }
        if (best_idx == -1) break;
        tokens[best_idx] = best_id; for (i = best_idx+1; i < (*n_tokens-1); i++) tokens[i] = tokens[i+1]; (*n_tokens)--;
    }
    if (eos) tokens[(*n_tokens)++] = 2; free(str_buffer);
}

void build_tokenizer(Tokenizer* t, char* path, int vs) {
    int i; FILE *file; t->vocab_size = vs; 
    t->vocab = (char**)malloc(vs * sizeof(char*)); t->vocab_scores = (float*)malloc(vs * sizeof(float)); t->sorted_vocab = NULL;
    for (i = 0; i < 256; i++) { t->byte_pieces[i*2] = (unsigned char)i; t->byte_pieces[i*2+1] = '\0'; }
    file = fopen(path, "rb"); 
    if (!file) { 
        char err[100]; sprintf(err, "Tokenizer missing: %s", path);
        {
            unsigned char pErr[101];
            strcpy((char*)pErr, err);
            C2PStr((char*)pErr);
            StopAlert(kAlertStopAlert, pErr);
        }
        ExitToShell(); 
    }
    fread(&t->max_token_length, sizeof(int), 1, file); t->max_token_length = bswap32(t->max_token_length);
    for (i = 0; i < vs; i++) { 
        int len; fread(t->vocab_scores + i, sizeof(float), 1, file); t->vocab_scores[i] = bswap_float(t->vocab_scores[i]);
        fread(&len, sizeof(int), 1, file); len = bswap32(len);
        t->vocab[i] = (char*)malloc(len + 1); fread(t->vocab[i], len, 1, file); t->vocab[i][len] = '\0'; 
    }
    fclose(file);
}

/* Add helper to detect NaN without C99 <math.h> macros if needed */
int is_nan_safe(float x) {
    unsigned long *u = (unsigned long*)&x;
    return ((*u & 0x7F800000UL) == 0x7F800000UL) && (*u & 0x007FFFFFUL);
}

int sample(Sampler* s, float* logits) {
    int i; int q;
    
    /* Safety Check: If the first logit is NaN, the model has collapsed. Return Newline. */
    if (is_nan_safe(logits[0])) {
        return 198; /* ID for newline in Llama 2 tokenizer */
    }

    if (s->temperature == 0.0f) { 
        int max_i=0; float max_p=logits[0]; 
        for(i=1;i<s->vocab_size;i++) if(logits[i]>max_p){max_i=i;max_p=logits[i];} 
        return max_i; 
    }
    
    for (q=0; q<s->vocab_size; q++) logits[q] /= s->temperature; 
    softmax(logits, s->vocab_size);
    
    if (s->topk > 0 && s->topk < s->vocab_size) {
        float topk_sum = 0.0f; float coin; float cdf = 0.0f;
        for (i = 0; i < s->vocab_size; i++) { s->probindex[i].index = i; s->probindex[i].prob = logits[i]; }
        qsort(s->probindex, s->vocab_size, sizeof(ProbIndex), compare_probindex);
        
        for (i = 0; i < s->topk; i++) topk_sum += s->probindex[i].prob;
        
        coin = (float)rand() / (float)RAND_MAX * topk_sum; 
        
        for (i = 0; i < s->topk; i++) { 
            cdf += s->probindex[i].prob; 
            if (coin < cdf) return s->probindex[i].index; 
        }
        return s->probindex[s->topk-1].index;
    } else {
        float coin = (float)rand() / (float)RAND_MAX; float cdf = 0.0f; 
        for (i = 0; i < s->vocab_size; i++) { 
            cdf += logits[i]; 
            if (coin < cdf) return i; 
        }
        return s->vocab_size - 1;
    }
}

/* ---------------------------------------------------------------------------- */
/* LOOPS */
/* ---------------------------------------------------------------------------- */

void generate(Transformer *t, Tokenizer *tok, Sampler *samp, char *prompt, int steps) {
    int num_prompt_tokens = 0; int* prompt_tokens; long start = 0; int next; int token; int pos = 0;
    float* logits; char* empty = "";
    if (prompt == NULL) prompt = empty;
    prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int));
    encode(tok, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    token = prompt_tokens[0]; 
    while (pos < steps) {
        if (stop_generation) { printf("\n^C"); break; }
        YieldToOS();
        logits = forward(t, token, pos, steps);
        if (pos < num_prompt_tokens - 1) next = prompt_tokens[pos + 1]; else next = prompt_tokens[pos + 1]; if (pos >= num_prompt_tokens - 1) next = sample(samp, logits);
        pos++;
        if (next == 2) break; 
        { char* piece = decode(tok, token, next); if (piece) { printf("%s", piece); fflush(stdout); } }
        token = next;
        if (start == 0) start = time_in_ms();
    }
    printf("\n");
    if (pos > 1) { long end = time_in_ms(); fprintf(stderr, "Speed: %f tok/s\n", (pos-1) / (double)(end-start)*1000); }
    free(prompt_tokens);
}

/* Restored robust chat loop */
void chat_loop(Transformer *t, Tokenizer *tok, Sampler *samp, int n_ctx, char* cli_system_prompt) {
    int id_nl = 198; int id_user = 4093; int id_ass1 = 520; int id_ass2 = 9531;   
    char* sys_prompt = "You are SmolLM, a helpful assistant.";
    int* tokens; int n_tok = 0, n_chunk = 0; int pos = 0; int initial_pos; int i; char input_buf[1024];
    int* user_tokens; int n_user_tokens; int* prompt_tokens; int n_prompt; int token;
    
    /* LOGIC FIX: If NULL, use default. If not NULL (even empty), use valid custom/empty string. */
    if (cli_system_prompt != NULL) sys_prompt = cli_system_prompt;
    
    /* Allocating large arrays on heap to assume small Mac OS 9 stack sizes */
    tokens = (int*)malloc(n_ctx * sizeof(int));
    user_tokens = (int*)malloc(2048 * sizeof(int));
    prompt_tokens = (int*)malloc(2048 * sizeof(int));
    
    /* Reset Stdin to prevent immediate EOF from modal dialog interaction */
    clearerr(stdin);

    InitCursor(); SetCursor(*GetCursor(watchCursor)); 
    
    /* LOGIC FIX: Only encode and forward System Prompt if it actually exists */
    if (sys_prompt && strlen(sys_prompt) > 0) {
        tokens[n_tok++] = 1; tokens[n_tok++] = 9690; tokens[n_tok++] = id_nl;
        encode(tok, sys_prompt, 0, 0, tokens+n_tok, &n_chunk); n_tok += n_chunk;
        tokens[n_tok++] = 2; tokens[n_tok++] = id_nl;
        for(i=0; i<n_tok; i++) forward(t, tokens[i], pos++, n_ctx);
    }
    
    initial_pos = pos; 
    InitCursor();

    printf("\n>>> "); fflush(stdout);

    while(1) {
        long start = 0;
        
        if(!fgets(input_buf, 1024, stdin)) break;

        { size_t len=strlen(input_buf); if(len>0 && input_buf[len-1]=='\n') input_buf[len-1]=0; }
        if (strlen(input_buf) == 0) { printf(">>> "); fflush(stdout); continue; }

        if (input_buf[0] == '/') {
            if (strncmp(input_buf, "/bye", 4)==0) { break; } 
            if (strncmp(input_buf, "/clear", 6)==0) { 
                /* Actually implement clear for Mac console */
                printf("\f"); /* Form feed usually clears SIOUX console */
                printf(">>> "); 
                fflush(stdout); 
                pos = initial_pos; /* Reset context */
                continue; 
        }
    if (strncmp(input_buf, "/?", 2)==0 || strncmp(input_buf, "/help", 5)==0) {
         printf("Commands:\n");
         printf("  /set parameter temperature <val>\n");
         printf("  /set parameter top_k <val>\n");
         printf("  /set parameter top_p <val>\n");
         printf("  /clear\n");
         printf("  /bye\n>>> ");
         fflush(stdout);
         continue;
    }
    /* Re-implemented Set Commands */
    if (strncmp(input_buf, "/set parameter temperature", 26)==0) { samp->temperature = atof(input_buf + 27); continue; }
    if (strncmp(input_buf, "/set parameter temp", 19)==0) { samp->temperature = atof(input_buf + 20); continue; }
    if (strncmp(input_buf, "/set parameter top_k", 20)==0) { samp->topk = atoi(input_buf + 21); continue; }
    if (strncmp(input_buf, "/set parameter top_p", 20)==0) { samp->topp = atof(input_buf + 21); continue; }
}

        stop_generation = 0;
        n_prompt = 0;
        prompt_tokens[n_prompt++] = 1; prompt_tokens[n_prompt++] = id_user; prompt_tokens[n_prompt++] = id_nl;
        encode(tok, input_buf, 0, 0, user_tokens, &n_user_tokens);
        for(i=0; i<n_user_tokens; i++) prompt_tokens[n_prompt++] = user_tokens[i];
        prompt_tokens[n_prompt++] = 2; prompt_tokens[n_prompt++] = id_nl;
        prompt_tokens[n_prompt++] = 1; prompt_tokens[n_prompt++] = id_ass1; prompt_tokens[n_prompt++] = id_ass2; prompt_tokens[n_prompt++] = id_nl;
        
        for(i=0; i<n_prompt; i++) {
            if (stop_generation) break; 
            forward(t, prompt_tokens[i], pos++, n_ctx);
        }
        if (stop_generation) { printf("\n[Interrupted]"); continue; }
        
        token = prompt_tokens[n_prompt-1];
        start = time_in_ms();
        
        while (pos < n_ctx) {
            float* logits; int next; char* piece;
            if (stop_generation) { printf("\n[Interrupted]"); break; }
            logits = forward(t, token, pos, n_ctx);
            next = sample(samp, logits); pos++;
            if (next == 2 || next == 2) break; 
            piece = decode(tok, token, next);
            printf("%s", piece); fflush(stdout);
            token = next;
            YieldToOS(); 
        }
        printf("\n>>> "); fflush(stdout);
    }
    free(tokens); free(user_tokens); free(prompt_tokens);
}

int main(void) {
    unsigned long long rng_seed;

    InitGraf(&qd.thePort); InitFonts(); InitWindows(); InitMenus(); TEInit(); InitDialogs(0);

    SIOUXSettings.autocloseonquit = 0;  
    SIOUXSettings.asktosaveonclose = 0; 
    SIOUXSettings.rows = 40;
    
    CheckForAltiVec();
    signal(SIGINT, HandleSigInt);
    
    /* Save the App Directory ID before anything else happens */
    HGetVol(NULL, &gAppVol, &gAppDir);
    DumpDebugLog("App Started", "Volumes Saved", 0);
    
    LoadPreferences();
    if (!gPrefs.hasLastModel) {
        NoteAlert(kAlertID, NULL); 
        if (!SelectModelFile()) ExitToShell();
        SavePreferences(); 
        DoConfigDialog(); 
    } else {
        DoConfigDialog();
    }
    
    if (gPrefs.useRandomSeed) rng_seed = (unsigned long long)time(NULL);
    else rng_seed = (unsigned long long)gPrefs.fixedSeed;
    srand((unsigned int)rng_seed); 
    
    ClearConsole();
    printf("SmolLM Engine (PPC G4/AltiVec)\nModel: %.*s\nMode: %s\n----------------------------\n", 
           gPrefs.lastModel.name[0], gPrefs.lastModel.name+1, 
           gPrefs.mode == 1 ? "Generate" : "Chat");
    
    {
        Transformer transformer; Tokenizer tok; Sampler samp; 
        
        DumpDebugLog("Main", "Calling BuildTransformer", 0);
        /* This function now manages HSetVol save/restore so CWD is safe */
        build_transformer(&transformer, gPrefs.steps);
        DumpDebugLog("Main", "BuildTransformer Returned", 0);
        
        /* This will now look in the App folder properly */
        build_tokenizer(&tok, "tokenizer.bin", transformer.config.vocab_size);
        
        samp.vocab_size = transformer.config.vocab_size; 
        samp.temperature = gPrefs.temperature; 
        samp.topp = gPrefs.topp; 
        samp.topk = gPrefs.topk; 
        samp.rng_state = rng_seed;
        samp.probindex = (ProbIndex*)malloc(samp.vocab_size*sizeof(ProbIndex));
        
        DumpDebugLog("Starting Loop", gPrefs.mode == 1 ? "Generate" : "Chat", gPrefs.disablePrompt);

        if (gPrefs.mode == 1) generate(&transformer, &tok, &samp, gPrefs.prompt, gPrefs.steps);
        else {
            /* LOGIC FIX: Pass NULL for default, "" for disabled, or String for custom */
            char *p = NULL;
            if (gPrefs.disablePrompt) p = ""; 
            else if (strlen(gPrefs.prompt) > 0) p = gPrefs.prompt;
            /* else p remains NULL, triggering Default in chat_loop */

            chat_loop(&transformer, &tok, &samp, gPrefs.steps, p);
        }
        
        free(samp.probindex); 
        free_transformer(&transformer);
    }
    
DumpDebugLog("Clean Exit", "Success", 0);

    /* FIX: Remove manual loop. SIOUXSettings.autocloseonquit = 0 
       automatically keeps the window open and handles events now. */
    printf("\nProgram Ended. Command-Q to Quit.");
    
    return 0;
}
