#ifndef WIN_H
#define WIN_H

// 1. COMPATIBILITY MACROS --------------
// Force Win95 target
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0400
#endif
#ifndef WINVER
#define WINVER 0x0400
#endif

// THE CRITICAL HACK: 
// By defining _CRTIMP to nothing, we strip "__declspec(dllimport)" 
// from the standard headers. This allows our local implementations 
// of _strtoui64 to override the DLL requirements.
#define _CRTIMP 
#define _DLL 

#include <windows.h>
#include <io.h>
#include <ctype.h>
#include <stdlib.h> // Include this HERE so the hack applies to it

// 2. STANDARD MAPPINGS -----------------
#ifndef MAP_FAILED
#define MAP_FAILED ((void *)-1)
#endif
#ifndef PROT_READ
#define PROT_READ 1
#endif
#ifndef MAP_PRIVATE
#define MAP_PRIVATE 2
#endif
#ifndef O_RDONLY
#define O_RDONLY _O_RDONLY
#endif

// 3. MMAP IMPLEMENTATION ---------------
static inline void* mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset) {
    HANDLE hFile = (HANDLE)_get_osfhandle(fd);
    if (hFile == INVALID_HANDLE_VALUE) return MAP_FAILED;
    HANDLE hMapping = CreateFileMapping(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (hMapping == NULL) return MAP_FAILED;
    void* ptr = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, length);
    CloseHandle(hMapping); 
    return (ptr != NULL) ? ptr : MAP_FAILED;
}

static inline int munmap(void *addr, size_t length) {
    return UnmapViewOfFile(addr) ? 0 : -1;
}

// 4. TIMING ----------------------------
static inline int win98_clock_gettime(int clk_id, struct timespec *tp) {
    LARGE_INTEGER frequency;
    LARGE_INTEGER count;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&count);
    tp->tv_sec = count.QuadPart / frequency.QuadPart;
    tp->tv_nsec = (long)((count.QuadPart % frequency.QuadPart) * 1000000000 / frequency.QuadPart);
    return 0;
}
#ifdef clock_gettime
#undef clock_gettime
#endif
#define clock_gettime win98_clock_gettime

// 5. MISSING EXPORTS IMPLEMENTATION ----

// Kernel Wrappers (stdcall)
PVOID __attribute__((stdcall)) __wrap_AddVectoredExceptionHandler(ULONG First, PVOID Handler) { return 0; }
ULONG __attribute__((stdcall)) __wrap_RemoveVectoredExceptionHandler(PVOID Handle) { return 0; }
BOOL  __attribute__((stdcall)) __wrap_SetThreadStackGuarantee(PULONG StackSizeInBytes) { return 0; }

// CRT Wrappers (cdecl) -> NOW STATICALLY LINKED due to _CRTIMP hack
unsigned long long _strtoui64(const char *nptr, char **endptr, int base) {
    const char *s = nptr;
    unsigned long long acc;
    int c;
    unsigned long long cutoff;
    int neg = 0, any, cutlim;
    do { c = *s++; } while (isspace(c));
    if (c == '-') { neg = 1; c = *s++; } else if (c == '+') c = *s++;
    if ((base == 0 || base == 16) && c == '0' && (*s == 'x' || *s == 'X')) { c = s[1]; s += 2; base = 16; }
    if (base == 0) base = c == '0' ? 8 : 10;
    cutoff = (unsigned long long)0xFFFFFFFFFFFFFFFF / (unsigned long long)base;
    cutlim = (unsigned long long)0xFFFFFFFFFFFFFFFF % (unsigned long long)base;
    for (acc = 0, any = 0;; c = *s++) {
        if (isdigit(c)) c -= '0';
        else if (isalpha(c)) c -= isupper(c) ? 'A' - 10 : 'a' - 10;
        else break;
        if (c >= base) break;
        if (any < 0 || acc > cutoff || (acc == cutoff && c > cutlim)) any = -1;
        else { any = 1; acc *= base; acc += c; }
    }
    if (any < 0) acc = (unsigned long long)0xFFFFFFFFFFFFFFFF;
    else if (neg) acc = -acc;
    if (endptr != 0) *endptr = (char *)(any ? s - 1 : nptr);
    return acc;
}

long long _strtoi64(const char *nptr, char **endptr, int base) {
    return (long long)_strtoui64(nptr, endptr, base);
}

#endif
