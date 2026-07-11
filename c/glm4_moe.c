/* Motore GLM-4.5/4.6/4.7 (architettura glm4_moe / Glm4MoeForCausalLM) in C puro.
 * Porta lo streaming MoE di Colibri ai modelli GLM-4.x: architettura piu' semplice
 * di GLM-5 (GQA standard al posto di MLA, niente DSA indexer).
 *
 * Target: GLM-4.5-Air (106B/12B) — 128 expert, top-8, 46 layer, GQA 96h/8kv.
 *   build: make glm4   run: SNAP=./glm45air ./glm4 <cap> <expert_bits> <dense_bits>
 *
 * Differenze chiave rispetto a glm.c (GLM-5.2 / GlmMoeDsa):
 *   - Attenzione: GQA standard (Q/K/V/O lineari diretti, no MLA compresso)
 *   - KV cache: full key+value per-head (non latente compressa)
 *   - Niente DSA indexer
 *   - Router: sigmoid + noaux_tc (identico a GLM-5.2)
 *   - partial_rotary_factor: solo meta' dei dim della head sono roped
 *   - attention_bias: True (GLM-4.5 usa bias su Q/K/V, ma NON su o_proj)
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <limits.h>
#include <pthread.h>                              /* thread I/O del PILOTA */
#include <unistd.h>
#include <sys/resource.h>
#if defined(__APPLE__) || defined(__linux__)
#include <sys/mman.h>                             /* mlock: inchioda le pagine in RAM / wire pages into RAM */
#endif
#include "st.h"
#include "tok.h"
#include "tier.h"
#ifdef COLI_CUDA
#include <omp.h>
#include "backend_cuda.h"
#endif
#ifdef COLI_METAL
#include "backend_metal.h"
#endif
#ifdef __AVX2__
#include <immintrin.h>
static inline float hsum256(__m256 v){            /* somma orizzontale di 8 float */
    __m128 lo=_mm256_castps256_ps128(v), hi=_mm256_extractf128_ps(v,1);
    lo=_mm_add_ps(lo,hi); __m128 sh=_mm_movehl_ps(lo,lo); lo=_mm_add_ps(lo,sh);
    sh=_mm_shuffle_ps(lo,lo,1); lo=_mm_add_ss(lo,sh); return _mm_cvtss_f32(lo);
}
#elif defined(__ARM_NEON)
#include <arm_neon.h>                             /* Apple Silicon / aarch64: kernel NEON */
#endif
#ifdef __APPLE__
#include <mach/mach.h>                            /* host_statistics64: MemAvailable di macOS */
#endif

/* ---- Config ---- */
typedef struct {
    int hidden, n_layers, n_heads, n_kv_heads, head_dim;
    int n_experts, topk, moe_inter, dense_inter;
    int first_dense, n_shared, vocab;
    int n_group, topk_group, norm_topk;
    int stop_ids[8], n_stop;
    float eps, theta, attn_scale, routed_scale;
    float partial_rotary_factor;
    int attention_bias;
    int has_mtp;
} Cfg;

/* ---- Quantized tensor (shared with glm.c) ---- */
/* fmt: 0 F32, 1 INT8, 2 INT4 (2/byte), 3 INT2 (4/byte). q4 ospita sia int4 che int2 packed. */
typedef struct {
    int fmt; float *qf; int8_t *q8; uint8_t *q4; float *s; int O, I;
#ifdef COLI_CUDA
    ColiCudaTensor *cuda;
#endif
#ifdef COLI_METAL
    ColiMetalTensor *metal;
#endif
    int cuda_eligible, cuda_failed, cuda_device;
#ifdef COLI_METAL
    int metal_eligible, metal_failed;
#endif
} QT;

static int64_t qt_bytes(const QT *t){
    int64_t n=(int64_t)t->O*t->I;
    if(t->fmt==0) return n*4;
    if(t->fmt==1) return n + (int64_t)t->O*4;
    if(t->fmt==3) return (int64_t)t->O*((t->I+3)/4) + (int64_t)t->O*4;
    return (int64_t)t->O*((t->I+1)/2) + (int64_t)t->O*4;
}

/* ---- Layer: GQA attention ---- */
typedef struct {
    float *in_ln, *post_ln;
    /* GQA attention: standard Q/K/V/O projections */
    QT q_proj, k_proj, v_proj, o_proj;
    float *q_bias, *k_bias, *v_bias;  /* o_proj has NO bias */
    int sparse;
    /* dense mlp (sparse==0) */
    QT gate_proj, up_proj, down_proj;
    /* moe (sparse==1) */
    float *router, *router_bias;
    QT sh_gate, sh_up, sh_down;
} Layer;

/* ---- Expert slot (identical to glm.c) ---- */
typedef struct { int eid; QT g,u,d; uint8_t *slab; float *fslab;
                 int64_t slab_cap, fslab_cap; uint64_t used; } ESlot;

/* ---- Model ---- */
typedef struct {
    Cfg c; shards S;
    int ebits, dbits;
    QT embed, lm_head; float *final_norm;
    Layer *L;
    /* KV cache: full K and V per head (not compressed latent like MLA) */
    float **Kc, **Vc;   /* [n_layers][max_t * n_kv_heads * head_dim] */
    int max_t;
    ESlot **ecache; int *ecn; int ecap;
    ESlot ws[64];
    ESlot **pin; int *npin;
    uint32_t **eusage, **eheat;
    int **eroute; int *enr;
    uint64_t eclock, hits, miss, ereq;
    uint64_t gpu_expert_calls; int gpu_expert_count; int64_t gpu_expert_bytes;
    uint64_t n_fw, n_emit;
    double t_edisk, t_emm, t_attn, t_head;
    int64_t resident_bytes;
} Model;

static void usage_save(Model *m);        /* cache che impara: definita accanto a stats_dump */
#ifdef COLI_CUDA
static int g_cuda_enabled;
static double g_cuda_expert_gb;
static int g_cuda_dense;
static int g_cuda_devices[COLI_CUDA_MAX_DEVICES], g_cuda_ndev, g_cuda_rr;
static int64_t g_cuda_dense_projected[COLI_CUDA_MAX_DEVICES];
static void qt_cuda_reset(QT *t){
    if(t->cuda){ coli_cuda_tensor_free(t->cuda); t->cuda=NULL; }
    t->cuda_failed=0;
}
static int qt_cuda_upload(QT *t){
    const void *weights = t->fmt==0 ? (const void*)t->qf
                        : t->fmt==1 ? (const void*)t->q8 : (const void*)t->q4;
    return coli_cuda_tensor_upload(&t->cuda,weights,t->s,t->fmt,t->I,t->O,t->cuda_device);
}
static void cuda_stats_print(void){
    size_t n=0,b=0; coli_cuda_stats(-1,&n,&b);
    fprintf(stderr,"[CUDA] resident set: %zu tensor, %.2f GB VRAM\n",n,b/1e9);
    if(g_cuda_ndev>1) for(int i=0;i<g_cuda_ndev;i++){
        coli_cuda_stats(g_cuda_devices[i],&n,&b);
        fprintf(stderr,"[CUDA]   device %d: %zu tensor, %.2f GB\n",g_cuda_devices[i],n,b/1e9);
    }
}
static int parse_cuda_devices(const char *list, int *out){
    if(!list||!*list) return 0;
    int n=0; const char *p=list;
    while(*p){
        char *end=NULL; long v=strtol(p,&end,10);
        if(end==p||v<0||v>INT_MAX||n>=COLI_CUDA_MAX_DEVICES) return 0;
        for(int i=0;i<n;i++) if(out[i]==(int)v) return 0;
        out[n++]=(int)v; p=end;
        while(*p==' '||*p=='\t') p++;
        if(!*p) break;
        if(*p++!=',') return 0;
        while(*p==' '||*p=='\t') p++;
        if(!*p) return 0;
    }
    return n;
}
#endif
#ifdef COLI_METAL
static int g_metal_enabled;
static double g_metal_expert_gb;
static int g_metal_dense;
static void qt_metal_reset(QT *t){
    if(t->metal){ coli_metal_tensor_free(t->metal); t->metal=NULL; }
    t->metal_failed=0;
}
static int qt_metal_upload(QT *t){
    const void *weights = t->fmt==0 ? (const void*)t->qf
                        : t->fmt==1 ? (const void*)t->q8 : (const void*)t->q4;
    return coli_metal_tensor_upload(&t->metal,weights,t->s,t->fmt,t->I,t->O);
}
static void metal_stats_print(void){
    size_t n=0,b=0; coli_metal_stats(&n,&b);
    fprintf(stderr,"[Metal] resident set: %zu tensor, %.2f GB\n",n,b/1e9);
}
#endif
static double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static double rss_gb(void){ struct rusage r; getrusage(RUSAGE_SELF,&r);
#ifdef __APPLE__
    return r.ru_maxrss/(1024.0*1024.0*1024.0);   /* macOS: ru_maxrss in BYTE */
#else
    return r.ru_maxrss/(1024.0*1024.0);          /* Linux: in KB */
#endif
}
static float *falloc(int64_t n){
    if(n<0 || (uint64_t)n > SIZE_MAX/sizeof(float)){ fprintf(stderr,"falloc: n=%lld fuori range\n",(long long)n); exit(1); }
    float *p=malloc((size_t)n*sizeof(float)); if(!p){fprintf(stderr,"OOM\n");exit(1);} return p; }

/* y[S,O] = x[S,I] @ W^T, W[O,I] f32 */
static void matmul(float *y, const float *x, const float *W, int S, int I, int O){
    #pragma omp parallel for schedule(static)
    for (int o=0;o<O;o++){ const float *w=W+(int64_t)o*I;
        for (int s=0;s<S;s++){ const float *xs=x+(int64_t)s*I; float a=0; for(int i=0;i<I;i++) a+=xs[i]*w[i]; y[(int64_t)s*O+o]=a; } }
}
/* y[S,O] = x[S,I] @ W^T con W quantizzato int8 per-riga + scala[O] (dequant-on-use) */
static void matmul_q(float *y, const float *x, const int8_t *q, const float *scale, int S, int I, int O){
    #pragma omp parallel for schedule(static)
    for (int o=0;o<O;o++){ const int8_t *w=q+(int64_t)o*I; float sc=scale[o];
        for (int s=0;s<S;s++){ const float *xs=x+(int64_t)s*I; float a=0; int i=0;
#ifdef __AVX2__
            __m256 acc=_mm256_setzero_ps();
            for(;i+8<=I;i+=8){ __m256i wi=_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(w+i)));
                acc=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i), _mm256_cvtepi32_ps(wi), acc); }
            a=hsum256(acc);
#elif defined(__ARM_NEON)
            float32x4_t ac0=vdupq_n_f32(0), ac1=vdupq_n_f32(0);
            for(;i+8<=I;i+=8){ int16x8_t w16=vmovl_s8(vld1_s8(w+i));
                ac0=vfmaq_f32(ac0, vld1q_f32(xs+i),   vcvtq_f32_s32(vmovl_s16(vget_low_s16(w16))));
                ac1=vfmaq_f32(ac1, vld1q_f32(xs+i+4), vcvtq_f32_s32(vmovl_s16(vget_high_s16(w16)))); }
            a=vaddvq_f32(vaddq_f32(ac0,ac1));
#endif
            for(;i<I;i++) a+=xs[i]*(float)w[i]; y[(int64_t)s*O+o]=a*sc; } }
}
/* y[S,O] = x[S,I] @ W^T con W int4 impacchettato (2 valori/byte) + scala[O]. */
static void matmul_i4(float *y, const float *x, const uint8_t *q4, const float *scale, int S, int I, int O){
    int rb=(I+1)/2;
    #pragma omp parallel for schedule(static)
    for (int o=0;o<O;o++){ const uint8_t *w=q4+(int64_t)o*rb; float sc=scale[o];
        for (int s=0;s<S;s++){ const float *xs=x+(int64_t)s*I; float a=0; int i=0;
#ifdef __AVX2__
            const __m128i m4=_mm_set1_epi8(0x0F); const __m256i b8=_mm256_set1_epi32(8);
            __m256 acc=_mm256_setzero_ps();
            for(;i+16<=I;i+=16){ __m128i by=_mm_loadl_epi64((const __m128i*)(w+(i>>1)));
                __m128i lo=_mm_and_si128(by,m4), hi=_mm_and_si128(_mm_srli_epi16(by,4),m4);
                __m128i nib=_mm_unpacklo_epi8(lo,hi);
                __m256 w0=_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(nib),b8));
                __m256 w1=_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(nib,8)),b8));
                acc=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i),   w0, acc);
                acc=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i+8), w1, acc); }
            a=hsum256(acc);
#elif defined(__ARM_NEON)
            const uint8x8_t m4=vdup_n_u8(0x0F); const int8x8_t b8=vdup_n_s8(8);
            float32x4_t ac0=vdupq_n_f32(0), ac1=vdupq_n_f32(0);
            for(;i+16<=I;i+=16){ uint8x8_t by=vld1_u8(w+(i>>1));
                uint8x8x2_t z=vzip_u8(vand_u8(by,m4), vshr_n_u8(by,4));
                int16x8_t w0=vmovl_s8(vsub_s8(vreinterpret_s8_u8(z.val[0]),b8));
                int16x8_t w1=vmovl_s8(vsub_s8(vreinterpret_s8_u8(z.val[1]),b8));
                ac0=vfmaq_f32(ac0, vld1q_f32(xs+i),    vcvtq_f32_s32(vmovl_s16(vget_low_s16(w0))));
                ac1=vfmaq_f32(ac1, vld1q_f32(xs+i+4),  vcvtq_f32_s32(vmovl_s16(vget_high_s16(w0))));
                ac0=vfmaq_f32(ac0, vld1q_f32(xs+i+8),  vcvtq_f32_s32(vmovl_s16(vget_low_s16(w1))));
                ac1=vfmaq_f32(ac1, vld1q_f32(xs+i+12), vcvtq_f32_s32(vmovl_s16(vget_high_s16(w1)))); }
            a=vaddvq_f32(vaddq_f32(ac0,ac1));
#endif
            for(;i+1<I;i+=2){ uint8_t byte=w[i>>1]; int lo=(int)(byte&0xF)-8, hi=(int)(byte>>4)-8;
                a += xs[i]*(float)lo + xs[i+1]*(float)hi; }
            if(i<I){ uint8_t byte=w[i>>1]; int lo=(int)(byte&0xF)-8; a += xs[i]*(float)lo; }
            y[(int64_t)s*O+o]=a*sc; } }
}
/* y[S,O] = x[S,I] @ W^T con W int2 impacchettato (4 valori/byte) + scala[O]. nibble 2-bit -> [-2,1]. */
static void matmul_i2(float *y, const float *x, const uint8_t *q2, const float *scale, int S, int I, int O){
    int rb=(I+3)/4;
    #pragma omp parallel for schedule(static)
    for (int o=0;o<O;o++){ const uint8_t *w=q2+(int64_t)o*rb; float sc=scale[o];
        for (int s=0;s<S;s++){ const float *xs=x+(int64_t)s*I; float a=0; int i=0;
#ifdef __AVX2__
            const __m128i m2=_mm_set1_epi8(0x03); const __m256i b2=_mm256_set1_epi32(2);
            __m256 acc=_mm256_setzero_ps();
            for(;i+16<=I;i+=16){ __m128i by=_mm_cvtsi32_si128(*(const int*)(w+(i>>2)));
                __m128i p0=_mm_and_si128(by,m2), p1=_mm_and_si128(_mm_srli_epi16(by,2),m2);
                __m128i p2=_mm_and_si128(_mm_srli_epi16(by,4),m2), p3=_mm_and_si128(_mm_srli_epi16(by,6),m2);
                __m128i lo=_mm_unpacklo_epi8(p0,p1), hi=_mm_unpacklo_epi8(p2,p3);
                __m128i nib=_mm_unpacklo_epi16(lo,hi);
                __m256 w0=_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(nib),b2));
                __m256 w1=_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(nib,8)),b2));
                acc=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i),   w0, acc);
                acc=_mm256_fmadd_ps(_mm256_loadu_ps(xs+i+8), w1, acc); }
            a=hsum256(acc);
#elif defined(__ARM_NEON)
            const uint8x8_t m2v=vdup_n_u8(3); const int8x8_t b2v=vdup_n_s8(2);
            float32x4_t ac0=vdupq_n_f32(0), ac1=vdupq_n_f32(0);
            for(;i+16<=I;i+=16){ uint32_t wd; memcpy(&wd, w+(i>>2), 4);
                uint8x8_t by=vreinterpret_u8_u32(vdup_n_u32(wd));
                uint8x8x2_t z01=vzip_u8(vand_u8(by,m2v),              vand_u8(vshr_n_u8(by,2),m2v));
                uint8x8x2_t z23=vzip_u8(vand_u8(vshr_n_u8(by,4),m2v), vshr_n_u8(by,6));
                uint16x4x2_t zz=vzip_u16(vreinterpret_u16_u8(z01.val[0]), vreinterpret_u16_u8(z23.val[0]));
                int16x8_t w0=vmovl_s8(vsub_s8(vreinterpret_s8_u16(zz.val[0]),b2v));
                int16x8_t w1=vmovl_s8(vsub_s8(vreinterpret_s8_u16(zz.val[1]),b2v));
                ac0=vfmaq_f32(ac0, vld1q_f32(xs+i),    vcvtq_f32_s32(vmovl_s16(vget_low_s16(w0))));
                ac1=vfmaq_f32(ac1, vld1q_f32(xs+i+4),  vcvtq_f32_s32(vmovl_s16(vget_high_s16(w0))));
                ac0=vfmaq_f32(ac0, vld1q_f32(xs+i+8),  vcvtq_f32_s32(vmovl_s16(vget_low_s16(w1))));
                ac1=vfmaq_f32(ac1, vld1q_f32(xs+i+12), vcvtq_f32_s32(vmovl_s16(vget_high_s16(w1)))); }
            a=vaddvq_f32(vaddq_f32(ac0,ac1));
#endif
            for(;i<I;i++){ uint8_t byte=w[i>>2]; int sh=(i&3)*2; a += xs[i]*(float)((int)((byte>>sh)&3)-2); }
            y[(int64_t)s*O+o]=a*sc; } }
}
/* ---- KERNEL INTERI (IDOT): attivazioni quantizzate a int8 per riga ---- */
#if defined(__AVX512VNNI__) && defined(__AVX512BW__)
#define IDOT_KERNEL "avx512-vnni"
#elif defined(__AVX2__)
#define IDOT_KERNEL "avx2"
#elif defined(__ARM_NEON)
#define IDOT_KERNEL "neon"
#else
#define IDOT_KERNEL "scalar"
#endif
static int g_idot=1;
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
static int g_i4s=1;
#else
static int g_i4s=2;
#endif
static inline float qrow_i8(const float *x, int8_t *q, int I){
    float amax=0; for(int i=0;i<I;i++){ float a=fabsf(x[i]); if(a>amax)amax=a; }
    float s=amax/127.f; if(s<1e-12f) s=1e-12f; float inv=1.f/s;
    for(int i=0;i<I;i++) q[i]=(int8_t)lrintf(x[i]*inv);
    return s;
}
#ifdef __AVX2__
static inline int hsum256_i32(__m256i v){
    __m128i lo=_mm256_castsi256_si128(v), hi=_mm256_extracti128_si256(v,1);
    lo=_mm_add_epi32(lo,hi); lo=_mm_hadd_epi32(lo,lo); lo=_mm_hadd_epi32(lo,lo);
    return _mm_cvtsi128_si32(lo);
}
#endif
static inline int32_t dot_i8i8(const int8_t *w, const int8_t *x, int I){
    int32_t sum=0; int i=0;
#if defined(__AVX512VNNI__) && defined(__AVX512BW__)
    __m512i acc=_mm512_setzero_si512();
    for(;i+64<=I;i+=64){
        __m512i wv=_mm512_loadu_si512((const void*)(w+i));
        __m512i xv=_mm512_loadu_si512((const void*)(x+i));
        __mmask64 neg=_mm512_movepi8_mask(wv);
        __m512i xs=_mm512_mask_sub_epi8(xv,neg,_mm512_setzero_si512(),xv);
        acc=_mm512_dpbusd_epi32(acc,_mm512_abs_epi8(wv),xs);
    }
    sum=_mm512_reduce_add_epi32(acc);
#elif defined(__AVX2__)
    __m256i acc=_mm256_setzero_si256(); const __m256i ones=_mm256_set1_epi16(1);
    for(;i+32<=I;i+=32){
        __m256i wv=_mm256_loadu_si256((const __m256i*)(w+i));
        __m256i xv=_mm256_loadu_si256((const __m256i*)(x+i));
        __m256i p=_mm256_maddubs_epi16(_mm256_sign_epi8(wv,wv),_mm256_sign_epi8(xv,wv));
        acc=_mm256_add_epi32(acc,_mm256_madd_epi16(p,ones));
    }
    sum=hsum256_i32(acc);
#elif defined(__ARM_NEON)
    int32x4_t acc=vdupq_n_s32(0);
    for(;i+16<=I;i+=16){
        int8x16_t wv=vld1q_s8(w+i), xv=vld1q_s8(x+i);
#if defined(__ARM_FEATURE_DOTPROD)
        acc=vdotq_s32(acc,wv,xv);
#else
        int16x8_t p=vmull_s8(vget_low_s8(wv),vget_low_s8(xv));
        p=vmlal_s8(p,vget_high_s8(wv),vget_high_s8(xv));
        acc=vpadalq_s16(acc,p);
#endif
    }
    sum=vaddvq_s32(acc);
#endif
    for(;i<I;i++) sum+=(int32_t)w[i]*x[i];
    return sum;
}
static inline int32_t dot_i4i8(const uint8_t *w4, const int8_t *x, int I){
    int32_t sum=0; int i=0;
#if defined(__AVX512VNNI__) && defined(__AVX512BW__)
    const __m256i m4v=_mm256_set1_epi8(0x0F);
    const __m512i b8v=_mm512_set1_epi8(8);
    const __m512i xidx=_mm512_setr_epi64(0,1,4,5,2,3,6,7);
    __m512i acc=_mm512_setzero_si512();
    for(;i+64<=I;i+=64){
        __m256i by=_mm256_loadu_si256((const __m256i*)(w4+(i>>1)));
        __m256i lo=_mm256_and_si256(by,m4v), hi=_mm256_and_si256(_mm256_srli_epi16(by,4),m4v);
        __m256i z0=_mm256_unpacklo_epi8(lo,hi), z1=_mm256_unpackhi_epi8(lo,hi);
        __m512i wv=_mm512_sub_epi8(_mm512_inserti64x4(_mm512_castsi256_si512(z0),z1,1),b8v);
        __m512i xv=_mm512_permutexvar_epi64(xidx,_mm512_loadu_si512((const void*)(x+i)));
        __mmask64 neg=_mm512_movepi8_mask(wv);
        __m512i xs=_mm512_mask_sub_epi8(xv,neg,_mm512_setzero_si512(),xv);
        acc=_mm512_dpbusd_epi32(acc,_mm512_abs_epi8(wv),xs);
    }
    sum=_mm512_reduce_add_epi32(acc);
#elif defined(__AVX2__)
    const __m128i m4=_mm_set1_epi8(0x0F); const __m256i b8=_mm256_set1_epi8(8);
    const __m256i ones=_mm256_set1_epi16(1);
    __m256i acc=_mm256_setzero_si256();
    for(;i+32<=I;i+=32){
        __m128i by=_mm_loadu_si128((const __m128i*)(w4+(i>>1)));
        __m128i lo=_mm_and_si128(by,m4), hi=_mm_and_si128(_mm_srli_epi16(by,4),m4);
        __m128i n0=_mm_unpacklo_epi8(lo,hi), n1=_mm_unpackhi_epi8(lo,hi);
        __m256i wv=_mm256_sub_epi8(_mm256_set_m128i(n1,n0),b8);
        __m256i xv=_mm256_loadu_si256((const __m256i*)(x+i));
        __m256i p=_mm256_maddubs_epi16(_mm256_sign_epi8(wv,wv),_mm256_sign_epi8(xv,wv));
        acc=_mm256_add_epi32(acc,_mm256_madd_epi16(p,ones));
    }
    sum=hsum256_i32(acc);
#elif defined(__ARM_NEON)
    const uint8x16_t m4q=vdupq_n_u8(0x0F); const int8x16_t b8q=vdupq_n_s8(8);
    int32x4_t acc=vdupq_n_s32(0);
    for(;i+32<=I;i+=32){
        uint8x16_t by=vld1q_u8(w4+(i>>1));
        uint8x16x2_t z=vzipq_u8(vandq_u8(by,m4q), vshrq_n_u8(by,4));
        int8x16_t w0=vsubq_s8(vreinterpretq_s8_u8(z.val[0]),b8q);
        int8x16_t w1=vsubq_s8(vreinterpretq_s8_u8(z.val[1]),b8q);
        int8x16_t x0=vld1q_s8(x+i), x1=vld1q_s8(x+i+16);
#if defined(__ARM_FEATURE_DOTPROD)
        acc=vdotq_s32(acc,w0,x0); acc=vdotq_s32(acc,w1,x1);
#else
        int16x8_t p=vmull_s8(vget_low_s8(w0),vget_low_s8(x0));
        p=vmlal_s8(p,vget_high_s8(w0),vget_high_s8(x0));
        acc=vpadalq_s16(acc,p);
        p=vmull_s8(vget_low_s8(w1),vget_low_s8(x1));
        p=vmlal_s8(p,vget_high_s8(w1),vget_high_s8(x1));
        acc=vpadalq_s16(acc,p);
#endif
    }
    sum=vaddvq_s32(acc);
#endif
    for(;i+1<I;i+=2){ uint8_t b=w4[i>>1]; sum+=((int)(b&0xF)-8)*x[i]+((int)(b>>4)-8)*x[i+1]; }
    if(i<I){ uint8_t b=w4[i>>1]; sum+=((int)(b&0xF)-8)*x[i]; }
    return sum;
}
static void matmul_q_idot(float *y, const int8_t *xq, const float *sx, const int8_t *q,
                          const float *scale, int S, int I, int O){
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){ const int8_t *w=q+(int64_t)o*I; float sc=scale[o];
        for(int s=0;s<S;s++) y[(int64_t)s*O+o]=(float)dot_i8i8(w,xq+(int64_t)s*I,I)*sc*sx[s]; }
}
static void matmul_i4_idot(float *y, const int8_t *xq, const float *sx, const uint8_t *q4,
                           const float *scale, int S, int I, int O){
    int rb=(I+1)/2;
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){ const uint8_t *w=q4+(int64_t)o*rb; float sc=scale[o];
        for(int s=0;s<S;s++) y[(int64_t)s*O+o]=(float)dot_i4i8(w,xq+(int64_t)s*I,I)*sc*sx[s]; }
}

static void matmul_qt(float *y, const float *x, QT *w, int S){
#ifdef COLI_CUDA
    if(g_cuda_enabled && w->cuda_eligible && !w->cuda_failed && !omp_in_parallel()){
        const void *weights = w->fmt==0 ? (const void*)w->qf
                            : w->fmt==1 ? (const void*)w->q8 : (const void*)w->q4;
        if(coli_cuda_matmul(&w->cuda,y,x,weights,w->s,w->fmt,S,w->I,w->O,w->cuda_device)) return;
        w->cuda_failed=1;
        fprintf(stderr,"[CUDA] tensor [%d,%d] su device %d disabilitato dopo errore; fallback CPU\n",
            w->O,w->I,w->cuda_device);
    }
#endif
#ifdef COLI_METAL
    if(g_metal_enabled && w->metal_eligible && !w->metal_failed){
        const void *weights = w->fmt==0 ? (const void*)w->qf
                            : w->fmt==1 ? (const void*)w->q8 : (const void*)w->q4;
        if(coli_metal_matmul(&w->metal,y,x,weights,w->s,w->fmt,S,w->I,w->O)) return;
        w->metal_failed=1;
        fprintf(stderr,"[Metal] tensor [%d,%d] disabilitato dopo errore; fallback CPU\n",w->O,w->I);
    }
#endif
    if(w->fmt==0){ matmul(y,x,w->qf,S,w->I,w->O); return; }
    if(g_idot && (w->fmt==1 || (w->fmt==2 && S>=g_i4s))){
        int I=w->I;
        int8_t *xq=malloc((size_t)S*I); float sxb[64]; float *sx=S<=64?sxb:falloc(S);
        for(int s=0;s<S;s++) sx[s]=qrow_i8(x+(int64_t)s*I, xq+(int64_t)s*I, I);
        if(w->fmt==1) matmul_q_idot(y,xq,sx,w->q8,w->s,S,I,w->O);
        else matmul_i4_idot(y,xq,sx,w->q4,w->s,S,I,w->O);
        free(xq); if(sx!=sxb) free(sx);
        return;
    }
    if(w->fmt==1) matmul_q(y,x,w->q8,w->s,S,w->I,w->O);
    else if(w->fmt==3) matmul_i2(y,x,w->q4,w->s,S,w->I,w->O);
    else matmul_i4(y,x,w->q4,w->s,S,w->I,w->O);
}

/* quantizza w[O,I] f32 -> int8 q[O,I] + scala[O] simmetrica per riga */
static void quantize_rows(const float *w, int8_t *q, float *scale, int O, int I, int bits){
    int qmax=(1<<(bits-1))-1;
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){ const float *wr=w+(int64_t)o*I; float amax=0;
        for(int i=0;i<I;i++){ float a=fabsf(wr[i]); if(a>amax)amax=a; }
        float s=amax/qmax; if(s<1e-8f)s=1e-8f; scale[o]=s;
        int8_t *qr=q+(int64_t)o*I;
        for(int i=0;i<I;i++){ int v=(int)lrintf(wr[i]/s); if(v>qmax)v=qmax; if(v<-qmax-1)v=-qmax-1; qr[i]=(int8_t)v; }
    }
}
static void pack_int4(const float *w, uint8_t *q4, float *scale, int O, int I, int bits){
    int qmax=(1<<(bits-1))-1, rb=(I+1)/2;
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){ const float *wr=w+(int64_t)o*I; float amax=0;
        for(int i=0;i<I;i++){ float a=fabsf(wr[i]); if(a>amax)amax=a; }
        float s=amax/qmax; if(s<1e-8f)s=1e-8f; scale[o]=s;
        uint8_t *qr=q4+(int64_t)o*rb;
        for(int i=0;i<I;i+=2){
            int v0=(int)lrintf(wr[i]/s); if(v0>qmax)v0=qmax; if(v0<-8)v0=-8;
            int v1=0; if(i+1<I){ v1=(int)lrintf(wr[i+1]/s); if(v1>qmax)v1=qmax; if(v1<-8)v1=-8; }
            qr[i>>1] = (uint8_t)((v0+8) | ((v1+8)<<4));
        }
    }
}
static void pack_int2(const float *w, uint8_t *q2, float *scale, int O, int I, int bits){
    int qmax=(1<<(bits-1))-1, rb=(I+3)/4;
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){ const float *wr=w+(int64_t)o*I; float amax=0;
        for(int i=0;i<I;i++){ float a=fabsf(wr[i]); if(a>amax)amax=a; }
        float s=amax/qmax; if(s<1e-8f)s=1e-8f; scale[o]=s;
        uint8_t *qr=q2+(int64_t)o*rb;
        for(int i=0;i<I;i+=4){ uint8_t byte=0;
            for(int k=0;k<4 && i+k<I;k++){ int v=(int)lrintf(wr[i+k]/s); if(v>qmax)v=qmax; if(v<-2)v=-2; byte|=(uint8_t)((v+2)<<(k*2)); }
            qr[i>>2]=byte;
        }
    }
}

static int g_nopack=0;
static int g_drop=0;
static int g_prefetch=0;
static int g_direct=0;
static float g_temp=-1;
static float g_nuc=0.95f;
static int g_topk=0;
static float g_topp=0;
static int g_spec=1;
static int g_draft=0;
static int g_looka=0;
static int64_t la_hit[3], la_tot[3];
static int la_pred[2][130][16]; static signed char la_val[2][130];
static int g_pilot=0;
static int g_pilot_k=8;
static void qt_alloc(QT *t, int O, int I, int bits){
    t->O=O; t->I=I; t->qf=NULL; t->q8=NULL; t->q4=NULL; t->s=NULL;
    if(bits>=16){ t->fmt=0; t->qf=falloc((int64_t)O*I); }
    else if(bits>=5 || g_nopack){ t->fmt=1; t->q8=malloc((int64_t)O*I); t->s=falloc(O); }
    else if(bits>=3){ t->fmt=2; t->q4=malloc((int64_t)O*((I+1)/2)); t->s=falloc(O); }
    else { t->fmt=3; t->q4=malloc((int64_t)O*((I+3)/4)); t->s=falloc(O); }
}
static void qt_fill(QT *t, const float *w, int bits){
    if(t->fmt==0) memcpy(t->qf, w, (int64_t)t->O*t->I*sizeof(float));
    else if(t->fmt==1) quantize_rows(w, t->q8, t->s, t->O, t->I, bits);
    else if(t->fmt==3) pack_int2(w, t->q4, t->s, t->O, t->I, bits);
    else pack_int4(w, t->q4, t->s, t->O, t->I, bits);
}

static void rmsnorm(float *out, const float *x, const float *w, int D, float eps){
    double ms=0; for(int i=0;i<D;i++) ms+=(double)x[i]*x[i];
    float r=1.f/sqrtf((float)(ms/D)+eps); for(int i=0;i<D;i++) out[i]=x[i]*r*w[i];
}
static void softmax(float *x,int n){ float m=-1e30f; for(int i=0;i<n;i++) if(x[i]>m)m=x[i];
    float s=0; for(int i=0;i<n;i++){x[i]=expf(x[i]-m);s+=x[i];} for(int i=0;i<n;i++) x[i]/=s; }
static inline float sigmoidf(float x){ return 1.f/(1.f+expf(-x)); }
static inline float siluf(float x){ return x/(1.f+expf(-x)); }

/* RoPE standard (paired) su i primi rope_dim dimensioni di v.
 * GLM-4.5 usa RoPE NON interleaved: le coppie sono (v[0],v[1]), (v[2],v[3]), ... */
static void rope_half(float *v, int pos, int rope_dim, float theta){
    int half = rope_dim/2;
    for(int j=0;j<half;j++){
        float inv = powf(theta, -2.0f*j/rope_dim);
        float ang = pos*inv, cs=cosf(ang), sn=sinf(ang);
        float a=v[2*j], b=v[2*j+1];
        v[2*j]   = a*cs - b*sn;
        v[2*j+1] = b*cs + a*sn;
    }
}

/* ---------- config ---------- */
static jval* cfg_root(const char *snap, char **arena){
    char p[2048]; snprintf(p,sizeof(p),"%s/config.json",snap);
    FILE *f=fopen(p,"rb"); if(!f){perror(p);exit(1);}
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    char *b=malloc(n+1); if(fread(b,1,n,f)!=(size_t)n){} b[n]=0; fclose(f);
    return json_parse(b,arena);
}
static int gi(jval*r,const char*k){ jval*v=json_get(r,k); return v?(int)v->num:0; }
static void load_cfg(Cfg *c, const char *snap){
    char *ar=NULL; jval *r=cfg_root(snap,&ar);
    c->hidden=gi(r,"hidden_size"); c->n_layers=gi(r,"num_hidden_layers");
    c->n_heads=gi(r,"num_attention_heads"); c->n_kv_heads=gi(r,"num_key_value_heads");
    c->head_dim=gi(r,"head_dim");
    c->n_experts=gi(r,"n_routed_experts");
    c->topk=gi(r,"num_experts_per_tok"); c->moe_inter=gi(r,"moe_intermediate_size");
    c->dense_inter=gi(r,"intermediate_size"); c->first_dense=gi(r,"first_k_dense_replace");
    c->n_shared=gi(r,"n_shared_experts"); c->vocab=gi(r,"vocab_size");
    c->n_group=gi(r,"n_group"); c->topk_group=gi(r,"topk_group");
    jval *nt=json_get(r,"norm_topk_prob"); c->norm_topk=(nt&&nt->t==J_BOOL)?nt->boolean:0;
    jval *ep=json_get(r,"rms_norm_eps"); c->eps=ep?(float)ep->num:1e-5f;
    jval *rs=json_get(r,"routed_scaling_factor"); c->routed_scale=rs?(float)rs->num:1.f;
    /* rope_theta: puo' essere in rope_parameters o al top level */
    jval *rp=json_get(r,"rope_parameters");
    jval *th=rp?json_get(rp,"rope_theta"):NULL;
    if(!th) th=json_get(r,"rope_theta");
    c->theta = th?(float)th->num:10000.f;
    /* partial_rotary_factor: in rope_parameters o al top level */
    jval *prf=rp?json_get(rp,"partial_rotary_factor"):NULL;
    if(!prf) prf=json_get(r,"partial_rotary_factor");
    c->partial_rotary_factor = prf?(float)prf->num:0.5f;
    /* attention_bias */
    jval *ab=json_get(r,"attention_bias");
    c->attention_bias = (ab&&ab->t==J_BOOL)?ab->boolean:0;
    /* head_dim default: hidden / n_heads se mancante */
    if(!c->head_dim) c->head_dim = c->hidden / c->n_heads;
    c->attn_scale = 1.f / sqrtf((float)c->head_dim);
    /* stop tokens */
    c->n_stop=0;
    jval *eo=json_get(r,"eos_token_id");
    if(eo){ if(eo->t==J_NUM) c->stop_ids[c->n_stop++]=(int)eo->num;
            else if(eo->t==J_ARR) for(int i=0;i<eo->len && c->n_stop<8;i++)
                c->stop_ids[c->n_stop++]=(int)eo->kids[i]->num; }
    /* MTP */
    c->has_mtp = 0;  /* not supported in this engine */
    if(c->n_group!=1){ fprintf(stderr,"questo motore assume n_group=1\n"); exit(1); }
    /* VALIDAZIONE: range check */
    #define CKR(name,v,lo,hi) if((v)<(lo)||(v)>(hi)){ \
        fprintf(stderr,"config: %s=%d fuori range [%d,%d]\n",name,(int)(v),(int)(lo),(int)(hi)); exit(1); }
    CKR("hidden_size",c->hidden,1,1<<20)         CKR("num_hidden_layers",c->n_layers,1,128)
    CKR("num_attention_heads",c->n_heads,1,1024)  CKR("num_key_value_heads",c->n_kv_heads,1,c->n_heads)
    CKR("head_dim",c->head_dim,1,1<<16)
    CKR("n_routed_experts",c->n_experts,1,4096)
    CKR("num_experts_per_tok",c->topk,1,64)       CKR("moe_intermediate_size",c->moe_inter,1,1<<20)
    CKR("intermediate_size",c->dense_inter,1,1<<24) CKR("first_k_dense_replace",c->first_dense,0,c->n_layers)
    CKR("n_shared_experts",c->n_shared,0,64)
    CKR("vocab_size",c->vocab,1,1<<24)
    #undef CKR
    free(ar);
}

/* costruisce un QT [O,I] dal disco */
static void qt_from_disk(Model *m, const char *name, int O, int I, int bits, int drop, QT *t){
    char sn[300]; snprintf(sn,sizeof(sn),"%s.qs",name);
    if(st_has(&m->S,sn)){
        int64_t nb=st_nbytes(&m->S,name);
        int fmt = (nb==(int64_t)O*I)?1 : (nb==(int64_t)O*((I+1)/2))?2 : 3;
        if(fmt==1){ if(t->fmt!=1||!t->q8){ t->fmt=1; t->O=O; t->I=I; t->q8=malloc(nb); t->s=falloc(O); } st_read_raw(&m->S,name,t->q8,drop); }
        else      { if(t->fmt!=fmt||!t->q4){ t->fmt=fmt; t->O=O; t->I=I; t->q4=malloc(nb); t->s=falloc(O); } st_read_raw(&m->S,name,t->q4,drop); }
        st_read_f32(&m->S,sn,t->s,drop);
    } else {
        if(!t->qf && !t->q8 && !t->q4) qt_alloc(t,O,I,bits);
        if(t->fmt==0) st_read_f32(&m->S,name,t->qf,drop);
        else { float *tmp=falloc((int64_t)O*I); st_read_f32(&m->S,name,tmp,drop); qt_fill(t,tmp,bits); free(tmp); }
    }
}
static QT qt_load(Model *m, const char *name, int O, int I, int bits){
    QT t; memset(&t,0,sizeof(t)); qt_from_disk(m,name,O,I,bits,0,&t);
#ifdef COLI_CUDA
    if(g_cuda_enabled&&g_cuda_dense){
        t.cuda_eligible=1;
        int slot=g_cuda_rr++%g_cuda_ndev; t.cuda_device=g_cuda_devices[slot];
        g_cuda_dense_projected[slot]+=qt_bytes(&t);
    }
#endif
#ifdef COLI_METAL
    if(g_metal_enabled&&g_metal_dense) t.metal_eligible=1;
#endif
    return t;
}
static float *ld(Model *m, const char *name){   /* tensore 1D f32 residente (norme/bias) */
    int64_t n=st_numel(&m->S,name); if(n<0){fprintf(stderr,"manca %s\n",name);exit(1);}
    float *p=falloc(n); st_read_f32(&m->S,name,p,0); return p;
}

static void model_init(Model *m, const char *snap, int cap, int ebits, int dbits){
    memset(m,0,sizeof(*m)); m->ebits=ebits; m->dbits=dbits;
    load_cfg(&m->c,snap); st_init(&m->S,snap);
    Cfg *c=&m->c; char nm[256]; int H=c->n_heads, Hkv=c->n_kv_heads, D=c->hidden, hd=c->head_dim;
    int io_bits = dbits>=8 ? 16 : dbits;
    m->embed   = qt_load(m,"model.embed_tokens.weight", c->vocab, D, io_bits);
    m->lm_head = qt_load(m,"lm_head.weight", c->vocab, D, io_bits);
    m->final_norm = ld(m,"model.norm.weight");
    m->L=calloc(c->n_layers,sizeof(Layer));
    m->ecap=cap; m->ecache=calloc(c->n_layers,sizeof(ESlot*)); m->ecn=calloc(c->n_layers,sizeof(int));
    m->eroute=calloc(c->n_layers,sizeof(int*)); m->enr=calloc(c->n_layers,sizeof(int));
    m->pin=calloc(c->n_layers,sizeof(ESlot*)); m->npin=calloc(c->n_layers,sizeof(int));
    m->eusage=calloc(c->n_layers,sizeof(uint32_t*)); m->eheat=calloc(c->n_layers,sizeof(uint32_t*));
    for(int i=0;i<c->n_layers;i++){
        Layer *l=&m->L[i];
        #define P(s) (snprintf(nm,sizeof(nm),"model.layers.%d." s,i),nm)
        l->in_ln=ld(m,P("input_layernorm.weight"));
        l->post_ln=ld(m,P("post_attention_layernorm.weight"));
        /* GQA attention: Q/K/V/O projections */
        l->q_proj = qt_load(m,P("self_attn.q_proj.weight"), H*hd, D, dbits);
        l->k_proj = qt_load(m,P("self_attn.k_proj.weight"), Hkv*hd, D, dbits);
        l->v_proj = qt_load(m,P("self_attn.v_proj.weight"), Hkv*hd, D, dbits);
        l->o_proj = qt_load(m,P("self_attn.o_proj.weight"), D, H*hd, dbits);
        /* biases: Q/K/V have bias, o_proj does NOT */
        if(c->attention_bias){
            l->q_bias = ld(m,P("self_attn.q_proj.bias"));
            l->k_bias = ld(m,P("self_attn.k_proj.bias"));
            l->v_bias = ld(m,P("self_attn.v_proj.bias"));
        }
        l->sparse = (i >= c->first_dense);
        if(!l->sparse){
            l->gate_proj = qt_load(m,P("mlp.gate_proj.weight"), c->dense_inter, D, dbits);
            l->up_proj   = qt_load(m,P("mlp.up_proj.weight"),   c->dense_inter, D, dbits);
            l->down_proj = qt_load(m,P("mlp.down_proj.weight"), D, c->dense_inter, dbits);
        } else {
            l->router=ld(m,P("mlp.gate.weight"));
            l->router_bias=ld(m,P("mlp.gate.e_score_correction_bias"));
            int sI=c->moe_inter*c->n_shared;
            l->sh_gate = qt_load(m,P("mlp.shared_experts.gate_proj.weight"), sI, D, dbits);
            l->sh_up   = qt_load(m,P("mlp.shared_experts.up_proj.weight"),   sI, D, dbits);
            l->sh_down = qt_load(m,P("mlp.shared_experts.down_proj.weight"), D, sI, dbits);
            m->ecache[i]=calloc(cap,sizeof(ESlot));
            m->eroute[i]=calloc(c->topk,sizeof(int));
            m->eusage[i]=calloc(c->n_experts,sizeof(uint32_t));
            m->eheat[i]=calloc(c->n_experts,sizeof(uint32_t));
        }
        #undef P
    }
    /* byte della parte DENSA residente */
    int64_t rb=qt_bytes(&m->embed)+qt_bytes(&m->lm_head);
    for(int i=0;i<c->n_layers;i++){ Layer *l=&m->L[i];
        rb+=qt_bytes(&l->q_proj)+qt_bytes(&l->k_proj)+qt_bytes(&l->v_proj)+qt_bytes(&l->o_proj);
        if(!l->sparse) rb+=qt_bytes(&l->gate_proj)+qt_bytes(&l->up_proj)+qt_bytes(&l->down_proj);
        else rb+=qt_bytes(&l->sh_gate)+qt_bytes(&l->sh_up)+qt_bytes(&l->sh_down);
    }
    m->resident_bytes=rb;
}

/* embed: dequantizza la riga del token (scala per-riga) in x[hidden] */
static void embed_row(Model *m, int tok, float *x){
    int D=m->c.hidden; QT *e=&m->embed;
    if(e->fmt==0){ memcpy(x, e->qf+(int64_t)tok*D, D*sizeof(float)); return; }
    if(e->fmt==1){ const int8_t *q=e->q8+(int64_t)tok*D; float s=e->s[tok];
        for(int i=0;i<D;i++) x[i]=(float)q[i]*s; return; }
    if(e->fmt==2){ const uint8_t *q=e->q4+(int64_t)tok*((D+1)/2); float s=e->s[tok];
        for(int i=0;i<D;i+=2){ uint8_t byte=q[i>>1]; x[i]=(float)((int)(byte&0xF)-8)*s;
            if(i+1<D) x[i+1]=(float)((int)(byte>>4)-8)*s; }
        return; }
    const uint8_t *q=e->q4+(int64_t)tok*((D+3)/4); float s=e->s[tok];
    for(int i=0;i<D;i++){ uint8_t byte=q[i>>2]; int sh=(i&3)*2; x[i]=(float)((int)((byte>>sh)&3)-2)*s; }
}

/* carica un expert nello slot — identico a glm.c */
static void expert_load(Model *m, int layer, int eid, ESlot *s){
#ifdef COLI_CUDA
    if(s->eid!=eid){ qt_cuda_reset(&s->g); qt_cuda_reset(&s->u); qt_cuda_reset(&s->d); }
#endif
#ifdef COLI_METAL
    if(s->eid!=eid){ qt_metal_reset(&s->g); qt_metal_reset(&s->u); qt_metal_reset(&s->d); }
#endif
    Cfg *c=&m->c; int I=c->moe_inter, D=c->hidden, b=m->ebits;
    char nm[3][288]; const char *suf[3]={"gate_proj","up_proj","down_proj"};
    for(int k=0;k<3;k++) snprintf(nm[k],sizeof(nm[k]),"model.layers.%d.mlp.experts.%d.%s.weight",layer,eid,suf[k]);
    char qn[300]; snprintf(qn,sizeof(qn),"%s.qs",nm[0]);
    if(!st_has(&m->S,qn)){
        qt_from_disk(m,nm[0],I,D,b,g_drop,&s->g);
        qt_from_disk(m,nm[1],I,D,b,g_drop,&s->u);
        qt_from_disk(m,nm[2],D,I,b,g_drop,&s->d);
        s->eid=eid; return;
    }
    st_tensor *tw[3], *tq[3];
    for(int k=0;k<3;k++){
        tw[k]=st_find(&m->S,nm[k]);
        snprintf(qn,sizeof(qn),"%s.qs",nm[k]); tq[k]=st_find(&m->S,qn);
        if(!tw[k]||!tq[k]){ fprintf(stderr,"manca %s\n",nm[k]); exit(1); }
    }
    int64_t wtot=tw[0]->nbytes+tw[1]->nbytes+tw[2]->nbytes;
    int64_t ftot=(tq[0]->nbytes+tq[1]->nbytes+tq[2]->nbytes)/4;
    if(!s->slab || wtot+8192 > s->slab_cap){
        free(s->slab);
        if(posix_memalign((void**)&s->slab,4096,wtot+8192)){fprintf(stderr,"OOM slab\n");exit(1);}
        s->slab_cap=wtot+8192;
    }
    if(!s->fslab || ftot > s->fslab_cap){ free(s->fslab); s->fslab=falloc(ftot); s->fslab_cap=ftot; }
    int ord[3]={0,1,2};
    for(int a=0;a<3;a++) for(int bb=a+1;bb<3;bb++) if(tw[ord[bb]]->off<tw[ord[a]]->off){ int t=ord[a]; ord[a]=ord[bb]; ord[bb]=t; }
    int contig = tw[ord[0]]->fd==tw[ord[1]]->fd && tw[ord[1]]->fd==tw[ord[2]]->fd
              && tw[ord[0]]->off+tw[ord[0]]->nbytes==tw[ord[1]]->off
              && tw[ord[1]]->off+tw[ord[1]]->nbytes==tw[ord[2]]->off;
    int64_t pos[3]; int done=0;
    if(contig){
        int64_t off0=tw[ord[0]]->off;
        int dfd = g_direct ? st_direct_fd(&m->S, tw[ord[0]]->fd) : -1;
        if(dfd>=0){
            int64_t base=off0 & ~4095LL, need=(off0-base)+wtot;
            int64_t len=(need+4095)&~4095LL;
            ssize_t r=pread(dfd, s->slab, len, base);
            if(r>=need){
                pos[ord[0]]=off0-base; pos[ord[1]]=pos[ord[0]]+tw[ord[0]]->nbytes;
                pos[ord[2]]=pos[ord[1]]+tw[ord[1]]->nbytes; done=1;
            }
        }
        if(!done){
            if(pread(tw[ord[0]]->fd, s->slab, wtot, off0)!=wtot){ perror("pread expert"); exit(1); }
            pos[ord[0]]=0; pos[ord[1]]=tw[ord[0]]->nbytes; pos[ord[2]]=tw[ord[0]]->nbytes+tw[ord[1]]->nbytes; done=1;
        }
    }
    if(!done){
        int64_t o=0;
        for(int a=0;a<3;a++){ int k=ord[a];
            if(pread(tw[k]->fd, s->slab+o, tw[k]->nbytes, tw[k]->off)!=tw[k]->nbytes){ perror("pread expert"); exit(1); }
            pos[k]=o; o+=tw[k]->nbytes; }
    }
    float *fp[3]; int64_t fo=0;
    for(int k=0;k<3;k++){
        if(pread(tq[k]->fd, (char*)(s->fslab+fo), tq[k]->nbytes, tq[k]->off)!=tq[k]->nbytes){ perror("pread qs"); exit(1); }
        fp[k]=s->fslab+fo; fo+=tq[k]->nbytes/4; }
    if(g_drop){
        posix_fadvise(tw[ord[0]]->fd, tw[ord[0]]->off, wtot, POSIX_FADV_DONTNEED);
        for(int k=0;k<3;k++) posix_fadvise(tq[k]->fd, tq[k]->off, tq[k]->nbytes, POSIX_FADV_DONTNEED);
    }
    QT *qt[3]={&s->g,&s->u,&s->d}; int OO[3]={I,I,D}, II[3]={D,D,I};
    for(int k=0;k<3;k++){
        int64_t nb=tw[k]->nbytes;
        int fmt = (nb==(int64_t)OO[k]*II[k])?1 : (nb==(int64_t)OO[k]*((II[k]+1)/2))?2 : 3;
        qt[k]->fmt=fmt; qt[k]->O=OO[k]; qt[k]->I=II[k]; qt[k]->qf=NULL;
        qt[k]->q8=(int8_t*)(s->slab+pos[k]); qt[k]->q4=s->slab+pos[k]; qt[k]->s=fp[k];
    }
    s->eid=eid;
}

static void expert_prefetch(Model *m, int layer, int eid){
    char nm[300];
    const char *suf[3]={"gate_proj.weight","up_proj.weight","down_proj.weight"};
    for(int k=0;k<3;k++){
        snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.%d.%s",layer,eid,suf[k]); st_prefetch(&m->S,nm);
        char qs[320]; snprintf(qs,sizeof(qs),"%s.qs",nm); st_prefetch(&m->S,qs);
    }
}

/* ---- GQA Attention con partial RoPE ----
 * x[S, hidden] -> out[S, hidden], updates KV cache at pos_base..pos_base+S-1 */
static void attention(Model *m, Layer *l, int layer, float *x, int S, int pos_base, float *out){
    Cfg *c=&m->c; int H=c->n_heads, Hkv=c->n_kv_heads, hd=c->head_dim;
    int rope_dim = (int)(hd * c->partial_rotary_factor);  /* 64 of 128 for GLM-4.5 */
    int rep = H / Hkv;  /* GQA repeat factor (96/8 = 12 for GLM-4.5-Air) */
    double ta0=now_s();

    float *Q = falloc((int64_t)S * H * hd);
    float *Knew = falloc((int64_t)S * Hkv * hd);
    float *Vnew = falloc((int64_t)S * Hkv * hd);
    float *ctx = falloc((int64_t)S * H * hd);

    /* 1) Project Q, K, V */
    matmul_qt(Q, x, &l->q_proj, S);
    matmul_qt(Knew, x, &l->k_proj, S);
    matmul_qt(Vnew, x, &l->v_proj, S);

    /* Add bias if present (Q/K/V only) */
    if(c->attention_bias){
        for(int s=0;s<S;s++){
            float *qs=Q+(int64_t)s*H*hd;
            for(int i=0;i<H*hd;i++) qs[i]+=l->q_bias[i];
            float *ks=Knew+(int64_t)s*Hkv*hd;
            float *vs=Vnew+(int64_t)s*Hkv*hd;
            for(int i=0;i<Hkv*hd;i++){
                ks[i]+=l->k_bias[i];
                vs[i]+=l->v_bias[i];
            }
        }
    }

    /* 2) Apply partial RoPE (first rope_dim dimensions of each head) */
    for(int s=0;s<S;s++){
        int pos=pos_base+s;
        for(int h=0;h<H;h++)
            rope_half(Q + (int64_t)(s*H+h)*hd, pos, rope_dim, c->theta);
        for(int h=0;h<Hkv;h++)
            rope_half(Knew + (int64_t)(s*Hkv+h)*hd, pos, rope_dim, c->theta);
    }

    /* 3) Store K, V in cache */
    for(int s=0;s<S;s++){
        int pos=pos_base+s;
        memcpy(m->Kc[layer] + (int64_t)pos*Hkv*hd,
               Knew + (int64_t)s*Hkv*hd, Hkv*hd*sizeof(float));
        memcpy(m->Vc[layer] + (int64_t)pos*Hkv*hd,
               Vnew + (int64_t)s*Hkv*hd, Hkv*hd*sizeof(float));
    }

    /* 4) Attention: Q @ K^T * scale, causal mask, softmax, @ V */
    #pragma omp parallel for collapse(2) schedule(static)
    for(int s=0;s<S;s++) for(int h=0;h<H;h++){
        int pos=pos_base+s;
        int kv_h = h / rep;  /* GQA: map query head to KV head */
        const float *qh = Q + (int64_t)(s*H+h)*hd;
        /* dynamic alloc for large context support */
        float *sc = (float*)malloc((pos+1)*sizeof(float));
        for(int t=0;t<=pos;t++){
            const float *kt = m->Kc[layer] + (int64_t)(t*Hkv+kv_h)*hd;
            float a=0;
            for(int d=0;d<hd;d++) a+=qh[d]*kt[d];
            sc[t]=a*c->attn_scale;
        }
        softmax(sc, pos+1);
        float *cx = ctx + (int64_t)(s*H+h)*hd;
        memset(cx, 0, hd*sizeof(float));
        for(int t=0;t<=pos;t++){
            const float *vt = m->Vc[layer] + (int64_t)(t*Hkv+kv_h)*hd;
            float a=sc[t];
            for(int d=0;d<hd;d++) cx[d]+=a*vt[d];
        }
        free(sc);
    }

    /* 5) Output projection (no bias on o_proj) */
    matmul_qt(out, ctx, &l->o_proj, S);

    free(Q); free(Knew); free(Vnew); free(ctx);
    m->t_attn += now_s()-ta0;
}

/* MoE GLM su x[S,hidden] -> out — identico a glm.c */
static void moe(Model *m, Layer *l, int layer, float *x, int S, float *out){
    Cfg *c=&m->c; int D=c->hidden, E=c->n_experts, K=c->topk, I=c->moe_inter;
    float *logit=falloc(E), *sig=falloc(E), *choice=falloc(E);
    int sI=c->moe_inter*c->n_shared;
    int *idxs=malloc((size_t)S*K*sizeof(int)); float *ws=malloc((size_t)S*K*sizeof(float));
    int *keff=malloc(S*sizeof(int));
    for(int s=0;s<S;s++){
        const float *xs=x+(int64_t)s*D;
        matmul(logit, xs, l->router, 1, D, E);
        for(int e=0;e<E;e++){ sig[e]=sigmoidf(logit[e]); choice[e]=sig[e]+l->router_bias[e]; }
        int *idx=idxs+(int64_t)s*K; float *w=ws+(int64_t)s*K;
        int Ksel = g_topk>0 ? (g_topk<K?g_topk:K) : K;
        for(int kk=0;kk<Ksel;kk++){ int best=-1; float bv=-1e30f;
            for(int e=0;e<E;e++){ int tk=0; for(int j=0;j<kk;j++) if(idx[j]==e){tk=1;break;}
                if(!tk && choice[e]>bv){bv=choice[e];best=e;} }
            idx[kk]=best; w[kk]=sig[best];
        }
        int Ke=Ksel;
        if(g_topp>0 && g_topp<1.f){
            for(int a=1;a<Ksel;a++){ int ii=idx[a]; float ww=w[a]; int b=a-1;
                while(b>=0 && w[b]<ww){ w[b+1]=w[b]; idx[b+1]=idx[b]; b--; } w[b+1]=ww; idx[b+1]=ii; }
            float tot=1e-20f; for(int kk=0;kk<Ksel;kk++) tot+=w[kk];
            float cum=0; for(int kk=0;kk<Ksel;kk++){ cum+=w[kk]; if(cum>=g_topp*tot){ Ke=kk+1; break; } }
        }
        keff[s]=Ke; m->ereq+=Ke;
        for(int kk=0;kk<Ke;kk++){
            m->eusage[layer][idx[kk]]++;
            if(m->eheat[layer][idx[kk]]<UINT32_MAX) m->eheat[layer][idx[kk]]++;
        }
        if(c->norm_topk){ float sm=0; for(int kk=0;kk<Ke;kk++) sm+=w[kk]; sm+=1e-20f; for(int kk=0;kk<Ke;kk++) w[kk]/=sm; }
        for(int kk=0;kk<Ke;kk++) w[kk]*=c->routed_scale;
        for(int d=0;d<D;d++) out[(int64_t)s*D+d]=0;
    }
    if(g_looka && S==1 && layer<c->n_layers){
        int Ke=keff[0];
        if(m->enr[layer]>0){
            for(int kk=0;kk<Ke;kk++) for(int z=0;z<m->enr[layer];z++)
                if(m->eroute[layer][z]==idxs[kk]){ la_hit[0]++; break; }
            la_tot[0]+=Ke;
        }
        for(int kind=0;kind<2;kind++) if(la_val[kind][layer]){
            for(int kk=0;kk<Ke;kk++) for(int z=0;z<K;z++)
                if(la_pred[kind][layer][z]==idxs[kk]){ la_hit[1+kind]++; break; }
            la_tot[1+kind]+=Ke; la_val[kind][layer]=0;
        }
    }
    m->enr[layer]=keff[S-1]; for(int kk=0;kk<keff[S-1];kk++) m->eroute[layer][kk]=idxs[(int64_t)(S-1)*K+kk];
    /* union degli expert del batch */
    int *uniq=malloc((size_t)E*sizeof(int)); int nu=0;
    { char *seen=calloc(E,1);
      for(int s=0;s<S;s++) for(int kk=0;kk<keff[s];kk++){ int e=idxs[(int64_t)s*K+kk];
          if(!seen[e]){ seen[e]=1; uniq[nu++]=e; } }
      free(seen); }
    float *xg=falloc((int64_t)S*D), *gg=falloc((int64_t)S*I), *uu=falloc((int64_t)S*I), *hh=falloc((int64_t)S*D);
    int *rows=malloc(S*sizeof(int)); float *rw=malloc(S*sizeof(float));
    for(int base=0;base<nu;base+=64){
        int nb = nu-base<64 ? nu-base : 64;
        ESlot *use[64]; int missk[64]; int nmiss=0;
        for(int j=0;j<nb;j++){ int eid=uniq[base+j]; use[j]=NULL;
            ESlot *P=m->pin[layer];
            for(int z=0;z<m->npin[layer];z++) if(P[z].eid==eid){ m->hits++; use[j]=&P[z]; break; }
            if(!use[j]){ ESlot *Sl=m->ecache[layer]; int nn=m->ecn[layer];
                for(int z=0;z<nn;z++) if(Sl[z].eid==eid){ m->hits++; Sl[z].used=++m->eclock; use[j]=&Sl[z]; break; } }
            if(!use[j]){ use[j]=&m->ws[nmiss]; missk[nmiss++]=j; m->miss++; }
        }
        if(nmiss){ double t0=now_s();
            #pragma omp parallel for schedule(dynamic,1)
            for(int q=0;q<nmiss;q++) expert_load(m,layer,uniq[base+missk[q]],&m->ws[q]);
            m->t_edisk += now_s()-t0; }
        if(base+64<nu){
            int nb2 = nu-(base+64)<64 ? nu-(base+64) : 64;
            for(int j=0;j<nb2;j++){ int eid=uniq[base+64+j]; int found=0;
                ESlot *P=m->pin[layer];
                for(int z=0;z<m->npin[layer] && !found;z++) if(P[z].eid==eid) found=1;
                ESlot *Sl=m->ecache[layer];
                for(int z=0;z<m->ecn[layer] && !found;z++) if(Sl[z].eid==eid) found=1;
                if(!found) expert_prefetch(m,layer,eid);
            }
        }
        for(int j=0;j<nb;j++){ int eid=uniq[base+j]; ESlot *e=use[j];
            int nr=0;
            for(int s=0;s<S;s++) for(int kk=0;kk<keff[s];kk++)
                if(idxs[(int64_t)s*K+kk]==eid){ rows[nr]=s; rw[nr]=ws[(int64_t)s*K+kk]; nr++; break; }
            if(!nr) continue;
#ifdef COLI_CUDA
            if(g_cuda_enabled && e->g.cuda_eligible) m->gpu_expert_calls++;
#endif
#ifdef COLI_METAL
            if(g_metal_enabled && e->g.metal_eligible) m->gpu_expert_calls++;
#endif
            for(int r=0;r<nr;r++) memcpy(xg+(int64_t)r*D, x+(int64_t)rows[r]*D, D*sizeof(float));
            double t0=now_s();
            matmul_qt(gg, xg, &e->g, nr);
            matmul_qt(uu, xg, &e->u, nr);
            for(int64_t z=0;z<(int64_t)nr*I;z++) gg[z]=siluf(gg[z])*uu[z];
            matmul_qt(hh, gg, &e->d, nr);
            for(int r=0;r<nr;r++){ float *os=out+(int64_t)rows[r]*D, wgt=rw[r], *hr=hh+(int64_t)r*D;
                for(int d=0;d<D;d++) os[d]+=wgt*hr[d]; }
            m->t_emm += now_s()-t0;
        }
        { ESlot *Sl=m->ecache[layer]; int *nn=&m->ecn[layer];
          int promo = nmiss<m->ecap ? nmiss : m->ecap;
          for(int a=0;a<promo;a++){ int q=nmiss-1-a; ESlot *dst;
              if(*nn<m->ecap) dst=&Sl[(*nn)++];
              else { int lru=0; for(int z=1;z<*nn;z++) if(Sl[z].used<Sl[lru].used) lru=z; dst=&Sl[lru]; }
              ESlot tmp=*dst; *dst=m->ws[q]; m->ws[q]=tmp; dst->used=++m->eclock; }
        }
    }
    /* shared expert */
    float *sg=falloc((int64_t)S*sI), *su=falloc((int64_t)S*sI);
    matmul_qt(sg, x, &l->sh_gate, S);
    matmul_qt(su, x, &l->sh_up,   S);
    for(int64_t z=0;z<(int64_t)S*sI;z++) sg[z]=siluf(sg[z])*su[z];
    matmul_qt(hh, sg, &l->sh_down, S);
    for(int64_t z=0;z<(int64_t)S*D;z++) out[z]+=hh[z];
    free(logit); free(sig); free(choice); free(idxs); free(ws); free(keff); free(uniq);
    free(xg); free(gg); free(uu); free(hh); free(rows); free(rw); free(sg); free(su);
}

static void dense_mlp(Layer *l, float *x, int S, int D, int I, float *out){
    float *g=falloc((int64_t)S*I), *u=falloc((int64_t)S*I);
    matmul_qt(g, x, &l->gate_proj, S);
    matmul_qt(u, x, &l->up_proj,   S);
    for(int64_t i=0;i<(int64_t)S*I;i++) g[i]=siluf(g[i])*u[i];
    matmul_qt(out, g, &l->down_proj, S);
    free(g); free(u);
}

static void la_predict(Model *m, int target, const float *h, int kind){
    Cfg *c=&m->c; Layer *l=&m->L[target]; int D=c->hidden, E=c->n_experts, K=c->topk;
    float *nrm=falloc(D), *ch=falloc(E);
    rmsnorm(nrm,h,l->post_ln,D,c->eps);
    matmul(ch,nrm,l->router,1,D,E);
    for(int e=0;e<E;e++) ch[e]=sigmoidf(ch[e])+l->router_bias[e];
    int *pred=la_pred[kind][target];
    for(int kk=0;kk<K;kk++){ int best=-1; float bv=-1e30f;
        for(int e=0;e<E;e++){ int tk=0; for(int j=0;j<kk;j++) if(pred[j]==e){tk=1;break;}
            if(!tk && ch[e]>bv){bv=ch[e];best=e;} }
        pred[kk]=best; }
    la_val[kind][target]=1;
    free(nrm); free(ch);
}

/* PILOTA: prefetch guidato dal router */
static struct { int l,e; } pilot_q[4096];
static volatile unsigned pilot_w=0, pilot_r=0;
static Model *pilot_m=NULL;
static void *pilot_worker(void *arg){
    (void)arg;
    for(;;){
        unsigned r=__atomic_load_n(&pilot_r,__ATOMIC_ACQUIRE);
        unsigned w=__atomic_load_n(&pilot_w,__ATOMIC_ACQUIRE);
        if(r==w){ usleep(200); continue; }
        expert_prefetch(pilot_m, pilot_q[r&4095].l, pilot_q[r&4095].e);
        __atomic_store_n(&pilot_r,r+1,__ATOMIC_RELEASE);
    }
    return NULL;
}
static void pilot_prefetch(Model *m, int lnext, const float *x, int S){
    Cfg *c=&m->c; Layer *l=&m->L[lnext]; int D=c->hidden, E=c->n_experts;
    int K = g_pilot_k<c->topk ? g_pilot_k : c->topk;
    if(!pilot_m){ pilot_m=m; pthread_t t; pthread_create(&t,NULL,pilot_worker,NULL); }
    float *nrm=falloc(D), *ch=falloc(E);
    for(int s=0;s<S;s++){
        rmsnorm(nrm, x+(int64_t)s*D, l->post_ln, D, c->eps);
        matmul(ch, nrm, l->router, 1, D, E);
        for(int e=0;e<E;e++) ch[e]=sigmoidf(ch[e])+l->router_bias[e];
        for(int kk=0;kk<K;kk++){
            int best=0; for(int e=1;e<E;e++) if(ch[e]>ch[best]) best=e;
            ch[best]=-2e30f;
            int found=0; ESlot *P=m->pin[lnext];
            for(int z=0;z<m->npin[lnext] && !found;z++) if(P[z].eid==best) found=1;
            ESlot *Sl=m->ecache[lnext];
            for(int z=0;z<m->ecn[lnext] && !found;z++) if(Sl[z].eid==best) found=1;
            if(!found){
                unsigned w=__atomic_load_n(&pilot_w,__ATOMIC_RELAXED);
                if(w-__atomic_load_n(&pilot_r,__ATOMIC_ACQUIRE)<4096){
                    pilot_q[w&4095].l=lnext; pilot_q[w&4095].e=best;
                    __atomic_store_n(&pilot_w,w+1,__ATOMIC_RELEASE);
                }
            }
        }
    }
    free(nrm); free(ch);
}

/* forward di UN layer */
static void layer_forward(Model *m, Layer *l, int li, float *x, int S, int pos_base, float *nrm, float *tmp){
    Cfg *c=&m->c; int D=c->hidden;
    if(g_spec && g_prefetch && l->sparse && m->enr[li]>0)
        for(int z=0;z<m->enr[li];z++) expert_prefetch(m,li,m->eroute[li][z]);
    if(g_looka && S==1 && li<c->n_layers && l->sparse) la_predict(m,li,x,0);
    for(int s=0;s<S;s++) rmsnorm(nrm+(int64_t)s*D, x+(int64_t)s*D, l->in_ln, D, c->eps);
    attention(m,l,li,nrm,S,pos_base,tmp);
    for(int64_t j=0;j<(int64_t)S*D;j++) x[j]+=tmp[j];
    if(g_pilot && S<=8 && li+1<c->n_layers && m->L[li+1].sparse) pilot_prefetch(m,li+1,x,S);
    if(g_looka && S==1 && li+1<c->n_layers && m->L[li+1].sparse) la_predict(m,li+1,x,1);
    for(int s=0;s<S;s++) rmsnorm(nrm+(int64_t)s*D, x+(int64_t)s*D, l->post_ln, D, c->eps);
    if(l->sparse) moe(m,l,li,nrm,S,tmp); else dense_mlp(l,nrm,S,D,c->dense_inter,tmp);
    for(int64_t j=0;j<(int64_t)S*D;j++) x[j]+=tmp[j];
}
static void layers_forward(Model *m, float *x, int S, int pos_base){
    Cfg *c=&m->c; int D=c->hidden;
    float *nrm=falloc((int64_t)S*D), *tmp=falloc((int64_t)S*D);
    for(int i=0;i<c->n_layers;i++){
        if(S>=8 && (i%4==0 || i==c->n_layers-1))
            fprintf(stderr,"[prefill] layer %d/%d · %d token\n", i+1, c->n_layers, S);
        layer_forward(m,&m->L[i],i,x,S,pos_base,nrm,tmp);
    }
    free(nrm); free(tmp);
}

static void kv_alloc(Model *m, int max_t){
    Cfg *c=&m->c;
    if(m->Kc){ for(int i=0;i<c->n_layers;i++){ free(m->Kc[i]); free(m->Vc[i]); } free(m->Kc); free(m->Vc); }
    m->max_t=max_t;
    m->Kc=calloc(c->n_layers,sizeof(float*));
    m->Vc=calloc(c->n_layers,sizeof(float*));
    for(int i=0;i<c->n_layers;i++){
        m->Kc[i]=falloc((int64_t)max_t*c->n_kv_heads*c->head_dim);
        m->Vc[i]=falloc((int64_t)max_t*c->n_kv_heads*c->head_dim);
    }
}

static float *step(Model *m, const int *ids, int S, int pos_base){
    Cfg *c=&m->c; int D=c->hidden;
    float *x=falloc((int64_t)S*D);
    for(int s=0;s<S;s++) embed_row(m, ids[s], x+(int64_t)s*D);
    layers_forward(m,x,S,pos_base);
    float *last=falloc(D); rmsnorm(last, x+(int64_t)(S-1)*D, m->final_norm, D, c->eps);
    double th0=now_s();
    float *logit=falloc(c->vocab); matmul_qt(logit,last,&m->lm_head,1);
    m->t_head += now_s()-th0;
    free(x); free(last); return logit;
}

static float *step_all(Model *m, const int *ids, int S, int pos_base){
    Cfg *c=&m->c; int D=c->hidden;
    float *x=falloc((int64_t)S*D);
    for(int s=0;s<S;s++) embed_row(m, ids[s], x+(int64_t)s*D);
    layers_forward(m,x,S,pos_base);
    float *lo=falloc((int64_t)S*c->vocab), *row=falloc(D);
    for(int s=0;s<S;s++){ rmsnorm(row, x+(int64_t)s*D, m->final_norm, D, c->eps);
        matmul_qt(lo+(int64_t)s*c->vocab, row, &m->lm_head, 1); }
    free(x); free(row); return lo;
}

/* prompt-lookup n-gram draft */
static int ngram_draft(const int *ids, int len, int G, int *draft){
    if(len<4 || G<1) return 0;
    int a=ids[len-2], b=ids[len-1];
    for(int i=len-3;i>=1;i--)
        if(ids[i-1]==a && ids[i]==b){
            int n=0; for(int j=i+1;j<len && n<G;j++) draft[n++]=ids[j];
            return n;
        }
    return 0;
}

static inline int argmax_v(const float *lo, int V){
    int b=0; float bv=lo[0]; for(int i=1;i<V;i++) if(lo[i]>bv){bv=lo[i];b=i;} return b;
}

/* ---- SAMPLING (temperatura + nucleus) con verifica speculativa LOSSLESS ---- */
static uint64_t g_rng=0x9E3779B97F4A7C15ULL;
static inline double rndu(void){ g_rng^=g_rng<<13; g_rng^=g_rng>>7; g_rng^=g_rng<<17;
    return (double)(g_rng>>11)*(1.0/9007199254740992.0); }
static float *g_pbuf=NULL; static int *g_pidx=NULL;
static int cmp_pdesc(const void *a,const void *b){
    float pa=g_pbuf[*(const int*)a], pb=g_pbuf[*(const int*)b];
    return pa<pb ? 1 : pa>pb ? -1 : 0; }
static void dist_build(const float *lo, int V){
    if(!g_pbuf){ g_pbuf=falloc(V); g_pidx=malloc(V*sizeof(int)); }
    float mx=lo[0]; for(int i=1;i<V;i++) if(lo[i]>mx) mx=lo[i];
    double s=0; float invt=1.f/(g_temp>1e-4f?g_temp:1e-4f);
    for(int i=0;i<V;i++){ g_pbuf[i]=expf((lo[i]-mx)*invt); s+=g_pbuf[i]; }
    for(int i=0;i<V;i++) g_pbuf[i]/=(float)s;
    if(g_nuc>0 && g_nuc<1.f){
        for(int i=0;i<V;i++) g_pidx[i]=i;
        qsort(g_pidx,V,sizeof(int),cmp_pdesc);
        double cum=0; int keep=V;
        for(int i=0;i<V;i++){ cum+=g_pbuf[g_pidx[i]]; if(cum>=g_nuc){ keep=i+1; break; } }
        double s2=0; for(int i=keep;i<V;i++) g_pbuf[g_pidx[i]]=0;
        for(int i=0;i<keep;i++) s2+=g_pbuf[g_pidx[i]];
        for(int i=0;i<keep;i++) g_pbuf[g_pidx[i]]/=(float)s2;
    }
}
static int dist_sample(int V, int ban){
    double z = 1.0 - (ban>=0 ? g_pbuf[ban] : 0.0); if(z<=1e-12) z=1e-12;
    double u = rndu()*z, cum=0;
    for(int i=0;i<V;i++){ if(i==ban) continue; cum+=g_pbuf[i]; if(cum>=u) return i; }
    for(int i=V-1;i>=0;i--) if(i!=ban && g_pbuf[i]>0) return i;
    return 0;
}
static int pick_tok(const float *lo, int V, int ban){
    if(g_temp<=0) return argmax_v(lo,V);
    dist_build(lo,V);
    return dist_sample(V,ban);
}

static int g_stop[9], g_nstop=0;
static inline int is_stop(int t){ for(int i=0;i<g_nstop;i++) if(t==g_stop[i]) return 1; return 0; }
static void stops_arm(const Cfg *c, int tok_eos){
    g_nstop=0;
    for(int i=0;i<c->n_stop;i++) g_stop[g_nstop++]=c->stop_ids[i];
    if(tok_eos>=0 && !is_stop(tok_eos)) g_stop[g_nstop++]=tok_eos;
    fprintf(stderr,"[stop] %d token di stop:",g_nstop);
    for(int i=0;i<g_nstop;i++) fprintf(stderr," %d",g_stop[i]);
    fprintf(stderr,"\n");
}

static int spec_decode(Model *m, int *all, int kv, int n_new, int eos, float *logit,
                       void (*emit)(int,void*), void *ud, int *kv_out){
    Cfg *c=&m->c; int V=c->vocab; int emitted=0, done=0;
    int draft[64]; if(g_draft>63) g_draft=63;
    int carry_ban=-1;
    while(emitted<n_new && !done){
        int next=pick_tok(logit,V,carry_ban); carry_ban=-1; free(logit); logit=NULL;
        if((eos>=0 && next==eos) || is_stop(next)) break;
        emit(next,ud); all[kv]=next; emitted++; m->n_emit++;
        if(emitted>=n_new) break;
        int g=0;
        if(g_draft>0) g=ngram_draft(all,kv+1,g_draft,draft);
        if(g>n_new-emitted) g=n_new-emitted;
        if(kv+1+g+1>m->max_t) g=m->max_t-kv-2;
        if(g<0) g=0;
        int S=1+g; int batch[64]; batch[0]=next; memcpy(batch+1,draft,g*sizeof(int));
        float *lo=step_all(m,batch,S,kv); m->n_fw++;
        int k=0;
        while(k<g && emitted<n_new){
            int accept;
            if(g_temp<=0) accept = (argmax_v(lo+(int64_t)k*V,V)==draft[k]);
            else { dist_build(lo+(int64_t)k*V,V);
                   accept = (rndu() < g_pbuf[draft[k]]); }
            if(!accept){ if(g_temp>0) carry_ban=draft[k]; break; }
            if((eos>=0 && draft[k]==eos) || is_stop(draft[k])){ done=1; break; }
            emit(draft[k],ud); all[kv+1+k]=draft[k]; emitted++; m->n_emit++; k++;
        }
        kv += 1+k;
        logit=falloc(V); memcpy(logit, lo+(int64_t)k*V, V*sizeof(float)); free(lo);
    }
    if(logit) free(logit);
    if(kv_out) *kv_out=kv;
    return emitted;
}

typedef struct { int *dst; int n; } EmitStore;
static void emit_store(int t, void *ud){ EmitStore *e=(EmitStore*)ud; e->dst[e->n++]=t; }
typedef struct { Tok *T; Model *m; double t0; int count; int quiet; } EmitStream;
static void emit_stream(int t, void *ud){
    EmitStream *e=(EmitStream*)ud; char dec[64];
    int dn=tok_decode(e->T,&t,1,dec,63); dec[dn]=0; fputs(dec,stdout); fflush(stdout);
    if(!e->quiet && ++e->count%16==0){ double tt=e->m->hits+e->m->miss;
        fprintf(stderr,"\n[t=%d  RSS %.2f GB  hit %.0f%%  %.2f tok/s]\n", e->count,
            rss_gb(), tt?100.0*e->m->hits/tt:0.0, e->count/(now_s()-e->t0)); }
}

/* teacher-forcing */
static void forward_all(Model *m, const int *ids, int S, int *pred){
    Cfg *c=&m->c; int D=c->hidden;
    kv_alloc(m,S);
    float *x=falloc((int64_t)S*D);
    for(int s=0;s<S;s++) embed_row(m, ids[s], x+(int64_t)s*D);
    layers_forward(m,x,S,0);
    float *lo=falloc(c->vocab);
    for(int s=0;s<S;s++){
        float *row=falloc(D); rmsnorm(row, x+(int64_t)s*D, m->final_norm, D, c->eps);
        matmul_qt(lo, row, &m->lm_head, 1);
        int best=0; float bv=lo[0]; for(int i=1;i<c->vocab;i++) if(lo[i]>bv){bv=lo[i];best=i;}
        pred[s]=best;
        free(row);
    }
    free(x); free(lo);
}

static void generate(Model *m, const int *prompt, int np, int n_new, int *out){
    kv_alloc(m,np+n_new+g_draft+2);
    for(int i=0;i<np;i++) out[i]=prompt[i];
    float *logit=step(m,prompt,np,0);
    EmitStore es={out+np,0};
    spec_decode(m,out,np,n_new,-1,logit,emit_store,&es,NULL);
}

static void profile_print(Model *m, double elapsed){
    double accounted=m->t_edisk+m->t_emm+m->t_attn+m->t_head;
    printf("PROFILO: expert-disk %.3fs | expert-matmul %.3fs | attention %.3fs "
           "| lm_head %.3fs | altro %.3fs\n",
        m->t_edisk,m->t_emm,m->t_attn,m->t_head,elapsed-accounted);
}

/* generazione reale: tokenizza PROMPT, prefill + decode greedy con stop su EOS */
static void run_text(Model *m, const char *snap, const char *prompt, int ngen){
    Cfg *c=&m->c; char tkp[2048]; snprintf(tkp,sizeof(tkp),"%s/tokenizer.json",snap);
    Tok T; tok_load(&T,tkp);
    int eos=tok_id_of(&T,"<|endoftext|>");
    stops_arm(&m->c, eos);
    if(g_temp<0) g_temp=0.7f;
    int cap=(int)strlen(prompt)+16; int *pids=malloc(cap*sizeof(int));
    int np=tok_encode(&T,prompt,(int)strlen(prompt),pids,cap);
    if(np<1){ fprintf(stderr,"prompt vuoto dopo tokenizzazione\n"); return; }
    printf("prompt: %d token | genero fino a %d (stop EOS=%d) | draft n-gram=%d\n", np, ngen, eos, g_draft);
    fputs(prompt,stdout); fflush(stdout);
    kv_alloc(m, np+ngen+g_draft+2);
    int *all=malloc((np+ngen+g_draft+2)*sizeof(int)); memcpy(all,pids,np*sizeof(int));
    double t=now_s();
    float *logit=step(m,pids,np,0);
    EmitStream es={&T,m,t,0,0};
    int produced=spec_decode(m,all,np,ngen,eos,logit,emit_stream,&es,NULL);
    double dt=now_s()-t;
    double tot=m->hits+m->miss;
    int nsp=0; for(int i=0;i<c->n_layers;i++) if(m->L[i].sparse) nsp++;
    printf("\n---\n%d token in %.2fs (%.2f tok/s) | hit-rate expert %.1f%% | RSS %.2f GB\n",
        produced, dt, produced/dt, tot?100.0*m->hits/tot:0.0, rss_gb());
    printf("expert caricati/token: %.1f (per-layer %.2f su %d; baseline topk=%d) | TOPK=%d TOPP=%.2f\n",
        produced?(double)m->ereq/produced:0.0, (produced&&nsp)?(double)m->ereq/produced/nsp:0.0, nsp, c->topk, g_topk, g_topp);
#ifdef COLI_CUDA
    if(g_cuda_enabled) cuda_stats_print();
#endif
#ifdef COLI_METAL
    if(g_metal_enabled) metal_stats_print();
#endif
    profile_print(m,dt);
    free(pids); free(all);
    usage_save(m);
}

static int *read_arr(jval*o,const char*k,int*n){ jval*a=json_get(o,k); int*r=malloc(a->len*sizeof(int));
    for(int i=0;i<a->len;i++) r[i]=(int)a->kids[i]->num; *n=a->len; return r; }

static int64_t tbytes(int O,int I,int bits){
    if(bits>=16) return (int64_t)O*I*4;
    if(bits>=5)  return (int64_t)O*I + (int64_t)O*4;
    return (int64_t)O*((I+1)/2) + (int64_t)O*4;
}
static int64_t expert_bytes_probe(Model *m, int ebits){
    Cfg *c=&m->c; int64_t eb=0; char nm[256];
    snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.0.gate_proj.weight",c->first_dense);
    if(st_nbytes(&m->S,nm)>0){
        const char *suf[3]={"gate_proj","up_proj","down_proj"};
        for(int k=0;k<3;k++){
            snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.0.%s.weight",c->first_dense,suf[k]);
            eb+=st_nbytes(&m->S,nm);
            snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.0.%s.weight.qs",c->first_dense,suf[k]);
            int64_t q=st_nbytes(&m->S,nm); if(q>0) eb+=q;
        }
    }
    if(eb<=0) eb = tbytes(c->moe_inter,c->hidden,ebits)*2 + tbytes(c->hidden,c->moe_inter,ebits);
    return eb;
}

/* stats */
static void stats_dump_q(Model *m, const char *path, int quiet){
    char tmp[2100]; snprintf(tmp,sizeof(tmp),"%s.tmp",path);
    FILE *f=fopen(tmp,"w"); if(!f){ if(!quiet) perror(tmp); return; }
    Cfg *c=&m->c; int64_t tot=0, nz=0;
    for(int i=0;i<c->n_layers;i++){ if(!m->eusage[i]) continue;
        for(int e=0;e<c->n_experts;e++) if(m->eusage[i][e]){ fprintf(f,"%d %d %u\n",i,e,m->eusage[i][e]); tot+=m->eusage[i][e]; nz++; } }
    fclose(f); rename(tmp,path);
    if(!quiet) fprintf(stderr,"[STATS] %lld selezioni su %lld expert distinti -> %s\n",(long long)tot,(long long)nz,path);
}
static void stats_dump(Model *m, const char *path){ stats_dump_q(m,path,0); }

static char g_usage_path[2100]="";
static int64_t usage_load(Model *m, const char *path){
    FILE *f=fopen(path,"r"); if(!f) return 0;
    Cfg *c=&m->c; int l,e; uint32_t cnt; int64_t tot=0;
    while(fscanf(f,"%d %d %u",&l,&e,&cnt)==3)
        if(l>=0&&l<c->n_layers&&e>=0&&e<c->n_experts&&m->eusage[l]){ m->eusage[l][e]+=cnt; tot+=cnt; }
    fclose(f); return tot;
}
static void usage_save(Model *m){ if(g_usage_path[0]) stats_dump_q(m,g_usage_path,1); }

/* HOT-STORE */
static int g_mlock=-1;
static int mem_should_wire(void){
    if(g_mlock>=0) return g_mlock;
#if defined(__APPLE__)
    return 1;
#else
    return 0;
#endif
}
static int mem_wire(void *addr, size_t len){
#if defined(__APPLE__) || defined(__linux__)
    return mlock(addr, len);
#else
    (void)addr; (void)len; return 0;
#endif
}
static void pin_wire(Model *m){
    if(!mem_should_wire()) return;
    Cfg *c=&m->c; double t0=now_s(); int64_t wired=0; long failed=0;
    for(int i=0;i<c->n_layers;i++) for(int z=0;z<m->npin[i];z++){
        ESlot *s=&m->pin[i][z];
        if(s->slab){  if(mem_wire(s->slab, s->slab_cap)==0) wired+=s->slab_cap; else failed++; }
        if(s->fslab){ size_t fl=(size_t)s->fslab_cap*sizeof(float);
                      if(mem_wire(s->fslab, fl)==0) wired+=fl; else failed++; }
    }
    if(failed)
        fprintf(stderr,"[PIN] mlock: %.1f GB wired, %ld alloc failed in %.0fs\n", wired/1e9, failed, now_s()-t0);
    else
        fprintf(stderr,"[PIN] mlock: %.1f GB wired in physical RAM in %.0fs\n", wired/1e9, now_s()-t0);
}

static void pin_load(Model *m, const char *statspath, double gb){
    FILE *f=fopen(statspath,"r"); if(!f){ perror(statspath); return; }
    typedef struct { int l,e; uint32_t c; } Rec;
    Cfg *c=&m->c; int cap=c->n_layers*c->n_experts;
    Rec *r=malloc((size_t)cap*sizeof(Rec)); int n=0;
    int l,e; uint32_t cnt;
    while(n<cap && fscanf(f,"%d %d %u",&l,&e,&cnt)==3){
        int ok = l>=0 && e>=0 && e<c->n_experts && l<c->n_layers && m->L[l].sparse;
        if(ok) r[n++]=(Rec){l,e,cnt};
    }
    fclose(f);
    for(int a=0;a<n;a++){ int best=a;
        for(int b=a+1;b<n;b++) if(r[b].c>r[best].c) best=b;
        Rec t=r[a]; r[a]=r[best]; r[best]=t;
        if(a>4095) break;
    }
    int64_t eb=expert_bytes_probe(m,m->ebits);
    int npin=(int)(gb*1e9/eb); if(npin>n) npin=n; if(npin>4096) npin=4096;
    if(npin<1){ free(r); return; }
    int *cnt_l=calloc(c->n_layers,sizeof(int));
    for(int a=0;a<npin;a++) cnt_l[r[a].l]++;
    for(int i=0;i<c->n_layers;i++) if(cnt_l[i]) m->pin[i]=calloc(cnt_l[i],sizeof(ESlot));
    double t0=now_s();
    #pragma omp parallel for schedule(dynamic,1)
    for(int a=0;a<npin;a++){
        int li=r[a].l, slot;
        #pragma omp critical
        slot=m->npin[li]++;
        expert_load(m,li,r[a].e,&m->pin[li][slot]);
    }
    m->resident_bytes += (int64_t)npin*eb;
    fprintf(stderr,"[PIN] hot-store: %d expert in RAM (%.1f GB) in %.0fs da %s\n",
        npin, npin*eb/1e9, now_s()-t0, statspath);
#ifdef COLI_METAL
    if(g_metal_enabled && g_metal_expert_gb>0){
        double budget=g_metal_expert_gb*1e9;
        size_t free_b=0,total_b=0;
        if(coli_metal_mem_info(&free_b,&total_b)){
            double safe=(double)free_b-2e9;
            if(safe<0) safe=0;
            if(budget>safe) budget=safe;
        }
        for(int a=0;a<npin && m->gpu_expert_bytes<budget;a++){
            int li=r[a].l;
            for(int z=0;z<m->npin[li];z++) if(m->pin[li][z].eid==r[a].e){
                ESlot *s=&m->pin[li][z];
                int64_t need=qt_bytes(&s->g)+qt_bytes(&s->u)+qt_bytes(&s->d);
                if(m->gpu_expert_bytes+need>budget) break;
                s->g.metal_eligible=s->u.metal_eligible=s->d.metal_eligible=1;
                if(qt_metal_upload(&s->g) && qt_metal_upload(&s->u) && qt_metal_upload(&s->d)){
                    int64_t actual=(int64_t)coli_metal_tensor_bytes(s->g.metal)
                                  +(int64_t)coli_metal_tensor_bytes(s->u.metal)
                                  +(int64_t)coli_metal_tensor_bytes(s->d.metal);
                    m->gpu_expert_count++; m->gpu_expert_bytes+=actual;
                } else {
                    qt_metal_reset(&s->g); qt_metal_reset(&s->u); qt_metal_reset(&s->d);
                    s->g.metal_eligible=s->u.metal_eligible=s->d.metal_eligible=0;
                }
                break;
            }
        }
        fprintf(stderr,"[Metal] hot expert tier: %d/%d expert, %.2f GB (budget %.1f GB)\n",
            m->gpu_expert_count,npin,m->gpu_expert_bytes/1e9,g_metal_expert_gb);
    }
#endif
    pin_wire(m);
    free(r); free(cnt_l);
}

static double g_mem_avail_boot=0;
static double mem_available_gb(void){
#ifdef __APPLE__
    mach_msg_type_number_t cnt=HOST_VM_INFO64_COUNT;
    vm_statistics64_data_t vm;
    if(host_statistics64(mach_host_self(),HOST_VM_INFO64,(host_info64_t)&vm,&cnt)!=KERN_SUCCESS) return 0;
    return ((double)vm.free_count+(double)vm.inactive_count+(double)vm.purgeable_count)
           * (double)sysconf(_SC_PAGESIZE) / 1e9;
#else
    FILE *f=fopen("/proc/meminfo","r"); if(!f) return 0;
    char ln[256]; double kb=0;
    while(fgets(ln,sizeof(ln),f)) if(sscanf(ln,"MemAvailable: %lf",&kb)==1) break;
    fclose(f); return kb/1e6;
#endif
}

static double kv_pool_bytes(Model *m, int max_ctx){
    Cfg *c=&m->c;
    return (double)c->n_layers*max_ctx*c->n_kv_heads*c->head_dim*2.0*4.0;
}

static double expert_avail(Model *m, double ram_gb, int ebits, int max_ctx){
    int64_t eb=expert_bytes_probe(m,ebits);
    if(ram_gb<=0){ ram_gb=g_mem_avail_boot*0.88; if(ram_gb<4) ram_gb=8; }
    double slack = 1.2e9 + 2.5e9 + 64.0*(double)eb + kv_pool_bytes(m,max_ctx);
    return ram_gb*1e9 - (double)m->resident_bytes - slack;
}

static void cap_for_ram(Model *m, double ram_gb, int ebits, int max_ctx){
    Cfg *c=&m->c; int nsp=0; for(int i=0;i<c->n_layers;i++) if(m->L[i].sparse) nsp++;
    int64_t eb=expert_bytes_probe(m,ebits);
    int auto_b = ram_gb<=0;
    if(auto_b){ ram_gb = g_mem_avail_boot*0.88;
        if(ram_gb<4){ fprintf(stderr,"[RAM] MemAvailable illeggibile/troppo bassa, assumo 8 GB\n"); ram_gb=8; } }
    double ws_b  = 64.0*(double)eb;
    double kv_b  = kv_pool_bytes(m,max_ctx);
    double pc_b  = 2.5e9;
    double slack = 1.2e9 + pc_b + ws_b + kv_b;
    double avail = ram_gb*1e9 - (double)m->resident_bytes - slack;
    int capmax = (avail>0 && nsp>0) ? (int)(avail/((double)nsp*eb)) : 0;
    if(capmax<1) capmax=1;
    if(capmax < m->ecap){
        fprintf(stderr,"[RAM_GB=%.1f%s] residente %.1f GB + slack %.1f GB -> cap abbassato %d->%d\n",
            ram_gb,auto_b?" auto":"",m->resident_bytes/1e9,slack/1e9, m->ecap, capmax);
        m->ecap=capmax;
    } else {
        int raise_on = getenv("CAP_RAISE")?atoi(getenv("CAP_RAISE")):1;
        int newcap = capmax>c->n_experts ? c->n_experts : capmax;
        if(raise_on && newcap>m->ecap){
            for(int i=0;i<c->n_layers;i++) if(m->ecache[i]){
                m->ecache[i]=realloc(m->ecache[i],(size_t)newcap*sizeof(ESlot));
                memset(m->ecache[i]+m->ecap,0,(size_t)(newcap-m->ecap)*sizeof(ESlot));
            }
            fprintf(stderr,"[RAM_GB=%.1f%s] cap ALZATO %d->%d\n",
                ram_gb, auto_b?" auto":"", m->ecap, newcap);
            m->ecap=newcap;
        } else
            fprintf(stderr,"[RAM_GB=%.1f%s] cap=%d ok\n", ram_gb, auto_b?" auto":"", m->ecap);
    }
}

int main(int argc, char **argv){
    if(!getenv("OMP_WAIT_POLICY")) setenv("OMP_WAIT_POLICY","passive",1);
    const char *snap=getenv("SNAP"); if(!snap){fprintf(stderr,"SNAP=<dir>\n");return 1;}
    g_nopack = getenv("NOPACK")?1:0;
    g_drop = getenv("DROP")?1:0;
    g_prefetch = getenv("PREFETCH")?atoi(getenv("PREFETCH")):0;
    g_topk = getenv("TOPK")?atoi(getenv("TOPK")):0;
    g_topp = getenv("TOPP")?atof(getenv("TOPP")):0;
    g_mlock  = getenv("MLOCK")?atoi(getenv("MLOCK")):-1;
    g_spec = getenv("SPEC")?atoi(getenv("SPEC")):1;
    g_draft = getenv("DRAFT")?atoi(getenv("DRAFT")):0;
    g_looka = getenv("LOOKA")?atoi(getenv("LOOKA")):0;
    g_pilot = getenv("PILOT")?atoi(getenv("PILOT")):0;
    g_pilot_k = getenv("PILOT_K")?atoi(getenv("PILOT_K")):8;
    if(g_pilot_k<1) g_pilot_k=1;
    g_direct = getenv("DIRECT")?atoi(getenv("DIRECT")):0;
    g_idot = getenv("IDOT")?atoi(getenv("IDOT")):1;
    g_temp = getenv("TEMP")?atof(getenv("TEMP")):-1;
    g_nuc  = getenv("NUCLEUS")?atof(getenv("NUCLEUS")):0.90f;
    if(getenv("SEED")) g_rng = (uint64_t)atoll(getenv("SEED"))*0x9E3779B97F4A7C15ULL+1;
    else { struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); g_rng ^= (uint64_t)ts.tv_nsec<<20 ^ (uint64_t)getpid(); }
    if(g_draft>63) g_draft=63;
    int cap  = argc>1?atoi(argv[1]):64;
    int ebits= argc>2?atoi(argv[2]):8;
    int dbits= argc>3?atoi(argv[3]):ebits;
#ifdef COLI_CUDA
    if(getenv("COLI_CUDA") && atoi(getenv("COLI_CUDA"))){
        const char *one=getenv("COLI_GPU"), *many=getenv("COLI_GPUS");
        if(one&&many){ fprintf(stderr,"usa COLI_GPU oppure COLI_GPUS, non entrambi\n"); return 2; }
        if(many) g_cuda_ndev=parse_cuda_devices(many,g_cuda_devices);
        else if(one) g_cuda_ndev=parse_cuda_devices(one,g_cuda_devices);
        else { g_cuda_ndev=1; g_cuda_devices[0]=0; }
        if(g_cuda_ndev<1){ fprintf(stderr,"COLI_GPUS non valido\n"); return 2; }
        g_cuda_enabled=coli_cuda_init(g_cuda_devices,g_cuda_ndev);
        if(!g_cuda_enabled){ fprintf(stderr,"[CUDA] backend richiesto ma non disponibile\n"); return 2; }
    }
    g_cuda_dense=getenv("CUDA_DENSE")?atoi(getenv("CUDA_DENSE")):0;
    g_cuda_expert_gb=getenv("CUDA_EXPERT_GB")?atof(getenv("CUDA_EXPERT_GB")):0;
#else
    if((getenv("COLI_CUDA") && atoi(getenv("COLI_CUDA"))) ||
       getenv("COLI_GPU") || getenv("COLI_GPUS") ||
       (getenv("CUDA_DENSE") && atoi(getenv("CUDA_DENSE"))) ||
       (getenv("CUDA_EXPERT_GB") && atof(getenv("CUDA_EXPERT_GB"))>0)){
        fprintf(stderr,"CUDA richiesto ma questo binario e' CPU-only; ricompila con: make CUDA=1\n");
        return 2;
    }
#endif
#ifdef COLI_METAL
    if(getenv("COLI_METAL") && atoi(getenv("COLI_METAL"))){
        g_metal_enabled=coli_metal_init();
        if(!g_metal_enabled){ fprintf(stderr,"[Metal] backend richiesto ma non disponibile\n"); return 2; }
    }
    g_metal_dense=getenv("METAL_DENSE")?atoi(getenv("METAL_DENSE")):1;
    g_metal_expert_gb=getenv("METAL_EXPERT_GB")?atof(getenv("METAL_EXPERT_GB")):0;
    if(g_metal_dense&&!g_metal_enabled&&getenv("METAL_DENSE")){ fprintf(stderr,"METAL_DENSE richiede COLI_METAL=1\n"); return 2; }
    if(g_metal_expert_gb>0&&!g_metal_enabled){ fprintf(stderr,"METAL_EXPERT_GB richiede COLI_METAL=1\n"); return 2; }
    if(g_metal_enabled) fprintf(stderr,"[Metal] mode: %s\n",g_metal_dense?"resident dense + routed experts":"routed experts only");
#else
    if((getenv("COLI_METAL") && atoi(getenv("COLI_METAL"))) ||
       (getenv("METAL_DENSE") && atoi(getenv("METAL_DENSE"))) ||
       (getenv("METAL_EXPERT_GB") && atof(getenv("METAL_EXPERT_GB"))>0)){
        fprintf(stderr,"Metal richiesto ma questo binario e' CPU-only; ricompila con: make METAL=1\n");
        return 2;
    }
#endif
    printf("== Motore C GLM4 (Glm4MoeForCausalLM), cache=%d expert/layer | expert@%d-bit densa@%d-bit | idot: " IDOT_KERNEL " ==\n", cap, ebits, dbits);
    g_mem_avail_boot = mem_available_gb();
    Model m; double t0=now_s(); model_init(&m,snap,cap,ebits,dbits);
    printf("caricato in %.2fs | densa residente: %.2f MB | layers=%d experts=%d | heads=%d kv_heads=%d head_dim=%d\n",
           now_s()-t0, m.resident_bytes/(1024.0*1024.0), m.c.n_layers, m.c.n_experts,
           m.c.n_heads, m.c.n_kv_heads, m.c.head_dim);
    /* PIN */
    if(getenv("PIN")) pin_load(&m, getenv("PIN"), getenv("PIN_GB")?atof(getenv("PIN_GB")):10.0);
    /* CACHE CHE IMPARA */
    { double ram_env = getenv("RAM_GB")?atof(getenv("RAM_GB")):0.0;
      int est_ctx = getenv("CTX")?atoi(getenv("CTX")):4096;
      snprintf(g_usage_path,sizeof(g_usage_path),"%s/.coli_usage",snap);
      int64_t hist = usage_load(&m,g_usage_path);
      if(hist>0) fprintf(stderr,"[USAGE] storia expert: %lld selezioni (%s)\n",(long long)hist,g_usage_path);
      int autopin = getenv("AUTOPIN")?atoi(getenv("AUTOPIN")):1;
      if(!getenv("PIN") && autopin && hist>=5000){
          double conf = (double)hist/200000.0; if(conf>1) conf=1;
          double pin_gb = expert_avail(&m,ram_env,ebits,est_ctx)*0.5*conf/1e9;
          if(pin_gb>=0.5) pin_load(&m, g_usage_path, pin_gb);
      }
      cap_for_ram(&m, ram_env, ebits, est_ctx); }
    const char *stats=getenv("STATS");

    /* modo testo reale */
    if(getenv("PROMPT")){
        int ngen=getenv("NGEN")?atoi(getenv("NGEN")):64;
        run_text(&m, snap, getenv("PROMPT"), ngen);
        if(stats) stats_dump(&m,stats);
        return 0;
    }

    /* altrimenti: validazione contro l'oracolo (ref_glm.json) */
    const char *refpath=getenv("REF")?getenv("REF"):"ref_glm.json";
    FILE *f=fopen(refpath,"rb"); if(!f){perror(refpath);return 1;}
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    char *b=malloc(n+1); if(fread(b,1,n,f)!=(size_t)n){} b[n]=0; fclose(f);
    char *ar=NULL; jval *ref=json_parse(b,&ar);
    int np,nfull; int *prompt=read_arr(ref,"prompt_ids",&np); int *full=read_arr(ref,"full_ids",&nfull);
    int n_new=nfull-np;

    if(getenv("TF")){
        int *tf=read_arr(ref,"tf_pred",&(int){0});
        int *pred=malloc(nfull*sizeof(int)); double tt=now_s();
        forward_all(&m, full, nfull, pred); double tdt=now_s()-tt;
        int ok=0; for(int i=0;i<nfull;i++) ok+=(pred[i]==tf[i]);
        printf("PREFILL (teacher-forcing) C vs oracolo: %d/%d posizioni | %.1f pos/s\n",
            ok,nfull,nfull/tdt);
        profile_print(&m,tdt);
#ifdef COLI_CUDA
        if(g_cuda_enabled) cuda_stats_print();
#endif
#ifdef COLI_METAL
        if(g_metal_enabled) metal_stats_print();
#endif
        return 0;
    }
    int *out=malloc((np+n_new)*sizeof(int));
    double t=now_s(); generate(&m,prompt,np,n_new,out); double dt=now_s()-t;
    int match=0;
    printf("\nRiferimento (oracolo): "); for(int i=np;i<nfull;i++) printf("%d ", full[i]);
    printf("\nMotore C GLM4        : "); for(int i=np;i<nfull;i++){ printf("%d ", out[i]); if(out[i]==full[i])match++; }
    printf("\nToken coincidenti: %d/%d\n", match, n_new);
    double tot=m.hits+m.miss;
    printf("Hit-rate cache expert: %.1f%% (hit=%llu miss=%llu) | RSS: %.2f GB | %.1f tok/s\n",
           tot?100.0*m.hits/tot:0.0, (unsigned long long)m.hits, (unsigned long long)m.miss, rss_gb(), n_new/dt);
    profile_print(&m,dt);
#ifdef COLI_CUDA
    if(g_cuda_enabled) cuda_stats_print();
#endif
#ifdef COLI_METAL
    if(g_metal_enabled) metal_stats_print();
#endif
    if(stats) stats_dump(&m,stats);
    return 0;
}
