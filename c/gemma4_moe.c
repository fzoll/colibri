/* Motore Gemma 4 MoE (architettura Gemma4ForConditionalGeneration) in C puro.
 * Text-only inference — vision tower non implementata.
 *
 * Target: Gemma 4 26B-A4B (google/gemma-4-26B-A4B-it)
 *   - 128 expert, top-8, ~4B active/token
 *   - 30 layers: 5x sliding_attention + 1x full_attention pattern
 *   - Hybrid attention: sliding window (1024) + global full attention
 *   - K=V (attention_k_eq_v): K and V share weights, only k_proj used
 *   - QK-Norm on Q and K
 *   - GQA: 16 heads, 8 kv_heads (sliding) / 2 kv_heads (global)
 *   - Different head_dim: 256 (sliding) / 512 (global)
 *   - partial_rotary_factor: 1.0 (sliding) / 0.25 (global)
 *   - gelu_pytorch_tanh activation (not SiLU)
 *   - Fused expert weights: experts.gate_up_proj / experts.down_proj (one tensor per layer)
 *   - Custom router: router.proj.weight + router.scale + router.per_expert_scale
 *   - tie_word_embeddings: embed = lm_head
 *   - final_logit_softcapping: 30.0
 *
 * Build: make gemma4   Run: SNAP=./gemma4_weights ./gemma4 <cap> <expert_bits> <dense_bits>
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <limits.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/resource.h>
#if defined(__APPLE__) || defined(__linux__)
#include <sys/mman.h>
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
static inline float hsum256(__m256 v){
    __m128 lo=_mm256_castps256_ps128(v), hi=_mm256_extractf128_ps(v,1);
    lo=_mm_add_ps(lo,hi); __m128 sh=_mm_movehl_ps(lo,lo); lo=_mm_add_ps(lo,sh);
    sh=_mm_shuffle_ps(lo,lo,1); lo=_mm_add_ss(lo,sh); return _mm_cvtss_f32(lo);
}
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#ifdef __APPLE__
#include <mach/mach.h>
#endif

/* ---- Layer type ---- */
#define LTYPE_SLIDING 0
#define LTYPE_GLOBAL  1

/* ---- Config ---- */
typedef struct {
    int hidden, n_layers, n_heads;
    /* sliding attention params */
    int slide_kv_heads, slide_head_dim;
    float slide_rope_theta, slide_partial_rotary;
    int sliding_window;
    /* global attention params */
    int global_kv_heads, global_head_dim;
    float global_rope_theta, global_partial_rotary;
    /* MoE */
    int n_experts, topk, moe_inter;
    /* dense MLP (layer 0) */
    int dense_inter, first_dense; /* first_k_dense_replace=1: layer 0 is dense */
    int vocab;
    int stop_ids[8], n_stop;
    float eps, logit_cap;
    int *layer_type; /* [n_layers] -> LTYPE_SLIDING or LTYPE_GLOBAL */
} Cfg;

/* ---- Quantized tensor ---- */
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

/* ---- Layer ---- */
typedef struct {
    float *in_ln, *post_ln;
    /* attention */
    QT q_proj, k_proj, o_proj;   /* no separate v_proj: K=V */
    float *q_norm, *k_norm;      /* per-head QK-norm weights */
    float layer_scalar;          /* per-layer learned residual scaling */
    /* dense mlp (layer 0 only) */
    int sparse;
    QT dense_gate, dense_up, dense_down;
    float *pre_ff_ln, *post_ff_ln;
    /* MoE (layers 1-29) */
    float *router_proj;          /* [n_experts, hidden] */
    float router_scale;          /* scalar */
    float *per_expert_scale;     /* [n_experts] */
    /* fused expert tensor names (for pread slicing) */
    char gate_up_name[256];      /* model.language_model.layers.N.experts.gate_up_proj */
    char down_name[256];         /* model.language_model.layers.N.experts.down_proj */
    /* extra layernorms for MoE layers */
    float *pre_ff_ln_2;          /* pre_feedforward_layernorm_2 (if present) */
    float *post_ff_ln_1;         /* post_feedforward_layernorm_1 (shared expert) */
    float *post_ff_ln_2;         /* post_feedforward_layernorm_2 */
} Layer;

/* ---- Expert slot ---- */
typedef struct { int eid; QT g,u,d; uint8_t *slab; float *fslab;
                 int64_t slab_cap, fslab_cap; uint64_t used; } ESlot;

/* ---- Model ---- */
typedef struct {
    Cfg c; shards S;
    int ebits, dbits;
    QT embed; float *final_norm;
    /* lm_head is tied to embed — pointer alias, no separate weight */
    Layer *L;
    /* KV cache: per-layer, but sizes differ for sliding vs global layers */
    float **Kc, **Vc;
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

static void usage_save(Model *m);
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
    return r.ru_maxrss/(1024.0*1024.0*1024.0);
#else
    return r.ru_maxrss/(1024.0*1024.0);
#endif
}
static float *falloc(int64_t n){
    if(n<0 || (uint64_t)n > SIZE_MAX/sizeof(float)){ fprintf(stderr,"falloc: n=%lld fuori range\n",(long long)n); exit(1); }
    float *p=malloc((size_t)n*sizeof(float)); if(!p){fprintf(stderr,"OOM\n");exit(1);} return p; }

/* ---- matmul kernels (identical to glm4_moe.c) ---- */
static void matmul(float *y, const float *x, const float *W, int S, int I, int O){
    #pragma omp parallel for schedule(static)
    for (int o=0;o<O;o++){ const float *w=W+(int64_t)o*I;
        for (int s=0;s<S;s++){ const float *xs=x+(int64_t)s*I; float a=0; for(int i=0;i<I;i++) a+=xs[i]*w[i]; y[(int64_t)s*O+o]=a; } }
}
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

/* ---- IDOT kernels ---- */
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

/* ---- quantization ---- */
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

/* ---- normalization and activations ---- */
static void rmsnorm(float *out, const float *x, const float *w, int D, float eps){
    double ms=0; for(int i=0;i<D;i++) ms+=(double)x[i]*x[i];
    float r=1.f/sqrtf((float)(ms/D)+eps); for(int i=0;i<D;i++) out[i]=x[i]*r*w[i];
}
/* RMSNorm without weight (unit norm) — used for QK-norm when norm weights are per-head */
static void rmsnorm_unit(float *out, const float *x, int D, float eps){
    double ms=0; for(int i=0;i<D;i++) ms+=(double)x[i]*x[i];
    float r=1.f/sqrtf((float)(ms/D)+eps); for(int i=0;i<D;i++) out[i]=x[i]*r;
}
static void softmax(float *x,int n){ float m=-1e30f; for(int i=0;i<n;i++) if(x[i]>m)m=x[i];
    float s=0; for(int i=0;i<n;i++){x[i]=expf(x[i]-m);s+=x[i];} for(int i=0;i<n;i++) x[i]/=s; }

/* GELU with tanh approximation (gelu_pytorch_tanh) */
static inline float gelu_tanh(float x){
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

/* RoPE: standard paired, on first rope_dim dimensions */
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

/* ---- config ---- */
static jval* cfg_root(const char *snap, char **arena){
    char p[2048]; snprintf(p,sizeof(p),"%s/config.json",snap);
    FILE *f=fopen(p,"rb"); if(!f){perror(p);exit(1);}
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    char *b=malloc(n+1); if(fread(b,1,n,f)!=(size_t)n){} b[n]=0; fclose(f);
    return json_parse(b,arena);
}
static int gi(jval*r,const char*k){ jval*v=json_get(r,k); return v?(int)v->num:0; }
static double gf(jval*r,const char*k){ jval*v=json_get(r,k); return v?v->num:0.0; }

static void load_cfg(Cfg *c, const char *snap){
    char *ar=NULL; jval *r=cfg_root(snap,&ar);
    /* Gemma 4 wraps text config under "text_config" if it's a VLM config */
    jval *tc=json_get(r,"text_config");
    jval *cfg = tc ? tc : r;

    c->hidden=gi(cfg,"hidden_size"); c->n_layers=gi(cfg,"num_hidden_layers");
    c->n_heads=gi(cfg,"num_attention_heads");
    /* sliding attention */
    c->slide_kv_heads=gi(cfg,"num_key_value_heads"); if(!c->slide_kv_heads) c->slide_kv_heads=8;
    c->slide_head_dim=gi(cfg,"head_dim"); if(!c->slide_head_dim) c->slide_head_dim=256;
    c->sliding_window=gi(cfg,"sliding_window"); if(!c->sliding_window) c->sliding_window=1024;
    c->slide_rope_theta=10000.f;
    { jval *v=json_get(cfg,"rope_theta"); if(v) c->slide_rope_theta=(float)v->num; }
    c->slide_partial_rotary=1.0f;
    { jval *v=json_get(cfg,"partial_rotary_factor"); if(v) c->slide_partial_rotary=(float)v->num; }
    /* global attention */
    c->global_kv_heads=gi(cfg,"num_global_key_value_heads"); if(!c->global_kv_heads) c->global_kv_heads=2;
    c->global_head_dim=gi(cfg,"global_head_dim"); if(!c->global_head_dim) c->global_head_dim=512;
    c->global_rope_theta=1000000.f;
    { jval *v=json_get(cfg,"global_rope_theta"); if(v) c->global_rope_theta=(float)v->num; }
    c->global_partial_rotary=0.25f;
    { jval *v=json_get(cfg,"global_partial_rotary_factor"); if(v) c->global_partial_rotary=(float)v->num; }
    /* MoE */
    c->n_experts=gi(cfg,"num_experts"); if(!c->n_experts) c->n_experts=128;
    c->topk=gi(cfg,"top_k_experts"); if(!c->topk) c->topk=gi(cfg,"num_experts_per_tok"); if(!c->topk) c->topk=8;
    c->moe_inter=gi(cfg,"moe_intermediate_size"); if(!c->moe_inter) c->moe_inter=704;
    c->dense_inter=gi(cfg,"intermediate_size"); if(!c->dense_inter) c->dense_inter=2112;
    c->first_dense=gi(cfg,"first_k_dense_replace"); if(!c->first_dense) c->first_dense=1;
    c->vocab=gi(cfg,"vocab_size"); if(!c->vocab) c->vocab=262144;
    /* epsilon */
    { jval *v=json_get(cfg,"rms_norm_eps"); c->eps=v?(float)v->num:1e-6f; }
    /* logit softcapping */
    { jval *v=json_get(cfg,"final_logit_softcapping"); c->logit_cap=v?(float)v->num:30.0f; }
    /* stop tokens */
    c->n_stop=0;
    jval *eo=json_get(cfg,"eos_token_id");
    if(!eo) eo=json_get(r,"eos_token_id");
    if(eo){ if(eo->t==J_NUM) c->stop_ids[c->n_stop++]=(int)eo->num;
            else if(eo->t==J_ARR) for(int i=0;i<eo->len && c->n_stop<8;i++)
                c->stop_ids[c->n_stop++]=(int)eo->kids[i]->num; }
    /* build layer_type array: [sliding]*5 + [global] repeated 5 times = 30 layers */
    c->layer_type=calloc(c->n_layers,sizeof(int));
    for(int i=0;i<c->n_layers;i++)
        c->layer_type[i] = ((i%6)==5) ? LTYPE_GLOBAL : LTYPE_SLIDING;
    /* try to read from config if present */
    jval *lt=json_get(cfg,"layer_types");
    if(lt && lt->t==J_ARR && lt->len==c->n_layers){
        for(int i=0;i<lt->len;i++){
            if(lt->kids[i]->t==J_STR){
                if(strstr(lt->kids[i]->str,"full") || strstr(lt->kids[i]->str,"global"))
                    c->layer_type[i]=LTYPE_GLOBAL;
                else
                    c->layer_type[i]=LTYPE_SLIDING;
            }
        }
    }
    /* validation */
    #define CKR(name,v,lo,hi) if((v)<(lo)||(v)>(hi)){ \
        fprintf(stderr,"config: %s=%d fuori range [%d,%d]\n",name,(int)(v),(int)(lo),(int)(hi)); exit(1); }
    CKR("hidden_size",c->hidden,1,1<<20)         CKR("num_hidden_layers",c->n_layers,1,128)
    CKR("num_attention_heads",c->n_heads,1,1024)
    CKR("num_experts",c->n_experts,1,4096)
    CKR("top_k_experts",c->topk,1,64)             CKR("moe_intermediate_size",c->moe_inter,1,1<<20)
    CKR("intermediate_size",c->dense_inter,1,1<<24)
    CKR("vocab_size",c->vocab,1,1<<24)
    #undef CKR
    free(ar);
}

/* ---- disk I/O helpers ---- */
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
static float *ld(Model *m, const char *name){
    int64_t n=st_numel(&m->S,name); if(n<0){fprintf(stderr,"manca %s\n",name);exit(1);}
    float *p=falloc(n); st_read_f32(&m->S,name,p,0); return p;
}
/* load optional — returns NULL if not present */
static float *ld_opt(Model *m, const char *name){
    if(!st_has(&m->S,name)) return NULL;
    int64_t n=st_numel(&m->S,name); if(n<0) return NULL;
    float *p=falloc(n); st_read_f32(&m->S,name,p,0); return p;
}

/* ---- model_init: load all Gemma 4 weights ---- */
static void model_init(Model *m, const char *snap, int cap, int ebits, int dbits){
    memset(m,0,sizeof(*m)); m->ebits=ebits; m->dbits=dbits;
    load_cfg(&m->c,snap); st_init(&m->S,snap);
    Cfg *c=&m->c; char nm[300];
    int H=c->n_heads, D=c->hidden;
    int io_bits = dbits>=8 ? 16 : dbits;

    /* embed_tokens — also serves as lm_head (tie_word_embeddings) */
    m->embed = qt_load(m,"model.language_model.embed_tokens.weight", c->vocab, D, io_bits);
    m->final_norm = ld(m,"model.language_model.norm.weight");

    m->L=calloc(c->n_layers,sizeof(Layer));
    m->ecap=cap; m->ecache=calloc(c->n_layers,sizeof(ESlot*)); m->ecn=calloc(c->n_layers,sizeof(int));
    m->eroute=calloc(c->n_layers,sizeof(int*)); m->enr=calloc(c->n_layers,sizeof(int));
    m->pin=calloc(c->n_layers,sizeof(ESlot*)); m->npin=calloc(c->n_layers,sizeof(int));
    m->eusage=calloc(c->n_layers,sizeof(uint32_t*)); m->eheat=calloc(c->n_layers,sizeof(uint32_t*));

    for(int i=0;i<c->n_layers;i++){
        Layer *l=&m->L[i];
        int ltype = c->layer_type[i];
        int hd = ltype==LTYPE_GLOBAL ? c->global_head_dim : c->slide_head_dim;
        int Hkv = ltype==LTYPE_GLOBAL ? c->global_kv_heads : c->slide_kv_heads;

        #define P(s) (snprintf(nm,sizeof(nm),"model.language_model.layers.%d." s,i),nm)
        l->in_ln=ld(m,P("input_layernorm.weight"));
        l->post_ln=ld(m,P("post_attention_layernorm.weight"));

        /* attention: Q, K (=V), O projections. No bias. */
        l->q_proj = qt_load(m,P("self_attn.q_proj.weight"), H*hd, D, dbits);
        l->k_proj = qt_load(m,P("self_attn.k_proj.weight"), Hkv*hd, D, dbits);
        /* v_proj not loaded — K=V: v_proj uses same weight as k_proj */
        l->o_proj = qt_load(m,P("self_attn.o_proj.weight"), D, H*hd, dbits);

        /* QK-norm weights (per-head, shape [head_dim]) */
        l->q_norm = ld(m,P("self_attn.q_norm.weight"));
        l->k_norm = ld(m,P("self_attn.k_norm.weight"));

        /* layer scalar */
        { float *ls=ld(m,P("layer_scalar")); l->layer_scalar=ls[0]; free(ls); }

        /* FFN: layer 0 is dense, layers 1-29 are MoE */
        l->sparse = (i >= c->first_dense);
        if(!l->sparse){
            /* dense MLP */
            l->dense_gate = qt_load(m,P("mlp.gate_proj.weight"), c->dense_inter, D, dbits);
            l->dense_up   = qt_load(m,P("mlp.up_proj.weight"),   c->dense_inter, D, dbits);
            l->dense_down = qt_load(m,P("mlp.down_proj.weight"), D, c->dense_inter, dbits);
            l->pre_ff_ln  = ld(m,P("pre_feedforward_layernorm.weight"));
            l->post_ff_ln = ld(m,P("post_feedforward_layernorm.weight"));
        } else {
            /* MoE: router */
            l->router_proj = ld(m,P("router.proj.weight")); /* [n_experts, hidden] */
            { float *rs=ld(m,P("router.scale")); l->router_scale=rs[0]; free(rs); }
            l->per_expert_scale = ld(m,P("router.per_expert_scale")); /* [n_experts] */
            /* fused expert weight tensor names for pread slicing */
            snprintf(l->gate_up_name,sizeof(l->gate_up_name),
                     "model.language_model.layers.%d.experts.gate_up_proj",i);
            snprintf(l->down_name,sizeof(l->down_name),
                     "model.language_model.layers.%d.experts.down_proj",i);
            /* extra layernorms */
            l->pre_ff_ln   = ld(m,P("pre_feedforward_layernorm.weight"));
            l->post_ff_ln  = ld(m,P("post_feedforward_layernorm.weight"));
            l->pre_ff_ln_2   = ld_opt(m,P("pre_feedforward_layernorm_2.weight"));
            l->post_ff_ln_1  = ld_opt(m,P("post_feedforward_layernorm_1.weight"));
            l->post_ff_ln_2  = ld_opt(m,P("post_feedforward_layernorm_2.weight"));
            /* expert cache */
            m->ecache[i]=calloc(cap,sizeof(ESlot));
            m->eroute[i]=calloc(c->topk,sizeof(int));
            m->eusage[i]=calloc(c->n_experts,sizeof(uint32_t));
            m->eheat[i]=calloc(c->n_experts,sizeof(uint32_t));
        }
        #undef P
    }
    /* resident bytes */
    int64_t rb=qt_bytes(&m->embed);
    for(int i=0;i<c->n_layers;i++){ Layer *l=&m->L[i];
        rb+=qt_bytes(&l->q_proj)+qt_bytes(&l->k_proj)+qt_bytes(&l->o_proj);
        if(!l->sparse) rb+=qt_bytes(&l->dense_gate)+qt_bytes(&l->dense_up)+qt_bytes(&l->dense_down);
    }
    m->resident_bytes=rb;
}

/* ---- embed (Gemma scales embeddings by sqrt(hidden_size)) ---- */
static void embed_row(Model *m, int tok, float *x){
    int D=m->c.hidden; QT *e=&m->embed;
    float emb_scale=sqrtf((float)D);
    if(e->fmt==0){ for(int i=0;i<D;i++) x[i]=e->qf[(int64_t)tok*D+i]*emb_scale; return; }
    if(e->fmt==1){ const int8_t *q=e->q8+(int64_t)tok*D; float s=e->s[tok]*emb_scale;
        for(int i=0;i<D;i++) x[i]=(float)q[i]*s; return; }
    if(e->fmt==2){ const uint8_t *q=e->q4+(int64_t)tok*((D+1)/2); float s=e->s[tok]*emb_scale;
        for(int i=0;i<D;i+=2){ uint8_t byte=q[i>>1]; x[i]=(float)((int)(byte&0xF)-8)*s;
            if(i+1<D) x[i+1]=(float)((int)(byte>>4)-8)*s; }
        return; }
    const uint8_t *q=e->q4+(int64_t)tok*((D+3)/4); float s=e->s[tok]*emb_scale;
    for(int i=0;i<D;i++){ uint8_t byte=q[i>>2]; int sh=(i&3)*2; x[i]=(float)((int)((byte>>sh)&3)-2)*s; }
}

/* ---- expert loading from fused tensors ----
 * Gemma 4 has one [n_experts, 2*moe_inter, hidden] tensor for gate_up_proj per layer
 * and one [n_experts, hidden, moe_inter] tensor for down_proj per layer.
 * We pread a slice for each expert. */
static void expert_load(Model *m, int layer, int eid, ESlot *s){
#ifdef COLI_CUDA
    if(s->eid!=eid){ qt_cuda_reset(&s->g); qt_cuda_reset(&s->u); qt_cuda_reset(&s->d); }
#endif
#ifdef COLI_METAL
    if(s->eid!=eid){ qt_metal_reset(&s->g); qt_metal_reset(&s->u); qt_metal_reset(&s->d); }
#endif
    Cfg *c=&m->c; int I=c->moe_inter, D=c->hidden, b=m->ebits;
    Layer *l=&m->L[layer];

    /* Try per-expert pre-quantized format first (from convert_gemma4_to_int4.py),
     * then fall back to fused BF16 tensor slicing. */
    char nm[3][300]; const char *suf[3]={"gate_proj","up_proj","down_proj"};
    int dims[3][2]={{I,D},{I,D},{D,I}};
    for(int k=0;k<3;k++) snprintf(nm[k],sizeof(nm[k]),
        "model.language_model.layers.%d.mlp.experts.%d.%s.weight",layer,eid,suf[k]);
    char qn[320]; snprintf(qn,sizeof(qn),"%s.qs",nm[0]);
    if(st_has(&m->S,qn)){
        /* pre-quantized per-expert container */
        qt_from_disk(m,nm[0],dims[0][0],dims[0][1],b,g_drop,&s->g);
        qt_from_disk(m,nm[1],dims[1][0],dims[1][1],b,g_drop,&s->u);
        qt_from_disk(m,nm[2],dims[2][0],dims[2][1],b,g_drop,&s->d);
        s->eid=eid; return;
    }

    /* Fused BF16 fallback: gate_up_proj [n_experts, 2*I, D], down_proj [n_experts, D, I] */
    int64_t gu_elems = (int64_t)2*I*D;
    float *gu_buf = falloc(gu_elems);
    st_read_slice_f32(&m->S, l->gate_up_name, (int64_t)eid*gu_elems, gu_elems, gu_buf, g_drop);
    float *gate_buf = gu_buf;
    float *up_buf   = gu_buf + (int64_t)I*D;

    int64_t down_elems = (int64_t)D*I;
    float *down_buf = falloc(down_elems);
    st_read_slice_f32(&m->S, l->down_name, (int64_t)eid*down_elems, down_elems, down_buf, g_drop);

    if(!s->g.qf && !s->g.q8 && !s->g.q4) qt_alloc(&s->g, I, D, b);
    qt_fill(&s->g, gate_buf, b);
    if(!s->u.qf && !s->u.q8 && !s->u.q4) qt_alloc(&s->u, I, D, b);
    qt_fill(&s->u, up_buf, b);
    if(!s->d.qf && !s->d.q8 && !s->d.q4) qt_alloc(&s->d, D, I, b);
    qt_fill(&s->d, down_buf, b);

    free(gu_buf); free(down_buf);
    s->eid=eid;
}

static void expert_prefetch(Model *m, int layer, int eid){
    Layer *l=&m->L[layer];
    /* prefetch the fused tensors — the kernel will read-ahead the right pages */
    st_prefetch(&m->S, l->gate_up_name);
    st_prefetch(&m->S, l->down_name);
}

/* ---- Hybrid Attention with K=V and QK-Norm ---- */
static void attention(Model *m, Layer *l, int layer, float *x, int S, int pos_base, float *out){
    Cfg *c=&m->c; int H=c->n_heads;
    int ltype = c->layer_type[layer];
    int hd = ltype==LTYPE_GLOBAL ? c->global_head_dim : c->slide_head_dim;
    int Hkv = ltype==LTYPE_GLOBAL ? c->global_kv_heads : c->slide_kv_heads;
    float theta = ltype==LTYPE_GLOBAL ? c->global_rope_theta : c->slide_rope_theta;
    float prf = ltype==LTYPE_GLOBAL ? c->global_partial_rotary : c->slide_partial_rotary;
    int rope_dim = (int)(hd * prf);
    int rep = H / Hkv;
    float scale = 1.f / sqrtf((float)hd);
    int window = (ltype==LTYPE_SLIDING) ? c->sliding_window : 0; /* 0 = no window = full causal */
    double ta0=now_s();

    float *Q = falloc((int64_t)S * H * hd);
    float *KVnew = falloc((int64_t)S * Hkv * hd);
    float *ctx = falloc((int64_t)S * H * hd);

    /* 1) Project Q, K (=V) */
    matmul_qt(Q, x, &l->q_proj, S);
    matmul_qt(KVnew, x, &l->k_proj, S);  /* K=V: same projection for both */

    /* 2) QK-Norm: RMSNorm on each head of Q and K independently, with learned weights */
    for(int s=0;s<S;s++){
        for(int h=0;h<H;h++){
            float *qh = Q + (int64_t)(s*H+h)*hd;
            /* apply per-head rmsnorm with q_norm weights */
            float tmp[512]; /* hd <= 512 */
            rmsnorm_unit(tmp, qh, hd, c->eps);
            for(int d=0;d<hd;d++) qh[d] = tmp[d] * l->q_norm[d];
        }
        for(int h=0;h<Hkv;h++){
            float *kh = KVnew + (int64_t)(s*Hkv+h)*hd;
            float tmp[512];
            rmsnorm_unit(tmp, kh, hd, c->eps);
            for(int d=0;d<hd;d++) kh[d] = tmp[d] * l->k_norm[d];
        }
    }

    /* 3) Apply partial RoPE */
    for(int s=0;s<S;s++){
        int pos=pos_base+s;
        for(int h=0;h<H;h++)
            rope_half(Q + (int64_t)(s*H+h)*hd, pos, rope_dim, theta);
        for(int h=0;h<Hkv;h++)
            rope_half(KVnew + (int64_t)(s*Hkv+h)*hd, pos, rope_dim, theta);
    }

    /* 4) Store K and V in cache (K=V, so both get the same values) */
    for(int s=0;s<S;s++){
        int pos=pos_base+s;
        memcpy(m->Kc[layer] + (int64_t)pos*Hkv*hd,
               KVnew + (int64_t)s*Hkv*hd, Hkv*hd*sizeof(float));
        memcpy(m->Vc[layer] + (int64_t)pos*Hkv*hd,
               KVnew + (int64_t)s*Hkv*hd, Hkv*hd*sizeof(float));
    }

    /* 5) Attention: Q @ K^T * scale, causal mask (+ sliding window if applicable), softmax, @ V */
    #pragma omp parallel for collapse(2) schedule(static)
    for(int s=0;s<S;s++) for(int h=0;h<H;h++){
        int pos=pos_base+s;
        int kv_h = h / rep;
        const float *qh = Q + (int64_t)(s*H+h)*hd;
        /* determine attention range */
        int t_start = 0;
        if(window > 0){
            t_start = pos - window + 1;
            if(t_start < 0) t_start = 0;
        }
        int n_attend = pos - t_start + 1;
        float *sc = (float*)malloc(n_attend*sizeof(float));
        for(int ti=0;ti<n_attend;ti++){
            int t = t_start + ti;
            const float *kt = m->Kc[layer] + (int64_t)(t*Hkv+kv_h)*hd;
            float a=0;
            for(int d=0;d<hd;d++) a+=qh[d]*kt[d];
            sc[ti]=a*scale;
        }
        softmax(sc, n_attend);
        float *cx = ctx + (int64_t)(s*H+h)*hd;
        memset(cx, 0, hd*sizeof(float));
        for(int ti=0;ti<n_attend;ti++){
            int t = t_start + ti;
            const float *vt = m->Vc[layer] + (int64_t)(t*Hkv+kv_h)*hd;
            float a=sc[ti];
            for(int d=0;d<hd;d++) cx[d]+=a*vt[d];
        }
        free(sc);
    }

    /* 6) Output projection */
    matmul_qt(out, ctx, &l->o_proj, S);

    free(Q); free(KVnew); free(ctx);
    m->t_attn += now_s()-ta0;
}

/* ---- MoE with Gemma 4 router (softmax + per_expert_scale) ---- */
static void moe(Model *m, Layer *l, int layer, float *x, int S, float *out){
    Cfg *c=&m->c; int D=c->hidden, E=c->n_experts, K=c->topk, I=c->moe_inter;
    float *logit=falloc(E), *scores=falloc(E);
    int *idxs=malloc((size_t)S*K*sizeof(int)); float *ws=malloc((size_t)S*K*sizeof(float));
    int *keff=malloc(S*sizeof(int));

    for(int s=0;s<S;s++){
        const float *xs=x+(int64_t)s*D;
        /* logits = x @ router_proj^T * router_scale */
        matmul(logit, xs, l->router_proj, 1, D, E);
        for(int e=0;e<E;e++) logit[e] *= l->router_scale;
        /* softmax */
        softmax(logit, E);
        /* multiply by per_expert_scale */
        for(int e=0;e<E;e++) scores[e] = logit[e] * l->per_expert_scale[e];
        /* top-K selection */
        int *idx=idxs+(int64_t)s*K; float *w=ws+(int64_t)s*K;
        int Ksel = g_topk>0 ? (g_topk<K?g_topk:K) : K;
        for(int kk=0;kk<Ksel;kk++){ int best=-1; float bv=-1e30f;
            for(int e=0;e<E;e++){ int tk=0; for(int j=0;j<kk;j++) if(idx[j]==e){tk=1;break;}
                if(!tk && scores[e]>bv){bv=scores[e];best=e;} }
            idx[kk]=best; w[kk]=scores[best];
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
        /* normalize weights for selected experts */
        { float sm=0; for(int kk=0;kk<Ke;kk++) sm+=w[kk]; sm+=1e-20f; for(int kk=0;kk<Ke;kk++) w[kk]/=sm; }
        for(int d=0;d<D;d++) out[(int64_t)s*D+d]=0;
    }
    m->enr[layer]=keff[S-1]; for(int kk=0;kk<keff[S-1];kk++) m->eroute[layer][kk]=idxs[(int64_t)(S-1)*K+kk];

    /* union of experts in batch */
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
            /* GELU activation: out = gelu(gate) * up */
            for(int64_t z=0;z<(int64_t)nr*I;z++) gg[z]=gelu_tanh(gg[z])*uu[z];
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
    free(logit); free(scores); free(idxs); free(ws); free(keff); free(uniq);
    free(xg); free(gg); free(uu); free(hh); free(rows); free(rw);
}

/* ---- dense MLP (layer 0) with GELU ---- */
static void dense_mlp(Layer *l, float *x, int S, int D, int I, float *out){
    float *g=falloc((int64_t)S*I), *u=falloc((int64_t)S*I);
    matmul_qt(g, x, &l->dense_gate, S);
    matmul_qt(u, x, &l->dense_up,   S);
    for(int64_t i=0;i<(int64_t)S*I;i++) g[i]=gelu_tanh(g[i])*u[i];
    matmul_qt(out, g, &l->dense_down, S);
    free(g); free(u);
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
    if(!l->sparse) return;
    if(!pilot_m){ pilot_m=m; pthread_t t; pthread_create(&t,NULL,pilot_worker,NULL); }
    float *nrm=falloc(D), *ch=falloc(E);
    for(int s=0;s<S;s++){
        rmsnorm(nrm, x+(int64_t)s*D, l->post_ln, D, c->eps);
        /* Gemma 4 router: matmul, scale, softmax, per_expert_scale */
        matmul(ch, nrm, l->router_proj, 1, D, E);
        for(int e=0;e<E;e++) ch[e] *= l->router_scale;
        softmax(ch, E);
        for(int e=0;e<E;e++) ch[e] *= l->per_expert_scale[e];
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

/* ---- forward di UN layer ---- */
static void layer_forward(Model *m, Layer *l, int li, float *x, int S, int pos_base, float *nrm, float *tmp){
    Cfg *c=&m->c; int D=c->hidden;
    if(g_spec && g_prefetch && l->sparse && m->enr[li]>0)
        for(int z=0;z<m->enr[li];z++) expert_prefetch(m,li,m->eroute[li][z]);

    /* attention sub-layer */
    for(int s=0;s<S;s++) rmsnorm(nrm+(int64_t)s*D, x+(int64_t)s*D, l->in_ln, D, c->eps);
    attention(m,l,li,nrm,S,pos_base,tmp);
    /* residual with layer_scalar */
    for(int64_t j=0;j<(int64_t)S*D;j++) x[j] += l->layer_scalar * tmp[j];

    if(g_pilot && S<=8 && li+1<c->n_layers && m->L[li+1].sparse) pilot_prefetch(m,li+1,x,S);

    /* FFN sub-layer */
    if(l->sparse){
        /* MoE path with extra layernorms */
        for(int s=0;s<S;s++) rmsnorm(nrm+(int64_t)s*D, x+(int64_t)s*D, l->pre_ff_ln, D, c->eps);
        moe(m,l,li,nrm,S,tmp);
        if(l->post_ff_ln)
            for(int s=0;s<S;s++){
                float *ts=tmp+(int64_t)s*D;
                float buf[4096]; /* D <= 2816 */
                rmsnorm(buf, ts, l->post_ff_ln, D, c->eps);
                memcpy(ts, buf, D*sizeof(float));
            }
    } else {
        /* dense path with pre/post feedforward layernorms */
        for(int s=0;s<S;s++) rmsnorm(nrm+(int64_t)s*D, x+(int64_t)s*D, l->pre_ff_ln, D, c->eps);
        dense_mlp(l,nrm,S,D,c->dense_inter,tmp);
        if(l->post_ff_ln)
            for(int s=0;s<S;s++){
                float *ts=tmp+(int64_t)s*D;
                float buf[4096];
                rmsnorm(buf, ts, l->post_ff_ln, D, c->eps);
                memcpy(ts, buf, D*sizeof(float));
            }
    }
    /* residual with layer_scalar */
    for(int64_t j=0;j<(int64_t)S*D;j++) x[j] += l->layer_scalar * tmp[j];
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

/* ---- KV cache allocation (dual head_dim) ---- */
static void kv_alloc(Model *m, int max_t){
    Cfg *c=&m->c;
    if(m->Kc){ for(int i=0;i<c->n_layers;i++){ free(m->Kc[i]); free(m->Vc[i]); } free(m->Kc); free(m->Vc); }
    m->max_t=max_t;
    m->Kc=calloc(c->n_layers,sizeof(float*));
    m->Vc=calloc(c->n_layers,sizeof(float*));
    for(int i=0;i<c->n_layers;i++){
        int ltype = c->layer_type[i];
        int hd = ltype==LTYPE_GLOBAL ? c->global_head_dim : c->slide_head_dim;
        int Hkv = ltype==LTYPE_GLOBAL ? c->global_kv_heads : c->slide_kv_heads;
        m->Kc[i]=falloc((int64_t)max_t*Hkv*hd);
        m->Vc[i]=falloc((int64_t)max_t*Hkv*hd);
    }
}

/* ---- step (forward + lm_head via tied embeddings + logit softcapping) ---- */
static float *step(Model *m, const int *ids, int S, int pos_base){
    Cfg *c=&m->c; int D=c->hidden;
    float *x=falloc((int64_t)S*D);
    for(int s=0;s<S;s++) embed_row(m, ids[s], x+(int64_t)s*D);
    layers_forward(m,x,S,pos_base);
    float *last=falloc(D); rmsnorm(last, x+(int64_t)(S-1)*D, m->final_norm, D, c->eps);
    double th0=now_s();
    /* lm_head = embed transposed (tie_word_embeddings) */
    float *logit=falloc(c->vocab); matmul_qt(logit,last,&m->embed,1);
    /* logit softcapping */
    if(c->logit_cap > 0){
        float cap=c->logit_cap;
        for(int i=0;i<c->vocab;i++) logit[i] = cap * tanhf(logit[i] / cap);
    }
    m->t_head += now_s()-th0;
    free(x); free(last); return logit;
}

static float *step_all(Model *m, const int *ids, int S, int pos_base){
    Cfg *c=&m->c; int D=c->hidden;
    float *x=falloc((int64_t)S*D);
    for(int s=0;s<S;s++) embed_row(m, ids[s], x+(int64_t)s*D);
    layers_forward(m,x,S,pos_base);
    float *lo=falloc((int64_t)S*c->vocab), *row=falloc(D);
    for(int s=0;s<S;s++){
        rmsnorm(row, x+(int64_t)s*D, m->final_norm, D, c->eps);
        matmul_qt(lo+(int64_t)s*c->vocab, row, &m->embed, 1);
        /* logit softcapping */
        if(c->logit_cap > 0){
            float cap=c->logit_cap;
            float *ls=lo+(int64_t)s*c->vocab;
            for(int i=0;i<c->vocab;i++) ls[i] = cap * tanhf(ls[i] / cap);
        }
    }
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

/* ---- sampling ---- */
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
    fprintf(stderr,"[stop] %d stop token:",g_nstop);
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
        matmul_qt(lo, row, &m->embed, 1);
        if(c->logit_cap > 0){ float cap=c->logit_cap; for(int i=0;i<c->vocab;i++) lo[i]=cap*tanhf(lo[i]/cap); }
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

/* generazione reale */
static void run_text(Model *m, const char *snap, const char *prompt, int ngen){
    Cfg *c=&m->c; char tkp[2048]; snprintf(tkp,sizeof(tkp),"%s/tokenizer.json",snap);
    Tok T; tok_load(&T,tkp);
    int eos=tok_id_of(&T,"<end_of_turn>");
    if(eos<0) eos=tok_id_of(&T,"<eos>");
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
    Cfg *c=&m->c;
    /* for fused weights: gate_up_proj is [n_experts, 2*I, D], down_proj is [n_experts, D, I]
     * per-expert bytes = tbytes(I,D,ebits)*2 + tbytes(D,I,ebits) */
    return tbytes(c->moe_inter,c->hidden,ebits)*2 + tbytes(c->hidden,c->moe_inter,ebits);
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
        if(s->g.qf||s->g.q8||s->g.q4){
            int64_t gb=qt_bytes(&s->g)+qt_bytes(&s->u)+qt_bytes(&s->d);
            /* wire the QT weight buffers directly */
            void *gp = s->g.fmt==0 ? (void*)s->g.qf : s->g.fmt==1 ? (void*)s->g.q8 : (void*)s->g.q4;
            if(gp && mem_wire(gp, qt_bytes(&s->g))==0) wired+=qt_bytes(&s->g); else failed++;
            void *up = s->u.fmt==0 ? (void*)s->u.qf : s->u.fmt==1 ? (void*)s->u.q8 : (void*)s->u.q4;
            if(up && mem_wire(up, qt_bytes(&s->u))==0) wired+=qt_bytes(&s->u); else failed++;
            void *dp = s->d.fmt==0 ? (void*)s->d.qf : s->d.fmt==1 ? (void*)s->d.q8 : (void*)s->d.q4;
            if(dp && mem_wire(dp, qt_bytes(&s->d))==0) wired+=qt_bytes(&s->d); else failed++;
            (void)gb;
        }
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
    Cfg *c=&m->c; double total=0;
    for(int i=0;i<c->n_layers;i++){
        int ltype=c->layer_type[i];
        int hd = ltype==LTYPE_GLOBAL ? c->global_head_dim : c->slide_head_dim;
        int Hkv = ltype==LTYPE_GLOBAL ? c->global_kv_heads : c->slide_kv_heads;
        total += (double)max_ctx*Hkv*hd*2.0*4.0;
    }
    return total;
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
    printf("== Motore C Gemma4 (Gemma4ForConditionalGeneration, text-only), cache=%d expert/layer | expert@%d-bit densa@%d-bit | idot: " IDOT_KERNEL " ==\n", cap, ebits, dbits);
    g_mem_avail_boot = mem_available_gb();
    Model m; double t0=now_s(); model_init(&m,snap,cap,ebits,dbits);
    Cfg *cc=&m.c;
    int ng=0,ns=0; for(int i=0;i<cc->n_layers;i++) if(cc->layer_type[i]==LTYPE_GLOBAL) ng++; else ns++;
    printf("caricato in %.2fs | densa residente: %.2f MB | layers=%d (%d sliding + %d global) | experts=%d\n",
           now_s()-t0, m.resident_bytes/(1024.0*1024.0), cc->n_layers, ns, ng, cc->n_experts);
    printf("heads=%d | sliding: hd=%d kv=%d | global: hd=%d kv=%d | window=%d\n",
           cc->n_heads, cc->slide_head_dim, cc->slide_kv_heads,
           cc->global_head_dim, cc->global_kv_heads, cc->sliding_window);
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

    /* altrimenti: validazione contro l'oracolo */
    const char *refpath=getenv("REF")?getenv("REF"):"ref_gemma4.json";
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
    printf("\nMotore C Gemma4      : "); for(int i=np;i<nfull;i++){ printf("%d ", out[i]); if(out[i]==full[i])match++; }
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
