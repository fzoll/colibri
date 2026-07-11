/*
 * backend_metal.m — Metal GPU backend for Colibri (Apple Silicon).
 *
 * Mirrors backend_cuda.cu but exploits unified memory:
 * - tensor_upload wraps existing RAM pointers (bytesNoCopy) → zero-copy
 * - matmul dispatches MSL compute kernels for quantized dot products
 * - single device (Apple Silicon GPU), no multi-device complexity
 *
 * Build: make glm METAL=1  (macOS only, requires Xcode command-line tools)
 *
 * TODO: implement — this is the scaffold for the next coding session.
 */

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "backend_metal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── Metal state ─────────────────────────────────────────────────── */

static id<MTLDevice>       g_device;
static id<MTLCommandQueue>  g_queue;
static id<MTLLibrary>       g_library;       /* compiled .metal shaders */

/* One pipeline state per quantization format (fmt 0-3). */
static id<MTLComputePipelineState> g_matmul_pipe[4];

static size_t g_tensor_count, g_tensor_bytes;

struct ColiMetalTensor {
    id<MTLBuffer> weights_buf;
    id<MTLBuffer> scales_buf;
    size_t weight_bytes;
    int fmt, I, O;
    int tracked;
};

/* ── Shader source (embedded MSL) ────────────────────────────────
 *
 * Embedding the shader as a string avoids needing a separate .metal file
 * and xcrun metallib compile step — the library is compiled at runtime via
 * newLibraryWithSource:. Cost: ~50ms one-time at init.
 *
 * The kernel mirrors backend_cuda.cu's quant_matmul:
 *   - one threadgroup per (output_row, sequence)
 *   - threads cooperatively reduce across the input dimension
 *   - shared memory reduction (like CUDA's __shared__ partial[256])
 */
static const char *g_metal_shader_src =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"\n"
"/* Two-stage reduction: simd_sum across 32-wide SIMD lane, then\n"
" * threadgroup reduction across SIMD groups. On Apple GPU the SIMD\n"
" * width is 32, so 256 threads = 8 SIMD groups = 3 barrier steps\n"
" * instead of 8. */\n"
"inline float tg_reduce_sum(float val, uint tid,\n"
"                           threadgroup float *shared) {\n"
"    val = simd_sum(val);\n"
"    uint lane = tid & 31;\n"
"    uint sg   = tid >> 5;\n"
"    if (lane == 0) shared[sg] = val;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    if (sg == 0) {\n"
"        val = (tid < 8) ? shared[tid] : 0.0f;\n"
"        val = simd_sum(val);\n"
"    }\n"
"    return val;\n"
"}\n"
"\n"
"/* Unpack int4 nibble: two values per byte, bias -8. */\n"
"inline float unpack_i4(device const uint8_t *base, int i) {\n"
"    uint8_t v = base[i >> 1];\n"
"    return float(int((i & 1) ? (v >> 4) : (v & 0x0F)) - 8);\n"
"}\n"
"\n"
"/* Unpack int2: four values per byte, bias -2. */\n"
"inline float unpack_i2(device const uint8_t *base, int i) {\n"
"    uint8_t v = base[i >> 2];\n"
"    return float(int((v >> ((i & 3) * 2)) & 3) - 2);\n"
"}\n"
"\n"
"kernel void quant_matmul_i8(\n"
"    device const float  *x      [[buffer(0)]],\n"
"    device const uint8_t *weights [[buffer(1)]],\n"
"    device const float  *scales  [[buffer(2)]],\n"
"    device       float  *y      [[buffer(3)]],\n"
"    constant     int    &S      [[buffer(4)]],\n"
"    constant     int    &I      [[buffer(5)]],\n"
"    constant     int    &O      [[buffer(6)]],\n"
"    constant     int    &rb     [[buffer(7)]],\n"
"    uint3 gid    [[threadgroup_position_in_grid]],\n"
"    uint3 tid_v  [[thread_position_in_threadgroup]],\n"
"    uint3 tsize_v [[threads_per_threadgroup]]\n"
") {\n"
"    int o = gid.x;\n"
"    int s = gid.y;\n"
"    device const float *xs = x + s * I;\n"
"    device const uint8_t *row = weights + o * rb;\n"
"    float sum = 0.0f;\n"
"    uint tid = tid_v.x; uint tsize = tsize_v.x;\n"
"    for (int i = tid; i < I; i += tsize)\n"
"        sum += xs[i] * float(((device const int8_t *)row)[i]);\n"
"    threadgroup float shared[8];\n"
"    sum = tg_reduce_sum(sum, tid, shared);\n"
"    if (tid == 0)\n"
"        y[s * O + o] = sum * scales[o];\n"
"}\n"
"\n"
"kernel void quant_matmul_i4(\n"
"    device const float  *x      [[buffer(0)]],\n"
"    device const uint8_t *weights [[buffer(1)]],\n"
"    device const float  *scales  [[buffer(2)]],\n"
"    device       float  *y      [[buffer(3)]],\n"
"    constant     int    &S      [[buffer(4)]],\n"
"    constant     int    &I      [[buffer(5)]],\n"
"    constant     int    &O      [[buffer(6)]],\n"
"    constant     int    &rb     [[buffer(7)]],\n"
"    uint3 gid    [[threadgroup_position_in_grid]],\n"
"    uint3 tid_v  [[thread_position_in_threadgroup]],\n"
"    uint3 tsize_v [[threads_per_threadgroup]]\n"
") {\n"
"    int o = gid.x;\n"
"    int s = gid.y;\n"
"    device const float *xs = x + s * I;\n"
"    device const uint8_t *row = weights + o * rb;\n"
"    float sum = 0.0f;\n"
"    uint tid = tid_v.x; uint tsize = tsize_v.x;\n"
"    for (int i = tid; i < I; i += tsize)\n"
"        sum += xs[i] * unpack_i4(row, i);\n"
"    threadgroup float shared[8];\n"
"    sum = tg_reduce_sum(sum, tid, shared);\n"
"    if (tid == 0)\n"
"        y[s * O + o] = sum * scales[o];\n"
"}\n"
"\n"
"kernel void quant_matmul_i2(\n"
"    device const float  *x      [[buffer(0)]],\n"
"    device const uint8_t *weights [[buffer(1)]],\n"
"    device const float  *scales  [[buffer(2)]],\n"
"    device       float  *y      [[buffer(3)]],\n"
"    constant     int    &S      [[buffer(4)]],\n"
"    constant     int    &I      [[buffer(5)]],\n"
"    constant     int    &O      [[buffer(6)]],\n"
"    constant     int    &rb     [[buffer(7)]],\n"
"    uint3 gid    [[threadgroup_position_in_grid]],\n"
"    uint3 tid_v  [[thread_position_in_threadgroup]],\n"
"    uint3 tsize_v [[threads_per_threadgroup]]\n"
") {\n"
"    int o = gid.x;\n"
"    int s = gid.y;\n"
"    device const float *xs = x + s * I;\n"
"    device const uint8_t *row = weights + o * rb;\n"
"    float sum = 0.0f;\n"
"    uint tid = tid_v.x; uint tsize = tsize_v.x;\n"
"    for (int i = tid; i < I; i += tsize)\n"
"        sum += xs[i] * unpack_i2(row, i);\n"
"    threadgroup float shared[8];\n"
"    sum = tg_reduce_sum(sum, tid, shared);\n"
"    if (tid == 0)\n"
"        y[s * O + o] = sum * scales[o];\n"
"}\n"
"\n"
"kernel void matmul_f32(\n"
"    device const float *x      [[buffer(0)]],\n"
"    device const float *weights [[buffer(1)]],\n"
"    device       float *y      [[buffer(3)]],\n"
"    constant     int   &S      [[buffer(4)]],\n"
"    constant     int   &I      [[buffer(5)]],\n"
"    constant     int   &O      [[buffer(6)]],\n"
"    uint3 gid    [[threadgroup_position_in_grid]],\n"
"    uint3 tid_v  [[thread_position_in_threadgroup]],\n"
"    uint3 tsize_v [[threads_per_threadgroup]]\n"
") {\n"
"    int o = gid.x;\n"
"    int s = gid.y;\n"
"    device const float *xs = x + s * I;\n"
"    device const float *row = weights + o * I;\n"
"    float sum = 0.0f;\n"
"    uint tid = tid_v.x; uint tsize = tsize_v.x;\n"
"    for (int i = tid; i < I; i += tsize)\n"
"        sum += xs[i] * row[i];\n"
"    threadgroup float shared[8];\n"
"    sum = tg_reduce_sum(sum, tid, shared);\n"
"    if (tid == 0)\n"
"        y[s * O + o] = sum;\n"
"}\n";

static const char *kernel_names[4] = {
    "matmul_f32", "quant_matmul_i8", "quant_matmul_i4", "quant_matmul_i2"
};

/* ── Helpers ─────────────────────────────────────────────────────── */

static size_t row_bytes(int fmt, int I) {
    if (fmt == 0) return (size_t)I * sizeof(float);
    if (fmt == 1) return (size_t)I;
    if (fmt == 2) return (size_t)(I + 1) / 2;
    if (fmt == 3) return (size_t)(I + 3) / 4;
    return 0;
}

/* ── Public API ──────────────────────────────────────────────────── */

int coli_metal_init(void) {
    g_device = MTLCreateSystemDefaultDevice();
    if (!g_device) {
        fprintf(stderr, "[Metal] no GPU device found\n");
        return 0;
    }
    g_queue = [g_device newCommandQueue];
    if (!g_queue) {
        fprintf(stderr, "[Metal] failed to create command queue\n");
        g_device = nil;
        return 0;
    }

    /* Compile shaders from embedded source. */
    NSError *err = nil;
    NSString *src = [NSString stringWithUTF8String:g_metal_shader_src];
    g_library = [g_device newLibraryWithSource:src options:nil error:&err];
    if (!g_library) {
        fprintf(stderr, "[Metal] shader compile failed: %s\n",
                err ? [[err localizedDescription] UTF8String] : "unknown");
        g_queue = nil; g_device = nil;
        return 0;
    }

    /* Create pipeline states for each format. */
    for (int fmt = 0; fmt < 4; fmt++) {
        id<MTLFunction> fn = [g_library newFunctionWithName:
                              [NSString stringWithUTF8String:kernel_names[fmt]]];
        if (!fn) {
            fprintf(stderr, "[Metal] kernel '%s' not found\n", kernel_names[fmt]);
            g_queue = nil; g_device = nil; g_library = nil;
            return 0;
        }
        g_matmul_pipe[fmt] = [g_device newComputePipelineStateWithFunction:fn error:&err];
        if (!g_matmul_pipe[fmt]) {
            fprintf(stderr, "[Metal] pipeline '%s' failed: %s\n", kernel_names[fmt],
                    err ? [[err localizedDescription] UTF8String] : "unknown");
            g_queue = nil; g_device = nil; g_library = nil;
            return 0;
        }
    }

    fprintf(stderr, "[Metal] %s, %.1f GB unified memory\n",
            [[g_device name] UTF8String],
            [g_device recommendedMaxWorkingSetSize] / 1e9);
    g_tensor_count = 0;
    g_tensor_bytes = 0;
    return 1;
}

void coli_metal_shutdown(void) {
    for (int i = 0; i < 4; i++) g_matmul_pipe[i] = nil;
    g_library = nil;
    g_queue = nil;
    g_device = nil;
    g_tensor_count = 0;
    g_tensor_bytes = 0;
}

int coli_metal_mem_info(size_t *free_bytes, size_t *total_bytes) {
    if (!g_device || !free_bytes || !total_bytes) return 0;
    size_t total = [g_device recommendedMaxWorkingSetSize];
    size_t used = [g_device currentAllocatedSize];
    *total_bytes = total;
    *free_bytes = total > used ? total - used : 0;
    return 1;
}

void coli_metal_stats(size_t *tensor_count, size_t *tensor_bytes) {
    if (tensor_count) *tensor_count = g_tensor_count;
    if (tensor_bytes) *tensor_bytes = g_tensor_bytes;
}

int coli_metal_tensor_upload(ColiMetalTensor **tensor,
                             const void *weights, const float *scales,
                             int fmt, int I, int O) {
    if (!tensor || !weights || !g_device || I < 1 || O < 1) return 0;
    size_t rb = row_bytes(fmt, I);
    if (!rb || (fmt && !scales)) return 0;

    /* Already uploaded — reuse. */
    if (*tensor) {
        ColiMetalTensor *t = *tensor;
        return t->fmt == fmt && t->I == I && t->O == O;
    }

    ColiMetalTensor *t = calloc(1, sizeof(*t));
    if (!t) return 0;
    t->fmt = fmt; t->I = I; t->O = O;
    t->weight_bytes = rb * (size_t)O;

    /*
     * Zero-copy: wrap the existing host pointer as a shared MTLBuffer.
     * Apple unified memory means the GPU reads directly from this RAM —
     * no DMA transfer, no VRAM copy. This is the key advantage over CUDA.
     *
     * Caveat: the caller must keep the pointer valid for the tensor's lifetime.
     * Colibri's resident tensors (dense layers) are allocated once at model load
     * and never freed, so this is safe.
     */
    t->weights_buf = [g_device newBufferWithBytesNoCopy:(void *)weights
                                                 length:t->weight_bytes
                                                options:MTLResourceStorageModeShared
                                            deallocator:nil];
    if (!t->weights_buf) {
        fprintf(stderr, "[Metal] failed to wrap weight buffer (%zu bytes)\n", t->weight_bytes);
        free(t);
        return 0;
    }

    if (fmt) {
        size_t scale_bytes = (size_t)O * sizeof(float);
        t->scales_buf = [g_device newBufferWithBytesNoCopy:(void *)scales
                                                    length:scale_bytes
                                                   options:MTLResourceStorageModeShared
                                               deallocator:nil];
        if (!t->scales_buf) {
            fprintf(stderr, "[Metal] failed to wrap scale buffer\n");
            t->weights_buf = nil;
            free(t);
            return 0;
        }
    }

    t->tracked = 1;
    g_tensor_count++;
    g_tensor_bytes += t->weight_bytes + (fmt ? (size_t)O * sizeof(float) : 0);
    *tensor = t;
    return 1;
}

int coli_metal_matmul(ColiMetalTensor **tensor,
                      float *y, const float *x,
                      const void *weights, const float *scales,
                      int fmt, int S, int I, int O) {
    if (S < 1 || !coli_metal_tensor_upload(tensor, weights, scales, fmt, I, O))
        return 0;

    ColiMetalTensor *t = *tensor;
    if (fmt < 0 || fmt > 3 || !g_matmul_pipe[fmt]) return 0;

    size_t rb = row_bytes(fmt, I);
    size_t x_bytes = (size_t)S * I * sizeof(float);
    size_t y_bytes = (size_t)S * O * sizeof(float);

    /* Wrap input/output as shared buffers (zero-copy). */
    id<MTLBuffer> x_buf = [g_device newBufferWithBytesNoCopy:(void *)x
                                                      length:x_bytes
                                                     options:MTLResourceStorageModeShared
                                                 deallocator:nil];
    id<MTLBuffer> y_buf = [g_device newBufferWithBytesNoCopy:y
                                                      length:y_bytes
                                                     options:MTLResourceStorageModeShared
                                                 deallocator:nil];
    if (!x_buf || !y_buf) return 0;

    int rb_int = (int)rb;

    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:g_matmul_pipe[fmt]];

    [enc setBuffer:x_buf              offset:0 atIndex:0];
    [enc setBuffer:t->weights_buf     offset:0 atIndex:1];
    if (fmt) [enc setBuffer:t->scales_buf offset:0 atIndex:2];
    [enc setBuffer:y_buf              offset:0 atIndex:3];
    [enc setBytes:&S      length:sizeof(int) atIndex:4];
    [enc setBytes:&I      length:sizeof(int) atIndex:5];
    [enc setBytes:&O      length:sizeof(int) atIndex:6];
    [enc setBytes:&rb_int length:sizeof(int) atIndex:7];

    MTLSize grid = MTLSizeMake(O, S, 1);
    MTLSize tg   = MTLSizeMake(256, 1, 1);
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
    [enc endEncoding];

    [cmd commit];
    [cmd waitUntilCompleted];

    if ([cmd status] != MTLCommandBufferStatusCompleted) {
        fprintf(stderr, "[Metal] matmul failed: %s\n",
                cmd.error ? [[cmd.error localizedDescription] UTF8String] : "unknown");
        return 0;
    }
    return 1;
}

void coli_metal_tensor_free(ColiMetalTensor *tensor) {
    if (!tensor) return;
    if (tensor->tracked) {
        size_t bytes = tensor->weight_bytes +
                       (tensor->fmt ? (size_t)tensor->O * sizeof(float) : 0);
        if (g_tensor_count) g_tensor_count--;
        if (g_tensor_bytes >= bytes) g_tensor_bytes -= bytes;
    }
    tensor->weights_buf = nil;
    tensor->scales_buf = nil;
    free(tensor);
}

size_t coli_metal_tensor_bytes(const ColiMetalTensor *tensor) {
    return tensor ? tensor->weight_bytes +
           (tensor->fmt ? (size_t)tensor->O * sizeof(float) : 0) : 0;
}
