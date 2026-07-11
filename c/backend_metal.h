#ifndef COLIBRI_BACKEND_METAL_H
#define COLIBRI_BACKEND_METAL_H

#include <stddef.h>
#include <stdint.h>

/*
 * Metal GPU backend for Apple Silicon — mirrors the CUDA backend interface.
 *
 * Key difference from CUDA: Apple's unified memory means weights already live
 * in GPU-accessible RAM. "Upload" creates a MTLBuffer wrapping existing memory
 * (MTLResourceStorageModeShared / makeBuffer:bytesNoCopy:) instead of copying
 * to discrete VRAM. This makes tensor_upload nearly free and expert slot reuse
 * viable on GPU.
 *
 * Build: make glm METAL=1   (macOS only)
 */

typedef struct ColiMetalTensor ColiMetalTensor;

int   coli_metal_init(void);
void  coli_metal_shutdown(void);
int   coli_metal_mem_info(size_t *free_bytes, size_t *total_bytes);
void  coli_metal_stats(size_t *tensor_count, size_t *tensor_bytes);

int   coli_metal_tensor_upload(ColiMetalTensor **tensor,
                               const void *weights, const float *scales,
                               int fmt, int I, int O);

/*
 * y[S,O] = x[S,I] @ W[O,I]^T.
 * fmt: 0=f32, 1=int8, 2=int4, 3=int2.
 * On Apple Silicon the weights pointer may alias unified memory already
 * wrapped by the tensor — no redundant copy occurs.
 */
int   coli_metal_matmul(ColiMetalTensor **tensor,
                        float *y, const float *x,
                        const void *weights, const float *scales,
                        int fmt, int S, int I, int O);

void  coli_metal_tensor_free(ColiMetalTensor *tensor);
size_t coli_metal_tensor_bytes(const ColiMetalTensor *tensor);

#endif
