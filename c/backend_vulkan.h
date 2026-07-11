#ifndef COLIBRI_BACKEND_VULKAN_H
#define COLIBRI_BACKEND_VULKAN_H

#include <stddef.h>
#include <stdint.h>

/*
 * Vulkan GPU backend for Colibri — mirrors the Metal backend interface.
 *
 * Targets Vulkan 1.0+ compute (RPi 5 VideoCore VII, discrete GPUs, etc.).
 * On devices with unified memory (HOST_VISIBLE | DEVICE_LOCAL), weights are
 * mapped zero-copy, similar to Metal's bytesNoCopy approach.  On discrete
 * GPUs a staging buffer is used for the initial upload.
 *
 * Build: make glm VULKAN=1   (requires Vulkan SDK / libvulkan)
 */

#ifdef COLI_VULKAN

typedef struct ColiVulkanTensor ColiVulkanTensor;

int    coli_vulkan_init(void);
void   coli_vulkan_shutdown(void);
int    coli_vulkan_mem_info(size_t *free_bytes, size_t *total_bytes);
void   coli_vulkan_stats(size_t *tensor_count, size_t *tensor_bytes);

int    coli_vulkan_tensor_upload(ColiVulkanTensor **tensor,
                                 const void *weights, const float *scales,
                                 int fmt, int I, int O);

/*
 * y[S,O] = x[S,I] @ W[O,I]^T.
 * fmt: 0=f32, 1=int8, 2=int4, 3=int2.
 * On unified-memory devices the weights pointer may alias a host-visible
 * buffer — no redundant copy occurs.
 */
int    coli_vulkan_matmul(ColiVulkanTensor **tensor,
                          float *y, const float *x,
                          const void *weights, const float *scales,
                          int fmt, int S, int I, int O);

void   coli_vulkan_tensor_free(ColiVulkanTensor *tensor);
size_t coli_vulkan_tensor_bytes(const ColiVulkanTensor *tensor);

#endif /* COLI_VULKAN */

#endif /* COLIBRI_BACKEND_VULKAN_H */
