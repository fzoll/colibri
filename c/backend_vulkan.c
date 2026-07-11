/*
 * backend_vulkan.c -- Vulkan compute backend for Colibri.
 *
 * Mirrors backend_metal.m but uses the Vulkan C API:
 * - tensor_upload creates VkBuffer + VkDeviceMemory for weights/scales.
 *   On unified-memory devices (HOST_VISIBLE | DEVICE_LOCAL) weights are
 *   mapped directly (zero-copy, like Metal's bytesNoCopy).  On discrete
 *   GPUs a staging buffer + copy is used.
 * - matmul dispatches GLSL compute shaders (pre-compiled SPIR-V embedded
 *   as static uint32_t arrays) for quantized dot products.
 * - Synchronous execution: vkQueueWaitIdle after each matmul, matching
 *   Metal's waitUntilCompleted simplicity.
 *
 * Build: make glm VULKAN=1  (requires Vulkan SDK / libvulkan)
 *
 * SPIR-V regeneration from GLSL source (shaders/matmul.comp):
 *   glslangValidator -V shaders/matmul.comp -o shaders/matmul.spv
 *   xxd -i shaders/matmul.spv > matmul_spv.h
 * Then replace the g_matmul_spirv[] array below.
 */

#ifdef COLI_VULKAN

#include "backend_vulkan.h"
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ================================================================
 * Embedded SPIR-V for shaders/matmul.comp
 *
 * This blob is the pre-compiled SPIR-V for the unified matmul shader.
 * The shader handles all four formats (f32, i8, i4, i2) via a push
 * constant `fmt`.  One pipeline is created from this single shader.
 *
 * To regenerate after editing matmul.comp:
 *   glslangValidator -V shaders/matmul.comp -o shaders/matmul.spv
 * Then use xxd or a script to produce the C array.
 *
 * The blob below was generated from the matmul.comp shipped with this
 * file.  If the shader source changes, regenerate this array.
 * ================================================================ */

/*
 * PLACEHOLDER: This must be regenerated from shaders/matmul.comp.
 * Build the SPIR-V with:
 *   glslangValidator -V shaders/matmul.comp -o shaders/matmul.spv
 *
 * Then embed it:
 *   xxd -i shaders/matmul.spv | sed 's/unsigned char/static const uint32_t/;s/\[\]/\[\] COLI_ALIGN(4)/' > matmul_spv.h
 *
 * Or use the generate_spirv.sh helper script.
 *
 * For now, we attempt runtime SPIR-V loading from "shaders/matmul.spv"
 * as a fallback when the embedded blob is empty.
 */
static const uint32_t g_matmul_spirv[] = { 0 }; /* placeholder */
static const size_t   g_matmul_spirv_size = 0;   /* placeholder */

/* ── Vulkan state ─────────────────────────────────────────────── */

static VkInstance               g_instance;
static VkPhysicalDevice         g_phys_device;
static VkDevice                 g_device;
static VkQueue                  g_queue;
static uint32_t                 g_queue_family;
static VkCommandPool            g_cmd_pool;
static VkCommandBuffer          g_cmd_buf;
static VkDescriptorPool         g_desc_pool;
static VkDescriptorSetLayout    g_desc_set_layout;
static VkPipelineLayout         g_pipeline_layout;
static VkPipeline               g_pipeline;
static VkShaderModule           g_shader_module;

/* Memory type indices cached at init. */
static uint32_t g_mem_type_device;       /* DEVICE_LOCAL (may also be HOST_VISIBLE) */
static uint32_t g_mem_type_host;         /* HOST_VISIBLE | HOST_COHERENT */
static int      g_unified_memory;        /* 1 if DEVICE_LOCAL is also HOST_VISIBLE */

static VkPhysicalDeviceMemoryProperties g_mem_props;

static size_t g_tensor_count, g_tensor_bytes;
static int    g_initialised;

/* Scratch buffers for x and y (reused across matmul calls). */
static VkBuffer       g_x_buf;
static VkDeviceMemory g_x_mem;
static size_t         g_x_cap;
static VkBuffer       g_y_buf;
static VkDeviceMemory g_y_mem;
static size_t         g_y_cap;

/* ── ColiVulkanTensor ─────────────────────────────────────────── */

struct ColiVulkanTensor {
    VkBuffer       weights_buf;
    VkDeviceMemory weights_mem;
    VkBuffer       scales_buf;
    VkDeviceMemory scales_mem;
    size_t         weight_bytes;
    int            fmt, I, O;
    int            tracked;
};

/* ── Push constants (must match shader layout) ────────────────── */

typedef struct {
    int S, I, O, rb, fmt;
} PushConstants;

/* ── Helpers ──────────────────────────────────────────────────── */

static size_t row_bytes(int fmt, int I) {
    if (fmt == 0) return (size_t)I * sizeof(float);
    if (fmt == 1) return (size_t)I;
    if (fmt == 2) return (size_t)(I + 1) / 2;
    if (fmt == 3) return (size_t)(I + 3) / 4;
    return 0;
}

static int vk_ok(VkResult r, const char *what) {
    if (r == VK_SUCCESS) return 1;
    fprintf(stderr, "[Vulkan] %s failed: VkResult %d\n", what, (int)r);
    return 0;
}

/* Find a memory type index matching the required properties. */
static int find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags props,
                            uint32_t *out) {
    for (uint32_t i = 0; i < g_mem_props.memoryTypeCount; i++) {
        if ((type_filter & (1u << i)) &&
            (g_mem_props.memoryTypes[i].propertyFlags & props) == props) {
            *out = i;
            return 1;
        }
    }
    return 0;
}

/* Create a VkBuffer + VkDeviceMemory.  Returns 1 on success. */
static int create_buffer(VkDeviceSize size, VkBufferUsageFlags usage,
                         VkMemoryPropertyFlags mem_props_flags,
                         VkBuffer *buf, VkDeviceMemory *mem) {
    VkBufferCreateInfo bi = {0};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size  = size;
    bi.usage = usage;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (!vk_ok(vkCreateBuffer(g_device, &bi, NULL, buf), "buffer create"))
        return 0;

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(g_device, *buf, &req);

    uint32_t mem_idx;
    if (!find_memory_type(req.memoryTypeBits, mem_props_flags, &mem_idx)) {
        fprintf(stderr, "[Vulkan] no suitable memory type\n");
        vkDestroyBuffer(g_device, *buf, NULL);
        *buf = VK_NULL_HANDLE;
        return 0;
    }

    VkMemoryAllocateInfo ai = {0};
    ai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize   = req.size;
    ai.memoryTypeIndex  = mem_idx;
    if (!vk_ok(vkAllocateMemory(g_device, &ai, NULL, mem), "memory alloc")) {
        vkDestroyBuffer(g_device, *buf, NULL);
        *buf = VK_NULL_HANDLE;
        return 0;
    }
    if (!vk_ok(vkBindBufferMemory(g_device, *buf, *mem, 0), "buffer bind")) {
        vkFreeMemory(g_device, *mem, NULL);
        vkDestroyBuffer(g_device, *buf, NULL);
        *buf = VK_NULL_HANDLE;
        *mem = VK_NULL_HANDLE;
        return 0;
    }
    return 1;
}

static void destroy_buffer(VkBuffer *buf, VkDeviceMemory *mem) {
    if (*buf) { vkDestroyBuffer(g_device, *buf, NULL); *buf = VK_NULL_HANDLE; }
    if (*mem) { vkFreeMemory(g_device, *mem, NULL);     *mem = VK_NULL_HANDLE; }
}

/* Upload host data to a device-local buffer.
 * If the device supports unified memory (HOST_VISIBLE | DEVICE_LOCAL) we map
 * and memcpy directly.  Otherwise we stage through a HOST_VISIBLE buffer. */
static int upload_to_buffer(VkBuffer dst, VkDeviceMemory dst_mem,
                            const void *src, size_t size) {
    if (g_unified_memory) {
        /* Direct map — dst_mem is HOST_VISIBLE */
        void *mapped;
        if (!vk_ok(vkMapMemory(g_device, dst_mem, 0, size, 0, &mapped),
                   "map unified"))
            return 0;
        memcpy(mapped, src, size);
        vkUnmapMemory(g_device, dst_mem);
        return 1;
    }

    /* Discrete GPU path: stage → copy */
    VkBuffer stage_buf = VK_NULL_HANDLE;
    VkDeviceMemory stage_mem = VK_NULL_HANDLE;
    if (!create_buffer(size,
                       VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                       &stage_buf, &stage_mem))
        return 0;

    void *mapped;
    if (!vk_ok(vkMapMemory(g_device, stage_mem, 0, size, 0, &mapped),
               "map staging")) {
        destroy_buffer(&stage_buf, &stage_mem);
        return 0;
    }
    memcpy(mapped, src, size);
    vkUnmapMemory(g_device, stage_mem);

    /* Record and submit a copy command. */
    VkCommandBufferBeginInfo begin = {0};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkResetCommandBuffer(g_cmd_buf, 0);
    vkBeginCommandBuffer(g_cmd_buf, &begin);

    VkBufferCopy region = {0};
    region.size = size;
    vkCmdCopyBuffer(g_cmd_buf, stage_buf, dst, 1, &region);

    vkEndCommandBuffer(g_cmd_buf);

    VkSubmitInfo submit = {0};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &g_cmd_buf;
    vkQueueSubmit(g_queue, 1, &submit, VK_NULL_HANDLE);
    vkQueueWaitIdle(g_queue);

    destroy_buffer(&stage_buf, &stage_mem);
    return 1;
}

/* Ensure the scratch x/y buffers are at least `bytes` large.
 * The buffers need STORAGE_BUFFER_BIT (for shader binding) and for
 * host I/O we need HOST_VISIBLE memory so we can map them. */
static int reserve_scratch(VkBuffer *buf, VkDeviceMemory *mem,
                           size_t *cap, size_t bytes,
                           VkBufferUsageFlags extra_usage) {
    if (*cap >= bytes) return 1;
    destroy_buffer(buf, mem);
    *cap = 0;
    VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | extra_usage;
    VkMemoryPropertyFlags props = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    /* Prefer device-local + host-visible on unified memory devices. */
    if (g_unified_memory)
        props |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    if (!create_buffer(bytes, usage, props, buf, mem))
        return 0;
    *cap = bytes;
    return 1;
}

/* Try to load SPIR-V from a file (fallback when the embedded blob is a placeholder). */
static uint32_t *load_spirv_file(const char *path, size_t *out_size) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    if (len <= 0 || (len & 3) != 0) { fclose(f); return NULL; }
    fseek(f, 0, SEEK_SET);
    uint32_t *buf = (uint32_t *)malloc((size_t)len);
    if (!buf) { fclose(f); return NULL; }
    if (fread(buf, 1, (size_t)len, f) != (size_t)len) {
        free(buf);
        fclose(f);
        return NULL;
    }
    fclose(f);
    *out_size = (size_t)len;
    return buf;
}

/* ── Public API ──────────────────────────────────────────────── */

int coli_vulkan_init(void) {
    if (g_initialised) return 1;

    /* ---- Instance ---- */
    VkApplicationInfo app_info = {0};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName   = "colibri";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pEngineName        = "colibri";
    app_info.engineVersion      = VK_MAKE_VERSION(1, 0, 0);
    app_info.apiVersion         = VK_API_VERSION_1_0;

    VkInstanceCreateInfo inst_ci = {0};
    inst_ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    inst_ci.pApplicationInfo = &app_info;
    if (!vk_ok(vkCreateInstance(&inst_ci, NULL, &g_instance), "instance create"))
        return 0;

    /* ---- Physical device (pick first) ---- */
    uint32_t dev_count = 0;
    vkEnumeratePhysicalDevices(g_instance, &dev_count, NULL);
    if (dev_count == 0) {
        fprintf(stderr, "[Vulkan] no GPU device found\n");
        vkDestroyInstance(g_instance, NULL);
        return 0;
    }
    VkPhysicalDevice *devices = (VkPhysicalDevice *)malloc(
        dev_count * sizeof(VkPhysicalDevice));
    vkEnumeratePhysicalDevices(g_instance, &dev_count, devices);
    g_phys_device = devices[0];
    free(devices);

    VkPhysicalDeviceProperties dev_props;
    vkGetPhysicalDeviceProperties(g_phys_device, &dev_props);

    /* ---- Find compute queue family ---- */
    uint32_t qf_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(g_phys_device, &qf_count, NULL);
    VkQueueFamilyProperties *qf_props = (VkQueueFamilyProperties *)malloc(
        qf_count * sizeof(VkQueueFamilyProperties));
    vkGetPhysicalDeviceQueueFamilyProperties(g_phys_device, &qf_count, qf_props);

    int found_queue = 0;
    g_queue_family = 0;
    /* Prefer a compute-only queue family if available. */
    for (uint32_t i = 0; i < qf_count; i++) {
        if ((qf_props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
            !(qf_props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
            g_queue_family = i;
            found_queue = 1;
            break;
        }
    }
    if (!found_queue) {
        for (uint32_t i = 0; i < qf_count; i++) {
            if (qf_props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                g_queue_family = i;
                found_queue = 1;
                break;
            }
        }
    }
    free(qf_props);
    if (!found_queue) {
        fprintf(stderr, "[Vulkan] no compute queue family\n");
        vkDestroyInstance(g_instance, NULL);
        return 0;
    }

    /* ---- Logical device + queue ---- */
    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_ci = {0};
    queue_ci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_ci.queueFamilyIndex = g_queue_family;
    queue_ci.queueCount = 1;
    queue_ci.pQueuePriorities = &queue_priority;

    VkDeviceCreateInfo dev_ci = {0};
    dev_ci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dev_ci.queueCreateInfoCount = 1;
    dev_ci.pQueueCreateInfos = &queue_ci;
    if (!vk_ok(vkCreateDevice(g_phys_device, &dev_ci, NULL, &g_device),
               "device create")) {
        vkDestroyInstance(g_instance, NULL);
        return 0;
    }
    vkGetDeviceQueue(g_device, g_queue_family, 0, &g_queue);

    /* ---- Memory properties ---- */
    vkGetPhysicalDeviceMemoryProperties(g_phys_device, &g_mem_props);

    /* Find preferred memory types.
     * g_mem_type_device: DEVICE_LOCAL (best for weight storage).
     * g_mem_type_host:   HOST_VISIBLE | HOST_COHERENT (for staging / scratch).
     * g_unified_memory:  set if DEVICE_LOCAL is also HOST_VISIBLE. */
    g_unified_memory = 0;
    g_mem_type_device = 0;
    g_mem_type_host = 0;

    /* Check if any memory type is both DEVICE_LOCAL and HOST_VISIBLE. */
    for (uint32_t i = 0; i < g_mem_props.memoryTypeCount; i++) {
        VkMemoryPropertyFlags f = g_mem_props.memoryTypes[i].propertyFlags;
        if ((f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
            (f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
            g_unified_memory = 1;
            g_mem_type_device = i;
            break;
        }
    }
    if (!g_unified_memory) {
        /* Find a plain DEVICE_LOCAL type. */
        for (uint32_t i = 0; i < g_mem_props.memoryTypeCount; i++) {
            if (g_mem_props.memoryTypes[i].propertyFlags &
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
                g_mem_type_device = i;
                break;
            }
        }
    }
    /* Find a HOST_VISIBLE | HOST_COHERENT type. */
    for (uint32_t i = 0; i < g_mem_props.memoryTypeCount; i++) {
        VkMemoryPropertyFlags f = g_mem_props.memoryTypes[i].propertyFlags;
        if ((f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
            (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            g_mem_type_host = i;
            break;
        }
    }

    /* ---- Command pool + buffer ---- */
    VkCommandPoolCreateInfo pool_ci = {0};
    pool_ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_ci.queueFamilyIndex = g_queue_family;
    if (!vk_ok(vkCreateCommandPool(g_device, &pool_ci, NULL, &g_cmd_pool),
               "command pool")) {
        vkDestroyDevice(g_device, NULL);
        vkDestroyInstance(g_instance, NULL);
        return 0;
    }

    VkCommandBufferAllocateInfo cb_ai = {0};
    cb_ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cb_ai.commandPool = g_cmd_pool;
    cb_ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cb_ai.commandBufferCount = 1;
    if (!vk_ok(vkAllocateCommandBuffers(g_device, &cb_ai, &g_cmd_buf),
               "command buffer alloc")) {
        vkDestroyCommandPool(g_device, g_cmd_pool, NULL);
        vkDestroyDevice(g_device, NULL);
        vkDestroyInstance(g_instance, NULL);
        return 0;
    }

    /* ---- Shader module ----
     * Try the embedded SPIR-V first; if it's a placeholder (size 0), fall
     * back to loading shaders/matmul.spv from disk. */
    const uint32_t *spirv_code = g_matmul_spirv;
    size_t spirv_size = g_matmul_spirv_size;
    uint32_t *file_spirv = NULL;

    if (spirv_size == 0) {
        file_spirv = load_spirv_file("shaders/matmul.spv", &spirv_size);
        if (!file_spirv) {
            fprintf(stderr, "[Vulkan] no embedded SPIR-V and shaders/matmul.spv not found.\n"
                    "  Compile with: glslangValidator -V shaders/matmul.comp -o shaders/matmul.spv\n");
            vkDestroyCommandPool(g_device, g_cmd_pool, NULL);
            vkDestroyDevice(g_device, NULL);
            vkDestroyInstance(g_instance, NULL);
            return 0;
        }
        spirv_code = file_spirv;
    }

    VkShaderModuleCreateInfo sm_ci = {0};
    sm_ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    sm_ci.codeSize = spirv_size;
    sm_ci.pCode = spirv_code;
    VkResult sm_r = vkCreateShaderModule(g_device, &sm_ci, NULL, &g_shader_module);
    free(file_spirv); /* safe even if NULL */
    if (!vk_ok(sm_r, "shader module")) {
        vkDestroyCommandPool(g_device, g_cmd_pool, NULL);
        vkDestroyDevice(g_device, NULL);
        vkDestroyInstance(g_instance, NULL);
        return 0;
    }

    /* ---- Descriptor set layout (4 storage buffers) ---- */
    VkDescriptorSetLayoutBinding bindings[4];
    memset(bindings, 0, sizeof(bindings));
    for (int i = 0; i < 4; i++) {
        bindings[i].binding = (uint32_t)i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo dsl_ci = {0};
    dsl_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dsl_ci.bindingCount = 4;
    dsl_ci.pBindings = bindings;
    if (!vk_ok(vkCreateDescriptorSetLayout(g_device, &dsl_ci, NULL,
                                           &g_desc_set_layout),
               "descriptor set layout")) {
        vkDestroyShaderModule(g_device, g_shader_module, NULL);
        vkDestroyCommandPool(g_device, g_cmd_pool, NULL);
        vkDestroyDevice(g_device, NULL);
        vkDestroyInstance(g_instance, NULL);
        return 0;
    }

    /* ---- Pipeline layout (push constants + descriptor set) ---- */
    VkPushConstantRange push_range = {0};
    push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_range.offset = 0;
    push_range.size = sizeof(PushConstants);

    VkPipelineLayoutCreateInfo pl_ci = {0};
    pl_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pl_ci.setLayoutCount = 1;
    pl_ci.pSetLayouts = &g_desc_set_layout;
    pl_ci.pushConstantRangeCount = 1;
    pl_ci.pPushConstantRanges = &push_range;
    if (!vk_ok(vkCreatePipelineLayout(g_device, &pl_ci, NULL, &g_pipeline_layout),
               "pipeline layout")) {
        vkDestroyDescriptorSetLayout(g_device, g_desc_set_layout, NULL);
        vkDestroyShaderModule(g_device, g_shader_module, NULL);
        vkDestroyCommandPool(g_device, g_cmd_pool, NULL);
        vkDestroyDevice(g_device, NULL);
        vkDestroyInstance(g_instance, NULL);
        return 0;
    }

    /* ---- Compute pipeline ---- */
    VkComputePipelineCreateInfo cp_ci = {0};
    cp_ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cp_ci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cp_ci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cp_ci.stage.module = g_shader_module;
    cp_ci.stage.pName  = "main";
    cp_ci.layout = g_pipeline_layout;
    if (!vk_ok(vkCreateComputePipelines(g_device, VK_NULL_HANDLE, 1, &cp_ci,
                                        NULL, &g_pipeline),
               "compute pipeline")) {
        vkDestroyPipelineLayout(g_device, g_pipeline_layout, NULL);
        vkDestroyDescriptorSetLayout(g_device, g_desc_set_layout, NULL);
        vkDestroyShaderModule(g_device, g_shader_module, NULL);
        vkDestroyCommandPool(g_device, g_cmd_pool, NULL);
        vkDestroyDevice(g_device, NULL);
        vkDestroyInstance(g_instance, NULL);
        return 0;
    }

    /* ---- Descriptor pool ----
     * We allocate a fresh descriptor set per matmul call and reset the pool
     * each time, keeping things simple (no caching). */
    VkDescriptorPoolSize pool_size = {0};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = 4;

    VkDescriptorPoolCreateInfo dp_ci = {0};
    dp_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dp_ci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    dp_ci.maxSets = 1;
    dp_ci.poolSizeCount = 1;
    dp_ci.pPoolSizes = &pool_size;
    if (!vk_ok(vkCreateDescriptorPool(g_device, &dp_ci, NULL, &g_desc_pool),
               "descriptor pool")) {
        vkDestroyPipeline(g_device, g_pipeline, NULL);
        vkDestroyPipelineLayout(g_device, g_pipeline_layout, NULL);
        vkDestroyDescriptorSetLayout(g_device, g_desc_set_layout, NULL);
        vkDestroyShaderModule(g_device, g_shader_module, NULL);
        vkDestroyCommandPool(g_device, g_cmd_pool, NULL);
        vkDestroyDevice(g_device, NULL);
        vkDestroyInstance(g_instance, NULL);
        return 0;
    }

    /* ---- Init scratch buffers ---- */
    g_x_buf = VK_NULL_HANDLE; g_x_mem = VK_NULL_HANDLE; g_x_cap = 0;
    g_y_buf = VK_NULL_HANDLE; g_y_mem = VK_NULL_HANDLE; g_y_cap = 0;
    g_tensor_count = 0;
    g_tensor_bytes = 0;

    /* ---- Report ---- */
    size_t total_mem = 0;
    for (uint32_t i = 0; i < g_mem_props.memoryHeapCount; i++) {
        if (g_mem_props.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
            total_mem += g_mem_props.memoryHeaps[i].size;
    }

    fprintf(stderr, "[Vulkan] %s, %.1f GB %s memory, Vulkan %d.%d.%d\n",
            dev_props.deviceName,
            total_mem / 1e9,
            g_unified_memory ? "unified" : "discrete",
            VK_VERSION_MAJOR(dev_props.apiVersion),
            VK_VERSION_MINOR(dev_props.apiVersion),
            VK_VERSION_PATCH(dev_props.apiVersion));

    g_initialised = 1;
    return 1;
}

void coli_vulkan_shutdown(void) {
    if (!g_initialised) return;
    vkDeviceWaitIdle(g_device);

    destroy_buffer(&g_x_buf, &g_x_mem); g_x_cap = 0;
    destroy_buffer(&g_y_buf, &g_y_mem); g_y_cap = 0;

    if (g_desc_pool)       vkDestroyDescriptorPool(g_device, g_desc_pool, NULL);
    if (g_pipeline)        vkDestroyPipeline(g_device, g_pipeline, NULL);
    if (g_pipeline_layout) vkDestroyPipelineLayout(g_device, g_pipeline_layout, NULL);
    if (g_desc_set_layout) vkDestroyDescriptorSetLayout(g_device, g_desc_set_layout, NULL);
    if (g_shader_module)   vkDestroyShaderModule(g_device, g_shader_module, NULL);
    if (g_cmd_pool)        vkDestroyCommandPool(g_device, g_cmd_pool, NULL);
    if (g_device)          vkDestroyDevice(g_device, NULL);
    if (g_instance)        vkDestroyInstance(g_instance, NULL);

    g_desc_pool       = VK_NULL_HANDLE;
    g_pipeline        = VK_NULL_HANDLE;
    g_pipeline_layout = VK_NULL_HANDLE;
    g_desc_set_layout = VK_NULL_HANDLE;
    g_shader_module   = VK_NULL_HANDLE;
    g_cmd_pool        = VK_NULL_HANDLE;
    g_device          = VK_NULL_HANDLE;
    g_instance        = VK_NULL_HANDLE;
    g_tensor_count    = 0;
    g_tensor_bytes    = 0;
    g_initialised     = 0;
}

int coli_vulkan_mem_info(size_t *free_bytes, size_t *total_bytes) {
    if (!g_initialised || !free_bytes || !total_bytes) return 0;

    /* Vulkan 1.0 has no standard query for free memory.
     * VK_EXT_memory_budget provides this but is not universally available.
     * We report the total device-local heap size and estimate free from it
     * minus tracked tensor bytes.  This is approximate but useful. */
    size_t total = 0;
    for (uint32_t i = 0; i < g_mem_props.memoryHeapCount; i++) {
        if (g_mem_props.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
            total += g_mem_props.memoryHeaps[i].size;
    }
    *total_bytes = total;
    *free_bytes  = total > g_tensor_bytes ? total - g_tensor_bytes : 0;
    return 1;
}

void coli_vulkan_stats(size_t *tensor_count, size_t *tensor_bytes) {
    if (tensor_count) *tensor_count = g_tensor_count;
    if (tensor_bytes) *tensor_bytes = g_tensor_bytes;
}

int coli_vulkan_tensor_upload(ColiVulkanTensor **tensor,
                              const void *weights, const float *scales,
                              int fmt, int I, int O) {
    if (!tensor || !weights || !g_initialised || I < 1 || O < 1) return 0;
    size_t rb = row_bytes(fmt, I);
    if (!rb || (fmt && !scales)) return 0;

    /* Already uploaded -- reuse. */
    if (*tensor) {
        ColiVulkanTensor *t = *tensor;
        return t->fmt == fmt && t->I == I && t->O == O;
    }

    ColiVulkanTensor *t = (ColiVulkanTensor *)calloc(1, sizeof(*t));
    if (!t) return 0;
    t->fmt = fmt;
    t->I   = I;
    t->O   = O;
    t->weight_bytes = rb * (size_t)O;

    /* Weight buffer: prefer DEVICE_LOCAL.  On unified memory this is also
     * HOST_VISIBLE so upload_to_buffer will map directly. */
    VkMemoryPropertyFlags weight_props = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    if (g_unified_memory)
        weight_props |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    VkBufferUsageFlags weight_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    if (!g_unified_memory)
        weight_usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    if (!create_buffer(t->weight_bytes, weight_usage, weight_props,
                       &t->weights_buf, &t->weights_mem)) {
        fprintf(stderr, "[Vulkan] failed to create weight buffer (%zu bytes)\n",
                t->weight_bytes);
        free(t);
        return 0;
    }
    if (!upload_to_buffer(t->weights_buf, t->weights_mem, weights,
                          t->weight_bytes)) {
        destroy_buffer(&t->weights_buf, &t->weights_mem);
        free(t);
        return 0;
    }

    /* Scales buffer (only for quantized formats). */
    if (fmt) {
        size_t scale_bytes = (size_t)O * sizeof(float);
        VkBufferUsageFlags scale_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        if (!g_unified_memory)
            scale_usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;

        if (!create_buffer(scale_bytes, scale_usage, weight_props,
                           &t->scales_buf, &t->scales_mem)) {
            fprintf(stderr, "[Vulkan] failed to create scale buffer\n");
            destroy_buffer(&t->weights_buf, &t->weights_mem);
            free(t);
            return 0;
        }
        if (!upload_to_buffer(t->scales_buf, t->scales_mem, scales,
                              scale_bytes)) {
            destroy_buffer(&t->scales_buf, &t->scales_mem);
            destroy_buffer(&t->weights_buf, &t->weights_mem);
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

int coli_vulkan_matmul(ColiVulkanTensor **tensor,
                       float *y, const float *x,
                       const void *weights, const float *scales,
                       int fmt, int S, int I, int O) {
    if (S < 1 || !coli_vulkan_tensor_upload(tensor, weights, scales, fmt, I, O))
        return 0;

    ColiVulkanTensor *t = *tensor;
    if (fmt < 0 || fmt > 3) return 0;

    size_t rb = row_bytes(fmt, I);
    size_t x_bytes = (size_t)S * I * sizeof(float);
    size_t y_bytes = (size_t)S * O * sizeof(float);

    /* Ensure scratch buffers are large enough. */
    if (!reserve_scratch(&g_x_buf, &g_x_mem, &g_x_cap, x_bytes, 0))
        return 0;
    if (!reserve_scratch(&g_y_buf, &g_y_mem, &g_y_cap, y_bytes, 0))
        return 0;

    /* Upload x to scratch buffer. */
    {
        void *mapped;
        if (!vk_ok(vkMapMemory(g_device, g_x_mem, 0, x_bytes, 0, &mapped),
                   "map x"))
            return 0;
        memcpy(mapped, x, x_bytes);
        vkUnmapMemory(g_device, g_x_mem);
    }

    /* Allocate descriptor set. */
    VkDescriptorSet desc_set;
    vkResetDescriptorPool(g_device, g_desc_pool, 0);

    VkDescriptorSetAllocateInfo ds_ai = {0};
    ds_ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ds_ai.descriptorPool = g_desc_pool;
    ds_ai.descriptorSetCount = 1;
    ds_ai.pSetLayouts = &g_desc_set_layout;
    if (!vk_ok(vkAllocateDescriptorSets(g_device, &ds_ai, &desc_set),
               "descriptor set alloc"))
        return 0;

    /* Write descriptors.
     * For fmt==0 (f32), the scales buffer is unused by the shader, but
     * Vulkan requires all bindings to be valid.  We bind a dummy (the y
     * buffer) so there's no validation error. */
    VkDescriptorBufferInfo buf_infos[4];
    memset(buf_infos, 0, sizeof(buf_infos));
    buf_infos[0].buffer = g_x_buf;
    buf_infos[0].offset = 0;
    buf_infos[0].range  = x_bytes;

    buf_infos[1].buffer = t->weights_buf;
    buf_infos[1].offset = 0;
    buf_infos[1].range  = t->weight_bytes;

    /* binding 2: scales (or dummy for f32) */
    if (fmt) {
        buf_infos[2].buffer = t->scales_buf;
        buf_infos[2].offset = 0;
        buf_infos[2].range  = (VkDeviceSize)O * sizeof(float);
    } else {
        buf_infos[2].buffer = g_y_buf;  /* dummy */
        buf_infos[2].offset = 0;
        buf_infos[2].range  = y_bytes;
    }

    buf_infos[3].buffer = g_y_buf;
    buf_infos[3].offset = 0;
    buf_infos[3].range  = y_bytes;

    VkWriteDescriptorSet writes[4];
    memset(writes, 0, sizeof(writes));
    for (int i = 0; i < 4; i++) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = desc_set;
        writes[i].dstBinding = (uint32_t)i;
        writes[i].dstArrayElement = 0;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &buf_infos[i];
    }
    vkUpdateDescriptorSets(g_device, 4, writes, 0, NULL);

    /* Record command buffer. */
    VkCommandBufferBeginInfo begin = {0};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkResetCommandBuffer(g_cmd_buf, 0);
    vkBeginCommandBuffer(g_cmd_buf, &begin);

    vkCmdBindPipeline(g_cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, g_pipeline);
    vkCmdBindDescriptorSets(g_cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
                            g_pipeline_layout, 0, 1, &desc_set, 0, NULL);

    PushConstants pc;
    pc.S   = S;
    pc.I   = I;
    pc.O   = O;
    pc.rb  = (int)rb;
    pc.fmt = fmt;
    vkCmdPushConstants(g_cmd_buf, g_pipeline_layout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(PushConstants), &pc);

    vkCmdDispatch(g_cmd_buf, (uint32_t)O, (uint32_t)S, 1);

    /* Memory barrier: compute writes → host reads. */
    VkMemoryBarrier barrier = {0};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(g_cmd_buf,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT,
                         0, 1, &barrier, 0, NULL, 0, NULL);

    vkEndCommandBuffer(g_cmd_buf);

    /* Submit and wait (synchronous, like Metal's waitUntilCompleted). */
    VkSubmitInfo submit = {0};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &g_cmd_buf;
    if (!vk_ok(vkQueueSubmit(g_queue, 1, &submit, VK_NULL_HANDLE), "matmul submit"))
        return 0;
    vkQueueWaitIdle(g_queue);

    /* Read back y. */
    {
        void *mapped;
        if (!vk_ok(vkMapMemory(g_device, g_y_mem, 0, y_bytes, 0, &mapped),
                   "map y"))
            return 0;
        memcpy(y, mapped, y_bytes);
        vkUnmapMemory(g_device, g_y_mem);
    }

    return 1;
}

void coli_vulkan_tensor_free(ColiVulkanTensor *tensor) {
    if (!tensor) return;
    if (tensor->tracked) {
        size_t bytes = tensor->weight_bytes +
                       (tensor->fmt ? (size_t)tensor->O * sizeof(float) : 0);
        if (g_tensor_count) g_tensor_count--;
        if (g_tensor_bytes >= bytes) g_tensor_bytes -= bytes;
    }
    if (g_device) {
        destroy_buffer(&tensor->weights_buf, &tensor->weights_mem);
        destroy_buffer(&tensor->scales_buf, &tensor->scales_mem);
    }
    free(tensor);
}

size_t coli_vulkan_tensor_bytes(const ColiVulkanTensor *tensor) {
    return tensor ? tensor->weight_bytes +
           (tensor->fmt ? (size_t)tensor->O * sizeof(float) : 0) : 0;
}

#endif /* COLI_VULKAN */
