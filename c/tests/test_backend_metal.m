/*
 * test_backend_metal.m — unit test for the Metal GPU backend.
 * Verifies matmul correctness for all quantization formats (f32, int8, int4, int2)
 * by comparing Metal results against a scalar CPU reference.
 *
 * Build & run:
 *   clang -O3 -framework Metal -framework Foundation tests/test_backend_metal.m backend_metal.m -o tests/test_backend_metal -lm
 *   ./tests/test_backend_metal
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../backend_metal.h"

static float randf(unsigned *seed) {
    *seed = *seed * 1103515245u + 12345u;
    return (float)(*seed >> 16) / 32768.0f - 1.0f;
}

static void ref_matmul_f32(float *y, const float *x, const float *w,
                           int S, int I, int O) {
    for (int s = 0; s < S; s++)
        for (int o = 0; o < O; o++) {
            float sum = 0;
            for (int i = 0; i < I; i++)
                sum += x[s * I + i] * w[o * I + i];
            y[s * O + o] = sum;
        }
}

static void ref_matmul_i8(float *y, const float *x, const int8_t *w,
                          const float *scales, int S, int I, int O) {
    for (int s = 0; s < S; s++)
        for (int o = 0; o < O; o++) {
            float sum = 0;
            for (int i = 0; i < I; i++)
                sum += x[s * I + i] * (float)w[o * I + i];
            y[s * O + o] = sum * scales[o];
        }
}

static void ref_matmul_i4(float *y, const float *x, const uint8_t *w,
                          const float *scales, int S, int I, int O) {
    int rb = (I + 1) / 2;
    for (int s = 0; s < S; s++)
        for (int o = 0; o < O; o++) {
            float sum = 0;
            const uint8_t *row = w + o * rb;
            for (int i = 0; i < I; i++) {
                uint8_t v = row[i >> 1];
                float wt = (float)(((i & 1) ? (v >> 4) : (v & 0x0F)) - 8);
                sum += x[s * I + i] * wt;
            }
            y[s * O + o] = sum * scales[o];
        }
}

static void ref_matmul_i2(float *y, const float *x, const uint8_t *w,
                          const float *scales, int S, int I, int O) {
    int rb = (I + 3) / 4;
    for (int s = 0; s < S; s++)
        for (int o = 0; o < O; o++) {
            float sum = 0;
            const uint8_t *row = w + o * rb;
            for (int i = 0; i < I; i++) {
                uint8_t v = row[i >> 2];
                float wt = (float)(((v >> ((i & 3) * 2)) & 3) - 2);
                sum += x[s * I + i] * wt;
            }
            y[s * O + o] = sum * scales[o];
        }
}

static int check(const char *name, const float *ref, const float *got, int n, float tol) {
    float max_err = 0;
    for (int i = 0; i < n; i++) {
        float err = fabsf(ref[i] - got[i]);
        if (err > max_err) max_err = err;
    }
    int ok = max_err <= tol;
    printf("  %-12s max_err=%.6f  %s\n", name, max_err, ok ? "OK" : "FAIL");
    return ok;
}

int main(void) {
    printf("Metal backend test\n");

    if (!coli_metal_init()) {
        printf("Metal not available — skipping\n");
        return 0;
    }

    int S = 4, I = 256, O = 128;
    unsigned seed = 42;
    int pass = 1;

    float *x = malloc(S * I * sizeof(float));
    for (int i = 0; i < S * I; i++) x[i] = randf(&seed);

    /* f32 */
    {
        float *w = malloc(O * I * sizeof(float));
        for (int i = 0; i < O * I; i++) w[i] = randf(&seed);
        float *y_ref = calloc(S * O, sizeof(float));
        float *y_gpu = calloc(S * O, sizeof(float));
        ref_matmul_f32(y_ref, x, w, S, I, O);
        ColiMetalTensor *t = NULL;
        if (!coli_metal_matmul(&t, y_gpu, x, w, NULL, 0, S, I, O)) {
            printf("  f32          Metal matmul failed\n"); pass = 0;
        } else {
            if (!check("f32", y_ref, y_gpu, S * O, 1e-3f)) pass = 0;
        }
        coli_metal_tensor_free(t);
        free(w); free(y_ref); free(y_gpu);
    }

    /* int8 */
    {
        int8_t *w = malloc(O * I);
        float *scales = malloc(O * sizeof(float));
        for (int i = 0; i < O * I; i++) w[i] = (int8_t)((int)(randf(&seed) * 64) % 127);
        for (int i = 0; i < O; i++) scales[i] = randf(&seed) * 0.1f + 0.01f;
        float *y_ref = calloc(S * O, sizeof(float));
        float *y_gpu = calloc(S * O, sizeof(float));
        ref_matmul_i8(y_ref, x, w, scales, S, I, O);
        ColiMetalTensor *t = NULL;
        if (!coli_metal_matmul(&t, y_gpu, x, w, scales, 1, S, I, O)) {
            printf("  int8         Metal matmul failed\n"); pass = 0;
        } else {
            if (!check("int8", y_ref, y_gpu, S * O, 1e-2f)) pass = 0;
        }
        coli_metal_tensor_free(t);
        free(w); free(scales); free(y_ref); free(y_gpu);
    }

    /* int4 */
    {
        int rb = (I + 1) / 2;
        uint8_t *w = malloc(O * rb);
        float *scales = malloc(O * sizeof(float));
        for (int i = 0; i < O * rb; i++) w[i] = (uint8_t)(seed = seed * 1103515245u + 12345u) >> 24;
        for (int i = 0; i < O; i++) scales[i] = randf(&seed) * 0.1f + 0.01f;
        float *y_ref = calloc(S * O, sizeof(float));
        float *y_gpu = calloc(S * O, sizeof(float));
        ref_matmul_i4(y_ref, x, w, scales, S, I, O);
        ColiMetalTensor *t = NULL;
        if (!coli_metal_matmul(&t, y_gpu, x, w, scales, 2, S, I, O)) {
            printf("  int4         Metal matmul failed\n"); pass = 0;
        } else {
            if (!check("int4", y_ref, y_gpu, S * O, 1e-2f)) pass = 0;
        }
        coli_metal_tensor_free(t);
        free(w); free(scales); free(y_ref); free(y_gpu);
    }

    /* int2 */
    {
        int rb = (I + 3) / 4;
        uint8_t *w = malloc(O * rb);
        float *scales = malloc(O * sizeof(float));
        for (int i = 0; i < O * rb; i++) w[i] = (uint8_t)(seed = seed * 1103515245u + 12345u) >> 24;
        for (int i = 0; i < O; i++) scales[i] = randf(&seed) * 0.1f + 0.01f;
        float *y_ref = calloc(S * O, sizeof(float));
        float *y_gpu = calloc(S * O, sizeof(float));
        ref_matmul_i2(y_ref, x, w, scales, S, I, O);
        ColiMetalTensor *t = NULL;
        if (!coli_metal_matmul(&t, y_gpu, x, w, scales, 3, S, I, O)) {
            printf("  int2         Metal matmul failed\n"); pass = 0;
        } else {
            if (!check("int2", y_ref, y_gpu, S * O, 1e-2f)) pass = 0;
        }
        coli_metal_tensor_free(t);
        free(w); free(scales); free(y_ref); free(y_gpu);
    }

    free(x);
    coli_metal_shutdown();

    printf("%s\n", pass ? "Metal backend tests: ok" : "Metal backend tests: FAILED");
    return pass ? 0 : 1;
}
