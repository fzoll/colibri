UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
# --- macOS / Apple Silicon ---
# Apple clang non include il runtime OpenMP: se c'e' libomp di Homebrew lo usa
# (brew install libomp), altrimenti compila single-thread (i pragma omp sono ignorati).
# Niente -march: su arm64 NEON e' baseline (i kernel __ARM_NEON si attivano da soli).
CC      = clang
OMPDIR := $(shell brew --prefix libomp 2>/dev/null)
ifneq ($(OMPDIR),)
OMPC    = -Xclang -fopenmp -I$(OMPDIR)/include
OMPL    = -L$(OMPDIR)/lib -lomp
else
$(warning libomp non trovato: build SINGLE-THREAD. Per il multithread: brew install libomp)
OMPC    =
OMPL    =
endif
CFLAGS  = -O3 $(OMPC) -Wall -Wextra -Wno-unused-parameter -Wno-misleading-indentation -Wno-unused-function
LDFLAGS = -lm $(OMPL)
else
# --- Linux x86-64 (percorso originale, invariato) ---
CC      = gcc
# ARCH=native -> ottimizzato per QUESTA macchina (default, piu' veloce).
# ARCH=x86-64-v3 -> binario PORTABILE su qualsiasi x86-64 moderno con AVX2 (per distribuire).
# ARCH=x86-64 -> massima compatibilita' (niente AVX2: usa il path scalare di fallback).
ARCH   ?= native
CFLAGS  = -O3 -march=$(ARCH) -fopenmp -Wall -Wextra -Wno-unused-parameter -Wno-misleading-indentation -Wno-unused-function
LDFLAGS = -lm -fopenmp
endif

# CUDA=1 adds an opt-in backend for resident tensors. The default build remains
# pure C and keeps the original zero-dependency runtime.
CUDA      ?= 0
CUDA_HOME ?= /usr/local/cuda
NVCC      ?= $(CUDA_HOME)/bin/nvcc
CUDA_ARCH ?= native
NVCCFLAGS ?= -O3 -std=c++17 -arch=$(CUDA_ARCH) -Xcompiler=-Wall,-Wextra
PYTHON    ?= python3
CUDA_OBJ   =
METAL_OBJ  =
VULKAN_OBJ =
TEST_BINS = tests/test_json tests/test_st tests/test_tier
ifeq ($(CUDA),1)
ifeq ($(UNAME_S),Darwin)
$(error CUDA=1 is supported only on Linux)
endif
CFLAGS   += -DCOLI_CUDA
LDFLAGS  += -L$(CUDA_HOME)/lib64 -Wl,-rpath,$(CUDA_HOME)/lib64 -lcudart -lstdc++
CUDA_OBJ  = backend_cuda.o
endif

# METAL=1 adds a Metal GPU backend for Apple Silicon. macOS only.
METAL     ?= 0
ifeq ($(METAL),1)
ifneq ($(UNAME_S),Darwin)
$(error METAL=1 is supported only on macOS)
endif
CFLAGS   += -DCOLI_METAL
LDFLAGS  += -framework Metal -framework Foundation
METAL_OBJ = backend_metal.o
endif

# VULKAN=1 adds a Vulkan compute backend. Works on Linux (discrete/integrated GPU)
# and RPi 5 (VideoCore VII). Requires Vulkan SDK / libvulkan-dev.
VULKAN    ?= 0
ifeq ($(VULKAN),1)
CFLAGS    += -DCOLI_VULKAN
LDFLAGS   += -lvulkan
VULKAN_OBJ = backend_vulkan.o
endif

all: glm

glm: glm.c st.h json.h tok.h tok_unicode.h compat.h $(CUDA_OBJ) $(METAL_OBJ) $(VULKAN_OBJ)
	$(CC) $(CFLAGS) glm.c $(CUDA_OBJ) $(METAL_OBJ) $(VULKAN_OBJ) -o glm $(LDFLAGS)

backend_metal.o: backend_metal.m backend_metal.h
	$(CC) $(CFLAGS) -c backend_metal.m -o $@

backend_vulkan.o: backend_vulkan.c backend_vulkan.h
	$(CC) $(CFLAGS) -c backend_vulkan.c -o $@

backend_cuda.o: backend_cuda.cu backend_cuda.h
	@command -v $(NVCC) >/dev/null 2>&1 || { echo "nvcc not found: set CUDA_HOME or NVCC" >&2; exit 1; }
	$(NVCC) $(NVCCFLAGS) -c backend_cuda.cu -o $@

cuda-test: backend_cuda.cu backend_cuda.h tests/test_backend_cuda.cu
	@command -v $(NVCC) >/dev/null 2>&1 || { echo "nvcc not found: set CUDA_HOME or NVCC" >&2; exit 1; }
	$(NVCC) $(NVCCFLAGS) backend_cuda.cu tests/test_backend_cuda.cu -o backend_cuda_test
	./backend_cuda_test

# Compile GLSL to SPIR-V (requires glslangValidator from Vulkan SDK).
shaders/matmul.spv: shaders/matmul.comp
	@command -v glslangValidator >/dev/null 2>&1 || { echo "glslangValidator not found: install Vulkan SDK" >&2; exit 1; }
	glslangValidator -V shaders/matmul.comp -o shaders/matmul.spv

metal-test: backend_metal.m backend_metal.h tests/test_backend_metal.m
	$(CC) $(CFLAGS) -DCOLI_METAL tests/test_backend_metal.m backend_metal.m -o tests/test_backend_metal $(LDFLAGS) -framework Metal -framework Foundation
	./tests/test_backend_metal

gemma4: gemma4_moe.c st.h json.h tok.h tok_unicode.h compat.h $(CUDA_OBJ) $(METAL_OBJ) $(VULKAN_OBJ)
	$(CC) $(CFLAGS) gemma4_moe.c $(CUDA_OBJ) $(METAL_OBJ) $(VULKAN_OBJ) -o gemma4 $(LDFLAGS)

glm4: glm4_moe.c st.h json.h tok.h tok_unicode.h compat.h $(CUDA_OBJ) $(METAL_OBJ) $(VULKAN_OBJ)
	$(CC) $(CFLAGS) glm4_moe.c $(CUDA_OBJ) $(METAL_OBJ) $(VULKAN_OBJ) -o glm4 $(LDFLAGS)

olmoe: olmoe.c st.h json.h
	$(CC) $(CFLAGS) olmoe.c -o olmoe $(LDFLAGS)

# binario portabile da distribuire su altre macchine x86-64
portable:
	$(MAKE) glm ARCH=x86-64-v3

tests/test_json: tests/test_json.c json.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_st: tests/test_st.c st.h json.h compat.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_tier: tests/test_tier.c tier.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

test-c: $(TEST_BINS)
	@for test in $(TEST_BINS); do ./$$test || exit 1; done

test-python:
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py'

test: test-c test-python

# Local validation: one portable CPU build and dependency-free tests.
check:
	$(MAKE) clean
	$(MAKE) portable
	$(MAKE) test

clean:
	rm -f olmoe glm glm4 gemma4 backend_cuda.o backend_metal.o backend_vulkan.o backend_cuda_test tests/test_backend_metal $(TEST_BINS)
	rm -rf tests/__pycache__

.PHONY: all cuda-test portable test-c test-python test check clean
