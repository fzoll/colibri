# Colibri — Project Context for Claude Code

## What is this?

Colibri (colibrì) is a pure-C inference engine that runs GLM-5.2 (744B MoE, ~40B active params/token) on consumer hardware by streaming MoE experts from SSD. Created by Vincenzo (JustVugg). Apache 2.0.

This is a **fork** (github.com/fzoll/colibri). Upstream: github.com/JustVugg/colibri

## Architecture

MoE inference with tiered storage:
- **Dense layers** (attention, embeddings, shared experts — ~17B params) stay resident in RAM (~9.9GB at int4)
- **21,504 routed experts** (~370GB) live on SSD, streamed on-demand via LRU cache
- Router (gate network) decides per-token-per-layer which experts to activate

## Key files

| File | Lines | Purpose |
|---|---|---|
| `c/glm.c` | ~2550 | Main engine — forward pass, expert loading, cache, inference loop |
| `c/olmoe.c` | ~390 | Smaller MoE model (OLMoE by AI2) — good for testing |
| `c/backend_cuda.cu` | ~230 | CUDA GPU backend — matmul offload for resident tensors |
| `c/backend_cuda.h` | ~49 | CUDA backend C interface |
| `c/tier.h` | ~31 | Expert swap heuristic — heat-based eviction (NOT plain LRU!) |
| `c/st.h` | ~233 | Storage/tensor I/O |
| `c/compat.h` | ~50 | macOS compatibility shims (posix_fadvise, O_DIRECT) |
| `c/tok.h` | ~278 | Tokenizer |
| `c/json.h` | ~270 | JSON parser |
| `c/openai_server.py` | ~700 | OpenAI-compatible API server |
| `c/resource_plan.py` | ~260 | Memory/storage planning tool |

## Platform support (already present in code)

- **Linux x86-64**: AVX2 SIMD kernels, optional AVX-512 VNNI. Primary platform.
- **macOS / Apple Silicon**: ARM NEON kernels already implemented! `compat.h` handles macOS differences (F_RDADVISE instead of posix_fadvise, F_NOCACHE instead of O_DIRECT). Makefile has full Darwin support with Homebrew libomp.
- **CUDA**: Opt-in backend (`CUDA=1`), Linux only. Uploads resident tensors to GPU VRAM for accelerated matmul.

## SIMD kernels in glm.c

Three quantized dot-product kernels, each with AVX2, AVX-512, and NEON variants:
- **int8 dot** (~line 211): Q8 weight × f32 activation
- **int4 dot** (~line 231): Packed nibble weight × f32 activation
- **int2 dot** (~line 266): 2-bit weight × f32 activation
- **integer dot (idot)** (~line 299): Pure integer Q8×Q8, AVX-512 VNNI / AVX2 / scalar

All NEON paths are already implemented and tested.

## Expert cache system (tier.h)

NOT plain LRU — uses a **heat-based swap heuristic**:
- `heat[]` array tracks per-expert activation frequency
- `tier_pick_swap()` finds coldest pinned expert and hottest non-resident expert
- Swap only if gain exceeds 25% margin + fixed offset (prevents ping-pong)
- `tier_decay()` halves all heat values (exponential decay)

## Build

```sh
# macOS (Apple Silicon) — CPU only
cd c
brew install libomp    # optional, for OpenMP multi-threading
make glm               # builds with NEON, no AVX2 needed

# macOS (Apple Silicon) — with Metal GPU
make glm METAL=1       # adds Metal compute shaders for matmul

# Linux x86
make glm               # AVX2 auto-detected

# CUDA (Linux only)
make glm CUDA=1

# Tests (no model needed)
make test
```

## Running with Metal

```sh
# Dense layers on GPU (default when Metal enabled)
COLI_METAL=1 ./glm /path/to/model

# Dense + hot experts on GPU (specify GB budget)
COLI_METAL=1 METAL_EXPERT_GB=10 ./glm /path/to/model

# Dense on CPU only (experts still eligible)
COLI_METAL=1 METAL_DENSE=0 ./glm /path/to/model
```

## Our goals (what this fork aims to add)

### 1. Metal GPU backend (priority: HIGH)

**Header already created: `c/backend_metal.h`** — mirrors CUDA interface, simplified (single device, no device param needed on Apple Silicon).

#### CUDA integration analysis (reference for Metal port)

How CUDA integrates with glm.c — Metal must follow the same pattern:

1. **Compile flag**: `COLI_CUDA` define gates all GPU code. Metal uses `COLI_METAL`.
2. **QT struct** (line ~69): each tensor has `ColiCudaTensor *cuda` pointer + `cuda_eligible`, `cuda_failed`, `cuda_device` fields. Metal needs equivalent `ColiMetalTensor *metal` + `metal_eligible`, `metal_failed`.
3. **Dispatch** (`matmul_qt`, line ~456): if GPU enabled AND tensor is `cuda_eligible` AND not `cuda_failed` AND not inside OpenMP parallel region → try GPU matmul, fallback to CPU on failure.
4. **Init** (line ~2416): env var `COLI_CUDA=1` triggers `coli_cuda_init()`. Metal: `COLI_METAL=1` triggers `coli_metal_init()`.
5. **Tensor lifecycle**: resident (dense) tensors uploaded at model load, persist forever. Expert tensors are NOT cuda_eligible — they use CPU path (streaming experts are reused slots).
6. **Config env vars**: `CUDA_DENSE=1` (offload dense layers), `CUDA_EXPERT_GB=N` (pin N GB of experts to GPU). Metal equivalents: `METAL_DENSE`, `METAL_EXPERT_GB`.
7. **Dense placement** (line ~2226): round-robin across devices, checks VRAM capacity. Metal: single device, simpler.
8. **Expert GPU offload** (line ~1827): hot experts can be uploaded to GPU based on heat. Tracks bytes per expert slot on GPU for budget management.

#### Key Apple Silicon advantage
CUDA path copies weights Host→Device (`cudaMemcpy`). Metal can wrap existing unified memory pointers with `MTLBuffer(bytesNoCopy:)` — **zero-copy upload**, making expert GPU offload nearly free. This is the main performance win over the CUDA approach.

#### Files — status:
- `c/backend_metal.h` — **DONE**: C interface, mirrors CUDA (simplified: single device)
- `c/backend_metal.m` — **DONE**: Full Objective-C implementation with embedded MSL shaders. Zero-copy via `newBufferWithBytesNoCopy:`. Compiles clean.
- Makefile `METAL=1` — **DONE**: compiles `.m`, links Metal+Foundation frameworks
- `make glm METAL=1` — **COMPILES** (tested on Apple Silicon)
- `make backend_metal.o` — **COMPILES** standalone

#### glm.c integration — DONE
All `#ifdef COLI_METAL` blocks added, mirroring CUDA integration points:

1. **QT struct**: `ColiMetalTensor *metal; int metal_eligible, metal_failed;`
2. **matmul_qt()**: Metal dispatch path before CPU fallback
3. **main() init**: `COLI_METAL=1` env var → `coli_metal_init()`, `METAL_DENSE` (default 1), `METAL_EXPERT_GB`
4. **Dense placement**: all dense tensors marked metal_eligible when `METAL_DENSE=1`
5. **Expert GPU pin/repin**: Metal expert upload/tracking/reset in pin_load and repin
6. **Stats**: `metal_stats_print()` at all reporting points
7. **Error fallback**: CPU-only binary warns if Metal env vars set without `make METAL=1`

Build verified:
- `make glm` (CPU-only) — compiles, tests pass
- `make glm METAL=1` — compiles, tests pass
- Both produce arm64 binaries

#### MSL shader design:
Shaders are embedded as a C string in `backend_metal.m` (compiled at runtime via `newLibraryWithSource:`). Four kernels: `matmul_f32`, `quant_matmul_i8`, `quant_matmul_i4`, `quant_matmul_i2`. Each uses 256-thread threadgroup reduction, same algorithm as CUDA's `quant_matmul`.

### 2. Future ideas (not yet prioritized)
- **Speculative SSD prefetch**: router scores are cheap to compute (RAM-resident); use them to async-prefetch likely experts from SSD before they're needed, overlapping I/O with compute
- **Session persistence**: save heat[] profile to disk, warm-start expert cache on next launch

### 3. Upstream preparation
- Keep changes modular and backward-compatible
- Default behavior unchanged (heat-based swap stays default)
- New backends are opt-in (like CUDA=1)
- Goal: submit as PR(s) to upstream JustVugg/colibri

## Testing

```sh
# Unit tests (no model, no GPU)
make test

# OLMoE — smaller MoE model for integration testing
make olmoe

# CI checks (portable build + all tests)
make check
```

## GLM-4.5 engine (glm4_moe.c) — IN PROGRESS

Target: GLM-4.5-Air (106B/12B active, `Glm4MoeForCausalLM`).
Same Z.ai family as GLM-5.2 but simpler architecture — no MLA, no DSA.

**Status**: scaffold file created with full architecture documentation in comments.
Compiles as `make glm4`. Needs shared code from glm.c copied in.

**Key differences from glm.c (GLM-5.2)**:
- Attention: standard GQA (Q/K/V/O linear projections, no LoRA decomposition)
- KV cache: full K+V per-head (not compressed latent)
- No DSA indexer
- partial_rotary_factor=0.5 (half of head_dim gets RoPE)
- attention_bias=True (GLM-5.2 has False)
- Router: identical sigmoid + noaux_tc

**What to reuse from glm.c** (copy verbatim):
- All matmul/SIMD code (matmul, matmul_qt, dot kernels)
- Quantization (pack_int4, pack_int2, quantize_rows)
- RMSNorm, softmax, SiLU
- Expert loading, disk streaming (expert_load, st.h)
- Tier/heat system (tier.h)
- Tokenizer (tok.h) — same tokenizer as GLM-5.2
- Sampling (temperature, nucleus, top-k)
- Metal/CUDA backend integration
- OpenAI server (openai_server.py)
- PIN/hot-store, cap_for_ram, mem_available

**What to write new**:
- `attention()`: GQA with partial RoPE (~60 lines vs ~200 in glm.c)
- `load_cfg()`: map Glm4Moe config fields
- `model_init()`: load Q/K/V/O weights (not q_a/q_b/kv_a/kv_b)
- KV cache allocation (full K+V, not compressed)
- `rope_half()`: apply RoPE to only first half of head_dim

**Weight naming**: almost identical to GLM-5.2 except attention projections:
- `self_attn.q_proj.weight/.bias` (was `self_attn.q_a_proj` + `q_b_proj`)
- `self_attn.k_proj.weight/.bias` (was `self_attn.kv_a_proj_with_mqa`)
- `self_attn.v_proj.weight/.bias` (was part of `kv_b_proj`)
- MoE weights: identical naming

**Testing**:
- Tiny bench model generated: `c/glm4_bench_tiny/` (408M params, 1.5GB, Glm4MoeForCausalLM)
- Generator: `c/tools/make_glm4_bench_model.py`
- Reference tokens in `c/glm4_bench_tiny/ref_glm.json` for validation
- GLM-4.5-Air real model: ~60GB int4 download (user has 40GB free — needs disk space)

**Bench model weight name patterns** (verified from safetensors):
- `self_attn.q_proj.weight` [H*hd, D] + `.bias` [H*hd]
- `self_attn.k_proj.weight` [Hkv*hd, D] + `.bias` [Hkv*hd]
- `self_attn.v_proj.weight` [Hkv*hd, D] + `.bias` [Hkv*hd]
- `self_attn.o_proj.weight` [D, H*hd] — NO bias
- `mlp.gate.weight` [n_experts, D] + `.e_score_correction_bias` [n_experts]
- Expert/shared/dense MLP: identical to GLM-5.2
- Final norm: `model.norm.weight`

## Gemma 4 engine (gemma4_moe.c) — IN PROGRESS

Target: Gemma 4 26B-A4B (google/gemma-4-26B-A4B-it, `Gemma4ForConditionalGeneration`).
Text-only inference, vision tower skipped.

**Status**: scaffold file created with full architecture documentation.
Compiles as `make gemma4`. Needs full implementation.

**Why Gemma 4:**
- 26B total / 4B active — small enough for RPi 5 (16GB RAM) with disk streaming
- ~13GB int4 on disk — fits easily
- Real model with real answers — not random bench weights
- Proves Metal backend + disk streaming on a non-GLM architecture

**Key architectural challenges (vs GLM engines):**
1. **Fused expert weights**: `experts.gate_up_proj` is [n_experts, 2*I, D] — one tensor per layer, not per-expert files. Need pread slice to extract single expert.
2. **Hybrid attention**: 5× sliding (window=1024, head_dim=256, 8 kv_heads) + 1× global (full context, head_dim=512, 2 kv_heads). Different params per layer type.
3. **K=V (attention_k_eq_v)**: V shares weights with K — only load k_proj.
4. **QK-Norm**: RMSNorm on Q and K per-head before computing attention scores.
5. **Custom router**: `sigmoid(x @ proj * scale) * per_expert_scale` — different from GLM's sigmoid+bias.
6. **GELU activation**: `gelu_pytorch_tanh(x)` instead of SiLU.
7. **Layer scalar**: per-layer learned residual scaling factor.
8. **Logit softcapping**: `cap * tanh(logits / cap)` with cap=30.0.
9. **tie_word_embeddings**: embed = lm_head (no separate lm_head tensor).
10. **Extra layernorms**: pre/post feedforward + shared expert variants.
11. **Weight prefix**: `model.language_model.layers.N.` not `model.layers.N.`

**What to reuse from glm4_moe.c / glm.c:**
- All SIMD matmul code, quantization, st.h, tier.h, compat.h
- Metal/CUDA backend integration
- Expert cache framework (but fused weight loading is different)
- Sampling, generation loop structure
- Memory management

## Remaining work

1. **Metal backend unit test**: write `tests/test_backend_metal.m` — synthetic matmul correctness test (no model needed), similar to `tests/test_backend_cuda.cu`
2. **End-to-end test with model weights**: download GLM-5.2 weights (~370GB) or OLMoE weights, run inference with `COLI_METAL=1`, verify output matches CPU-only path
3. **Performance benchmarking**: measure tok/s with and without Metal, document speedup
4. **MSL shader optimization**: current shaders are a direct port from CUDA — could benefit from Metal-specific optimizations (simdgroup operations, half-precision accumulation)
5. **Upstream PR preparation**: clean up, test on multiple macOS versions, write PR description

## Code style

- Pure C (C11), zero external dependencies for CPU path
- Italian comments throughout (author is Italian)
- Keep it simple — single-file engine philosophy
- CONTRIBUTING.md: focused changes, preserve dependency-free CPU path
- Benchmark reports need: commit, commands, hardware, storage, warm-up, run count, median throughput

## Hardware context

Developer's machine: MacBook Pro, Apple Silicon, 48GB unified memory.
Target use case: run GLM-5.2 (744B) locally on this machine via expert streaming + Metal GPU acceleration.
With 48GB: ~30-35GB available for expert cache = ~600-800 experts in RAM.
