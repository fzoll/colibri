# colibrì 🐦

**Tiny engine, immense model.** Run **GLM-5.2 (744B-parameter MoE)** on a consumer machine with ~25 GB of RAM — in pure C, with zero dependencies, by streaming experts from disk.

```
$ ./coli chat
  🐦 colibrì v1.0 — GLM-5.2 · 744B MoE · int4 · streaming CPU
  ✓ pronto in 32s · residente 9.9 GB
  › ciao!
  ◆ Ciao! 😊 Come posso aiutarti oggi?
```

## The idea

A 744B Mixture-of-Experts model activates only ~40B parameters per token — and only ~11 GB of those change from token to token (the routed experts). So:

- the **dense part** (attention, shared experts, embeddings — ~17B params) stays **resident in RAM at int4** (~9.9 GB);
- the **21,504 routed experts** (75 MoE layers × 256 experts + the MTP head, ~19 MB each at int4) live **on disk** (~370 GB) and are **streamed on demand**, with a per-layer LRU cache, an optional pinned hot-store, and the OS page cache as a free L2.

The engine is a single C file (`c/glm.c`, ~1,300 lines) plus small headers. No BLAS, no Python at runtime, no GPU.

## What's implemented

- **Faithful GLM-5.2 (`glm_moe_dsa`) forward** — validated token-exact against a `transformers` oracle (teacher-forcing 32/32, greedy 20/20 on a tiny-random model with the real architecture).
- **MLA attention** (q/kv-LoRA, interleaved partial RoPE) with **compressed KV-cache**: 576 floats/token instead of 32,768 (57× smaller — GLM-5.2 has 64 heads and no GQA).
- **DeepSeek-V3-style sigmoid router** (noaux_tc, routed_scaling_factor), shared expert, first-3-dense layers.
- **Native MTP speculative decoding** — GLM-5.2's own multi-token-prediction head (layer 78) drafts tokens that the main model verifies in one batched forward. Measured **2.00 tokens/forward (100% acceptance)** on structured text. Lossless: output identical to greedy.
- **Quantization kernels**: int8 / packed int4 / packed int2, per-row scales, AVX2, dequant-on-use. Packing validated bit-identical to the int8 container.
- **Batch-union MoE**: in prefill (and MTP verification), each unique expert of the batch is read once and applied to every position that routes to it.
- **Byte-level BPE tokenizer in C** (GPT-2-style with Unicode-property regex, 320k merges).
- **RAM safety**: the expert cache is auto-sized from `MemAvailable` at startup — an honest peak projection (working set, KV, MTP row, reconstruction buffers) so the kernel OOM-killer never fires.
- **Offline FP8→int4 converter** (`c/convert_fp8_to_int4.py`): downloads one shard at a time (~5 GB), dequants (128×128 block scales), requantizes to the engine's container, deletes the shard — the 756 GB FP8 checkpoint never needs to exist on disk at once. Resumable.

## Honest numbers (WSL2, 12 cores, 25 GB RAM, NVMe via VHDX)

| metric | value |
|---|---|
| model on disk (int4 container) | ~370 GB |
| resident RAM (dense, int4) | 9.9 GB |
| load time | ~30 s |
| peak RSS during chat | ~20 GB (auto-capped) |
| cold decode cost | ~11 GB disk reads/token (75 layers × 8 experts) |
| disk ceiling (VHDX random) | ~1 GB/s → ~0.05–0.1 tok/s cold |
| MTP speculation | 2.0 tok/forward measured |

This is not fast. It is a 744B frontier-class model **answering correctly on a machine that costs less than one H100 fan**. Warm cache, pinned hot experts and MTP push the useful-response latency down considerably; the physics of the disk does the rest.

## Quick start

```bash
cd c
./setup.sh                      # checks gcc/OpenMP, builds, self-tests

# convert the model (resumable, needs ~400 GB free on a real ext4/NVMe path):
./coli convert                  # from zai-org/GLM-5.2-FP8

# chat (RAM budget and expert cache size itself automatically):
COLI_MODEL=/path/to/glm52_i4 ./coli chat
```

Useful knobs (env or flags): `--topp 0.7` adaptive expert top-p (30–40% less disk), `--ngen N` max tokens, `STATS=f`/`PIN=f PIN_GB=g` record expert usage then pin the hottest in RAM, `THINK=1` enable GLM-5.2's reasoning block, `DRAFT=n` MTP draft depth, `TF=1` teacher-forcing validation.

## Got a better machine? Try it — here's what to expect

colibrì was built on deliberately humble hardware (12 cores, 25 GB RAM, NVMe behind a WSL2 VHDX that caps random reads at ~1 GB/s). **Every one of those constraints is a knob your machine can turn up.** The engine needs: Linux (or WSL2), gcc with OpenMP, AVX2, ≥16 GB RAM, and the ~370 GB int4 model on a local NVMe (ext4 — never a network/9p mount).

**How to test it, in order:**

```bash
cd c && ./setup.sh                 # build + architecture self-test (expects 32/32)

# 1) measure YOUR disk the way the engine uses it (parallel 19 MB random reads):
gcc -O2 -fopenmp iobench.c -o iobench
./iobench /path/to/glm52_i4/out-00069.safetensors 19 64 8 0   # buffered, 8 threads
./iobench /path/to/glm52_i4/out-00069.safetensors 19 64 8 1   # O_DIRECT

# 2) chat; watch the per-turn stats line (tok/s, expert hit-rate, RSS):
COLI_MODEL=/path/to/glm52_i4 ./coli chat

# 3) record expert usage, then pin the hottest experts in your spare RAM:
STATS=stats.txt ./coli chat
PIN=stats.txt PIN_GB=20 ./coli chat        # scale PIN_GB to your free RAM

# 4) quality benchmarks (MMLU/HellaSwag/ARC):
./coli bench
```

**Back-of-envelope predictions** (decode is disk-bound: a cold token costs ~11.4 GB of expert reads; MTP speculation roughly halves the effective cost; RAM turns cold reads into free cache hits):

| machine | expected |
|---|---|
| this dev box (WSL2 VHDX, ~1 GB/s, 25 GB RAM) | ~0.05–0.1 tok/s cold — proven baseline |
| native Linux, PCIe4 NVMe (~3–5 GB/s random), 32 GB | ~0.5–1 tok/s |
| PCIe5 NVMe or 2×NVMe RAID0 (~8–12 GB/s), 64 GB (PIN ~40 GB of hot experts) | ~2–4 tok/s |
| 128–256 GB RAM workstation (hot expert set mostly cached) | ~5–15 tok/s, matmul-bound — interactive |

These are estimates, not measurements — if you run colibrì on serious hardware, **please open an issue with your numbers**: real datapoints from better machines are exactly what this project needs next.

## Supporting the project

colibrì is a one-person project, written and tested entirely on a 12-core laptop with 25 GB of RAM — the numbers above are the ceiling of what I can measure at home. If this project is useful or interesting to you and you'd like to support its development (better test hardware translates *directly* into a faster engine for everyone: real NVMe scaling data, bigger pinned caches, int2/int3 quality sweeps on real benchmarks), you can:

- ⭐ star the repo and share it;
- 🐛 open issues with benchmark numbers from your hardware;
- 💬 reach out via GitHub issues if you'd like to sponsor development or donate hardware.

Every contribution, from a datapoint to a disk, moves the ceiling.

## Repo layout

```
c/glm.c          the engine (GLM-5.2 forward, streaming MoE, MTP, serve mode)
c/st.h           safetensors reader: pread + fadvise, no mmap (RSS stays flat)
c/tok.h          byte-level BPE tokenizer in C
c/coli           CLI: chat / run / bench / convert / info
c/iobench.c      parallel disk microbenchmark (measures what the engine feels)
c/convert_fp8_to_int4.py   disk-safe FP8 → int4 converter
c/make_glm_oracle.py       tiny-random oracle generator for validation
c/olmoe.c        stage-A engine (OLMoE), first validation target
```

## Why "colibrì"

The hummingbird weighs a few grams, hovers in place, and visits a thousand flowers a day. This engine keeps a 744-billion-parameter giant alive on hummingbird rations: 25 GB of RAM, twelve CPU cores, and a lot of disk patience.

## License

Apache 2.0. GLM-5.2 weights are released by Z.ai under MIT.
