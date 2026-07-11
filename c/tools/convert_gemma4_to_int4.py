"""Convert Gemma 4 26B-A4B BF16 safetensors to Colibri int4 container.

Reads the BF16 model shard by shard, quantizes weight matrices to int4
(same math as the C engine), and writes per-expert safetensors files.

Key difference from GLM converter: Gemma 4 has FUSED expert weights
(one [n_experts, 2*I, D] tensor per layer). This script splits them
into per-expert files compatible with Colibri's pread-based loading.

Usage:
  # from local BF16 safetensors directory:
  python3 tools/convert_gemma4_to_int4.py --indir gemma4_26b_real --outdir gemma4_26b_i4

  # shard-by-shard download+convert (disk-safe):
  python3 tools/convert_gemma4_to_int4.py --repo google/gemma-4-26B-A4B-it --outdir gemma4_26b_i4
"""
import os, sys, json, argparse, struct
import numpy as np

# ---- quantization (identical to C engine / GLM converter) ----
def quant_int4(w, bits=4):
    O, I = w.shape
    qmax = (1 << (bits - 1)) - 1
    amax = np.abs(w).max(axis=1, keepdims=True)
    s = np.maximum(amax / qmax, 1e-8)
    q = np.clip(np.rint(w / s), -8, qmax).astype(np.int32)
    rb = (I + 1) // 2
    out = np.zeros((O, rb), np.uint8)
    v0 = (q[:, 0::2] + 8).astype(np.uint8)
    out[:, :v0.shape[1]] = v0
    if I > 1:
        v1 = (q[:, 1::2] + 8).astype(np.uint8)
        out[:, :v1.shape[1]] |= (v1 << 4)
    return out.reshape(-1), s[:, 0].astype(np.float32)

def quant_int8(w, bits=8):
    qmax = (1 << (bits - 1)) - 1
    amax = np.abs(w).max(axis=1, keepdims=True)
    s = np.maximum(amax / qmax, 1e-8)
    q = np.clip(np.rint(w / s), -qmax - 1, qmax).astype(np.int8)
    return q.reshape(-1).view(np.uint8).copy(), s[:, 0].astype(np.float32)

# ---- safetensors writer ----
def st_write(path, tensors):
    """Write a dict of {name: (data_bytes, dtype_str, shape)} to safetensors."""
    header = {}
    offset = 0
    for name, (data, dtype, shape) in tensors.items():
        nb = len(data)
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + nb]}
        offset += nb
    hdr_json = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hdr_json)))
        f.write(hdr_json)
        for name, (data, dtype, shape) in tensors.items():
            f.write(data)

# ---- BF16 read helper ----
def read_bf16_as_f32(raw_bytes, numel):
    bf16 = np.frombuffer(raw_bytes, dtype=np.uint16)
    f32_bits = bf16.astype(np.uint32) << 16
    return f32_bits.view(np.float32).copy()

_current_shard_path = None

def read_tensor_f32(sf, name):
    """Read a tensor as float32. Handles BF16 via manual bit conversion (no torch needed)."""
    meta = sf.metadata()  # not useful, use get_slice
    sl = sf.get_slice(name)
    shape = list(sl.get_shape())
    dt = str(sl.get_dtype())
    numel = 1
    for s in shape:
        numel *= s
    if "F32" in dt:
        return np.array(sl[:], dtype=np.float32).reshape(shape), shape
    if "F16" in dt and "BF" not in dt:
        return np.array(sl[:], dtype=np.float16).astype(np.float32).reshape(shape), shape
    # BF16: safetensors can't load as numpy. Read raw bytes from file.
    # Parse header to find offset, then pread.
    return _read_bf16_raw(sf, name, shape, numel), shape

def _read_bf16_raw(sf, name, shape, numel):
    """Read BF16 tensor by parsing safetensors file directly."""
    # sf doesn't expose file path directly, but we cached it
    global _current_shard_path
    path = _current_shard_path
    if path is None:
        raise RuntimeError(f"Cannot read BF16 tensor {name} — file path not available")
    import struct as st_mod
    with open(path, 'rb') as f:
        hdr_len = st_mod.unpack('<Q', f.read(8))[0]
        hdr = json.loads(f.read(hdr_len))
        data_start = 8 + hdr_len
        meta = hdr[name]
        off_start, off_end = meta['data_offsets']
        f.seek(data_start + off_start)
        raw = f.read(off_end - off_start)
    bf16 = np.frombuffer(raw, dtype=np.uint16)[:numel]
    f32_bits = bf16.astype(np.uint32) << 16
    return f32_bits.view(np.float32).reshape(shape).copy()

# ---- tensor classification ----
def classify(name):
    """Classify a tensor for quantization treatment."""
    # Skip vision tower entirely
    if "vision_tower" in name or "embed_vision" in name:
        return "skip"
    # PLE embeddings — keep F32 (large lookup table, not matmul)
    if "embed_tokens_per_layer" in name or "per_layer_model_projection" in name:
        return "f32"
    if "per_layer_projection_norm" in name or "per_layer_input_gate" in name:
        return "f32"
    if "post_per_layer_input_norm" in name or "per_layer_projection" in name:
        return "f32"
    # Router weights — keep F32 (small, sensitive)
    if "router.proj.weight" in name or "router.scale" in name or "router.per_expert_scale" in name:
        return "f32"
    # Norms, scalars — keep F32
    if "norm.weight" in name or "layer_scalar" in name:
        return "f32"
    # Fused expert weights — split and quantize per-expert
    if "experts.gate_up_proj" in name:
        return "expert_gate_up"
    if "experts.down_proj" in name:
        return "expert_down"
    # Embedding (= lm_head for tie_word_embeddings)
    if "embed_tokens.weight" in name:
        return "io"
    # Dense MLP / attention weights — quantize
    if name.endswith(".weight"):
        return "q"
    # Everything else (bias etc) — F32
    return "f32"

def main():
    parser = argparse.ArgumentParser(description="Convert Gemma 4 BF16 to Colibri int4")
    parser.add_argument("--indir", required=True, help="Input directory with BF16 safetensors")
    parser.add_argument("--outdir", required=True, help="Output directory for int4 container")
    parser.add_argument("--ebits", type=int, default=4, help="Expert quantization bits (default 4)")
    parser.add_argument("--dbits", type=int, default=4, help="Dense quantization bits (default 4)")
    parser.add_argument("--io-bits", type=int, default=8, help="Embed/lm_head bits (default 8)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Copy config
    for f in ["config.json", "generation_config.json", "tokenizer.json",
              "tokenizer_config.json", "tokenizer.model", "special_tokens_map.json"]:
        src = os.path.join(args.indir, f)
        if os.path.exists(src):
            import shutil
            shutil.copy2(src, os.path.join(args.outdir, f))
            print(f"  copied {f}")

    # Find safetensors shards
    from safetensors import safe_open
    shards = sorted([f for f in os.listdir(args.indir) if f.endswith(".safetensors")])
    print(f"Found {len(shards)} shard(s)")

    out_tensors = {}
    shard_idx = 0
    total_in = 0
    total_out = 0

    def flush_shard():
        nonlocal out_tensors, shard_idx, total_out
        if not out_tensors:
            return
        shard_idx += 1
        path = os.path.join(args.outdir, f"model-{shard_idx:05d}-of-99999.safetensors")
        st_write(path, out_tensors)
        sz = os.path.getsize(path)
        total_out += sz
        print(f"  wrote {path} ({sz/1e6:.1f} MB, {len(out_tensors)} tensors)")
        out_tensors = {}

    for shard_name in shards:
        shard_path = os.path.join(args.indir, shard_name)
        print(f"\nProcessing {shard_name}...")
        sf = safe_open(shard_path, framework="numpy")
        global _current_shard_path
        _current_shard_path = shard_path

        for name in sorted(sf.keys()):
            cls = classify(name)
            if cls == "skip":
                continue

            w, shape = read_tensor_f32(sf, name)
            total_in += w.nbytes

            if cls == "f32":
                out_tensors[name] = (w.tobytes(), "F32", shape)

            elif cls == "io":
                bits = args.io_bits
                if w.ndim == 2:
                    if bits >= 5:
                        qb, sc = quant_int8(w, bits)
                    else:
                        qb, sc = quant_int4(w, bits)
                    out_tensors[name] = (qb.tobytes(), "U8", [shape[0], len(qb) // shape[0]])
                    out_tensors[name + ".qs"] = (sc.tobytes(), "F32", [shape[0]])
                else:
                    out_tensors[name] = (w.tobytes(), "F32", shape)

            elif cls == "q":
                bits = args.dbits
                if w.ndim == 2:
                    if bits >= 5:
                        qb, sc = quant_int8(w, bits)
                    else:
                        qb, sc = quant_int4(w, bits)
                    out_tensors[name] = (qb.tobytes(), "U8", [shape[0], len(qb) // shape[0]])
                    out_tensors[name + ".qs"] = (sc.tobytes(), "F32", [shape[0]])
                else:
                    out_tensors[name] = (w.tobytes(), "F32", shape)

            elif cls == "expert_gate_up":
                # Fused [n_experts, 2*moe_inter, hidden] → per-expert split + quantize
                n_experts = shape[0]
                fused_dim = shape[1]  # 2 * moe_inter
                hidden = shape[2]
                moe_inter = fused_dim // 2
                print(f"  splitting {name}: [{n_experts}, {fused_dim}, {hidden}] → {n_experts} experts")

                # Extract layer number from name
                # model.language_model.layers.N.experts.gate_up_proj
                parts = name.split(".")
                layer_num = None
                for i, p in enumerate(parts):
                    if p == "layers" and i + 1 < len(parts):
                        layer_num = parts[i + 1]
                        break

                for e in range(n_experts):
                    expert_w = w[e]  # [2*moe_inter, hidden]
                    gate_w = expert_w[:moe_inter]  # [moe_inter, hidden]
                    up_w = expert_w[moe_inter:]    # [moe_inter, hidden]

                    bits = args.ebits
                    prefix = f"model.language_model.layers.{layer_num}.mlp.experts.{e}"

                    if bits >= 5:
                        gq, gs = quant_int8(gate_w, bits)
                        uq, us = quant_int8(up_w, bits)
                    else:
                        gq, gs = quant_int4(gate_w, bits)
                        uq, us = quant_int4(up_w, bits)

                    gname = f"{prefix}.gate_proj.weight"
                    uname = f"{prefix}.up_proj.weight"
                    out_tensors[gname] = (gq.tobytes(), "U8", [moe_inter, len(gq) // moe_inter])
                    out_tensors[gname + ".qs"] = (gs.tobytes(), "F32", [moe_inter])
                    out_tensors[uname] = (uq.tobytes(), "U8", [moe_inter, len(uq) // moe_inter])
                    out_tensors[uname + ".qs"] = (us.tobytes(), "F32", [moe_inter])

                    # Flush periodically (every 16 experts) to limit memory
                    if (e + 1) % 16 == 0:
                        flush_shard()

                flush_shard()

            elif cls == "expert_down":
                # Fused [n_experts, hidden, moe_inter] → per-expert split + quantize
                n_experts = shape[0]
                hidden = shape[1]
                moe_inter = shape[2]
                print(f"  splitting {name}: [{n_experts}, {hidden}, {moe_inter}] → {n_experts} experts")

                parts = name.split(".")
                layer_num = None
                for i, p in enumerate(parts):
                    if p == "layers" and i + 1 < len(parts):
                        layer_num = parts[i + 1]
                        break

                for e in range(n_experts):
                    down_w = w[e]  # [hidden, moe_inter]
                    bits = args.ebits
                    prefix = f"model.language_model.layers.{layer_num}.mlp.experts.{e}"

                    if bits >= 5:
                        dq, ds = quant_int8(down_w, bits)
                    else:
                        dq, ds = quant_int4(down_w, bits)

                    dname = f"{prefix}.down_proj.weight"
                    out_tensors[dname] = (dq.tobytes(), "U8", [hidden, len(dq) // hidden])
                    out_tensors[dname + ".qs"] = (ds.tobytes(), "F32", [hidden])

                    if (e + 1) % 16 == 0:
                        flush_shard()

                flush_shard()

            print(f"  {cls:12s} {name:70s} {str(shape):>30s}")

            # Flush if accumulated output > 2GB
            total_pending = sum(len(d) for d, _, _ in out_tensors.values())
            if total_pending > 2e9:
                flush_shard()

    flush_shard()

    # Rename shards to correct count
    final_shards = sorted([f for f in os.listdir(args.outdir) if f.endswith(".safetensors")])
    n = len(final_shards)
    for i, old_name in enumerate(final_shards):
        new_name = f"model-{i+1:05d}-of-{n:05d}.safetensors"
        if old_name != new_name:
            os.rename(os.path.join(args.outdir, old_name), os.path.join(args.outdir, new_name))

    print(f"\nDone! Input: {total_in/1e9:.1f} GB → Output: {total_out/1e9:.1f} GB ({n} shards)")
    print(f"Expert bits: {args.ebits}, Dense bits: {args.dbits}, IO bits: {args.io_bits}")

if __name__ == "__main__":
    main()
