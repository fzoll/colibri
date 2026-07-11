"""Build a deterministic, tiny Gemma 4 MoE fixture for backend benchmarks.

This is not a useful language model. It preserves the real Gemma 4 data flow
(hybrid sliding/global attention, fused MoE experts, K=V, QK-norm, GELU)
while remaining small enough to test the engine without downloading 52 GB.

Usage:
    python tools/make_gemma4_bench_model.py --output gemma4_bench_tiny
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import Gemma4ForConditionalGeneration, Gemma4Config


def build_config() -> Gemma4Config:
    return Gemma4Config(
        text_config={
            "hidden_size": 512,
            "num_hidden_layers": 6,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "num_global_key_value_heads": 2,
            "head_dim": 64,
            "global_head_dim": 128,
            "intermediate_size": 1024,
            "moe_intermediate_size": 256,
            "num_experts": 16,
            "top_k_experts": 4,
            "vocab_size": 4096,
            "enable_moe_block": True,
            "hidden_activation": "gelu_pytorch_tanh",
            "sliding_window": 256,
            "attention_k_eq_v": True,
            "final_logit_softcapping": 30.0,
            "layer_types": [
                "sliding_attention", "sliding_attention", "full_attention",
                "sliding_attention", "sliding_attention", "full_attention",
            ],
            "rms_norm_eps": 1e-6,
            "rope_parameters": {
                "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
                "full_attention": {
                    "rope_theta": 1000000.0,
                    "rope_type": "proportional",
                    "partial_rotary_factor": 0.25,
                },
            },
            "tie_word_embeddings": True,
            "attention_bias": False,
            "first_k_dense_replace": 1,
            "max_position_embeddings": 4096,
            "bos_token_id": 2,
            "eos_token_id": 1,
            "pad_token_id": 0,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="gemma4_bench_tiny")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    cfg = build_config()
    cfg.text_config._attn_implementation = "eager"
    model = Gemma4ForConditionalGeneration(cfg).eval()
    with torch.no_grad():
        for param in model.parameters():
            if param.dim() >= 2:
                param.normal_(0, 0.02)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    params = sum(p.numel() for p in model.parameters())
    model.save_pretrained(output, safe_serialization=True, max_shard_size="4GB")

    model.to(args.device)
    prompt = [2, 14, 159, 26, 53, 58, 200, 11, 77, 240, 5, 99]
    ids = torch.tensor([prompt], device=args.device)
    with torch.inference_mode():
        full = model.generate(ids, max_new_tokens=8, do_sample=False, use_cache=True)[0]
        logits = model(full.unsqueeze(0), use_cache=False).logits[0]

    ref = {
        "prompt_ids": prompt,
        "full_ids": full.cpu().tolist(),
        "tf_pred": logits.argmax(-1).cpu().tolist(),
    }
    (output / "ref_gemma4.json").write_text(json.dumps(ref))
    manifest = {
        "seed": args.seed,
        "parameters": params,
        "parameters_billions": round(params / 1e9, 4),
        "architecture": "Gemma4ForConditionalGeneration",
        "purpose": "backend benchmark fixture; random weights, not a language model",
    }
    (output / "bench_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
