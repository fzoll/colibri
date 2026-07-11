"""Build a deterministic, medium-size GLM-4.x MoE fixture for backend benchmarks.

This is not a useful language model. It preserves the real Glm4Moe data flow
(GQA attention, MoE routing, partial RoPE) while remaining small enough to
generate locally and run repeated CPU/Metal A/B tests without downloading
the 60+ GB GLM-4.5-Air checkpoint.

Usage:
    python tools/make_glm4_bench_model.py --output glm4_bench_tiny
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import Glm4MoeConfig, Glm4MoeForCausalLM


def build_config() -> Glm4MoeConfig:
    return Glm4MoeConfig(
        vocab_size=8192,
        hidden_size=1024,
        intermediate_size=2048,
        moe_intermediate_size=512,
        num_hidden_layers=8,
        first_k_dense_replace=1,
        num_attention_heads=16,
        num_key_value_heads=4,
        head_dim=64,
        n_routed_experts=32,
        num_experts_per_tok=4,
        n_shared_experts=1,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
        rope_theta=10000.0,
        partial_rotary_factor=0.5,
        tie_word_embeddings=False,
        rms_norm_eps=1e-5,
        attention_bias=True,
        max_position_embeddings=4096,
        num_nextn_predict_layers=0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="glm4_bench_tiny")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    cfg = build_config()
    cfg._attn_implementation = "eager"
    model = Glm4MoeForCausalLM(cfg).eval()
    with torch.no_grad():
        for param in model.parameters():
            if param.dim() >= 2:
                param.normal_(0, 0.02)
        for layer in model.model.layers:
            if hasattr(layer.mlp, "gate"):
                layer.mlp.gate.e_score_correction_bias.copy_(
                    torch.linspace(-0.1, 0.1, cfg.n_routed_experts)
                )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    params = sum(p.numel() for p in model.parameters())
    model.save_pretrained(output, safe_serialization=True, max_shard_size="4GB")

    model.to(args.device)
    prompt = [3, 14, 159, 26, 53, 58, 200, 11, 77, 240, 5, 99]
    ids = torch.tensor([prompt], device=args.device)
    with torch.inference_mode():
        full = model.generate(ids, max_new_tokens=8, do_sample=False, use_cache=True)[0]
        logits = model(full.unsqueeze(0), use_cache=False).logits[0]

    ref = {
        "prompt_ids": prompt,
        "full_ids": full.cpu().tolist(),
        "tf_pred": logits.argmax(-1).cpu().tolist(),
    }
    (output / "ref_glm.json").write_text(json.dumps(ref))
    manifest = {
        "seed": args.seed,
        "parameters": params,
        "parameters_billions": round(params / 1e9, 4),
        "architecture": "Glm4MoeForCausalLM",
        "purpose": "backend benchmark fixture; random weights, not a language model",
    }
    (output / "bench_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
