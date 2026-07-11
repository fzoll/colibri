#!/usr/bin/env python3
"""Interactive chat with Gemma 4 via the Colibri engine.
Handles tokenization (HF tokenizers) and output decoding.

Usage:
    python3 tools/gemma4_chat.py --snap ./gemma4_26b_i8 [--vulkan] [--pin stats.txt --pin-gb 4]

Requires: pip install tokenizers
"""
import sys, os, json, subprocess, tempfile, argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snap", required=True, help="Model directory")
    parser.add_argument("--bits", type=int, default=8, help="Expert/dense bits (default 8)")
    parser.add_argument("--ngen", type=int, default=200, help="Max tokens to generate")
    parser.add_argument("--vulkan", action="store_true", help="Enable Vulkan GPU")
    parser.add_argument("--metal", action="store_true", help="Enable Metal GPU")
    parser.add_argument("--pin", default=None, help="PIN stats file for hot-store")
    parser.add_argument("--pin-gb", type=float, default=4, help="PIN GB budget")
    parser.add_argument("--engine", default="./gemma4", help="Engine binary path")
    args = parser.parse_args()

    from tokenizers import Tokenizer
    tok_path = os.path.join(args.snap, "tokenizer.json")
    if not os.path.exists(tok_path):
        print(f"Error: {tok_path} not found", file=sys.stderr)
        sys.exit(1)
    tok = Tokenizer.from_file(tok_path)

    print("Gemma 4 Chat (Colibri engine)")
    print("Type your message, press Enter. Type 'quit' to exit.\n")

    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not user or user.lower() in ('quit', 'exit', 'q'):
            break

        prompt = f"<start_of_turn>user\n{user}<end_of_turn>\n<start_of_turn>model\n"
        ids = tok.encode(prompt).ids

        full_ids = list(ids) + [0] * args.ngen
        tf_pred = [0] * len(full_ids)
        ref = {"prompt_ids": list(ids), "full_ids": full_ids, "tf_pred": tf_pred}

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(ref, f)
            ref_path = f.name

        env = dict(os.environ)
        env["SNAP"] = args.snap
        env["REF"] = ref_path
        if args.vulkan:
            env["COLI_VULKAN"] = "1"
        if args.metal:
            env["COLI_METAL"] = "1"
        if args.pin:
            env["PIN"] = args.pin
            env["PIN_GB"] = str(args.pin_gb)

        cmd = [args.engine, "8", str(args.bits), str(args.bits)]
        try:
            result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
        except subprocess.TimeoutExpired:
            print("[timeout]")
            continue
        finally:
            os.unlink(ref_path)

        # Parse output token IDs from engine output
        output_ids = []
        for line in result.stdout.split('\n'):
            if line.startswith("Motore C Gemma4"):
                parts = line.split(':')
                if len(parts) >= 2:
                    nums = parts[-1].strip().split()
                    output_ids = [int(x) for x in nums if x.isdigit() or (x.startswith('-') and x[1:].isdigit())]

        if output_ids:
            # Filter out padding zeros and stop tokens
            filtered = [x for x in output_ids if x > 0 and x != 1]
            text = tok.decode(filtered)
            print(f"\nGemma 4: {text}\n")
        else:
            # Try stderr for streaming output
            print(f"\n[No output parsed. stderr excerpt:]")
            for line in result.stderr.split('\n')[-5:]:
                print(f"  {line}")
            print()

if __name__ == "__main__":
    main()
