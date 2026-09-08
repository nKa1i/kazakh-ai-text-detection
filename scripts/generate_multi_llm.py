import argparse
import json
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.prompt_builder import build_generation_prompt

def generate_reviews(
    model_id: str,
    input_seeds_path: str = "data/seed_human_reviews_1k.json",
    output_path: str = "data/generated_qwen2.5_7b.json",
    device: str = "cuda",
    load_in_4bit: bool = True,
    dry_run: bool = False,
    max_samples: int = None
):
    """
    Generates synthetic Kazakh reviews using a specified open LLM conditioned
    on seed human review metadata. Supports both GPU execution and dry_run testing.
    """
    with open(input_seeds_path, "r", encoding="utf-8") as f:
        seeds = json.load(f)

    if max_samples:
        seeds = seeds[:max_samples]

    results = []

    if dry_run:
        print(f"[DRY RUN] Simulating generation for model: {model_id} on {len(seeds)} samples...")
        for seed in seeds:
            prompt = build_generation_prompt(seed)
            simulated_text = f"[{model_id} генерациясы]: {seed['text'][:40]}... сапасы керемет!"
            results.append({
                "id": seed["id"],
                "text": simulated_text,
                "label": 1,
                "generator": model_id,
                "length_bracket": seed["length_bracket"],
                "char_length": len(simulated_text),
                "domain": seed.get("domain", "consumer_reviews")
            })
    else:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

            print(f"Loading tokenizer and model: {model_id} (device: {device}, 4-bit: {load_in_4bit})...")
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            
            kwargs = {"trust_remote_code": True}
            if device == "cuda" and torch.cuda.is_available():
                kwargs["device_map"] = "auto"
                if load_in_4bit:
                    kwargs["load_in_4bit"] = True
                else:
                    kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            else:
                kwargs["device_map"] = "cpu"

            model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
            generator_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

            for idx, seed in enumerate(seeds):
                prompt = build_generation_prompt(seed)
                max_new_tokens = 60 if seed["length_bracket"] == "short" else (120 if seed["length_bracket"] == "medium" else 200)
                
                outputs = generator_pipe(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.eos_token_id
                )
                generated_full = outputs[0]["generated_text"]
                clean_text = generated_full[len(prompt):].strip()
                if not clean_text:
                    clean_text = generated_full.strip()

                results.append({
                    "id": seed["id"],
                    "text": clean_text,
                    "label": 1,
                    "generator": model_id,
                    "length_bracket": seed["length_bracket"],
                    "char_length": len(clean_text),
                    "domain": seed.get("domain", "consumer_reviews")
                })
                if (idx + 1) % 50 == 0 or (idx + 1) == len(seeds):
                    print(f"Progress: [{idx + 1}/{len(seeds)}] generated.")

        except Exception as e:
            print(f"Error during GPU generation: {e}. Falling back to dry_run output.", file=sys.stderr)
            raise e

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(results)} generated samples to {output_path}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-LLM Kazakh Review Generator")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Hugging Face Model ID")
    parser.add_argument("--input_seeds", type=str, default="data/seed_human_reviews_1k.json")
    parser.add_argument("--output_path", type=str, default="data/generated_sample.json")
    parser.add_argument("--dry_run", action="store_true", help="Simulate generation without loading heavy model")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples")
    args = parser.parse_args()

    generate_reviews(
        model_id=args.model_id,
        input_seeds_path=args.input_seeds,
        output_path=args.output_path,
        dry_run=args.dry_run,
        max_samples=args.max_samples
    )
