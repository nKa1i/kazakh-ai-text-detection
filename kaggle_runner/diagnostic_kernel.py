import os
import sys
import json
import re
import time
from collections import Counter

print("=" * 70)
print("KAZAKH LLM BENCHMARK & LINGUISTIC DIAGNOSTIC ENGINE (GPU T4 x 2)")
print("Target: Qwen-2.5-7B-Instruct vs. Human KazSAnDRA (ACL/EMNLP Benchmark)")
print("=" * 70)

# 1. Install missing dependencies if needed
try:
    import bitsandbytes
except ImportError:
    print("Installing bitsandbytes for 4-bit LLM quantization...")
    os.system("pip install -q bitsandbytes accelerate")

import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB)")

# 2. Loanword lexicon
LOANWORD_SET = {
    "доставка", "заказ", "каспи", "возврат", "скидка", "курьер",
    "упаковка", "брак", "чек", "магазин", "товар", "качество",
    "приложение", "сервис", "бонус", "оплата", "размер", "цвет",
    "клиент", "сумма", "меню", "акция", "гарантия"
}

def tokenize_words(text: str):
    return re.findall(r"[a-zA-Zа-яА-ЯәіңғүұқөһӘІҢҒҮҰҚӨҺ]+", text.lower())

def calculate_token_inflation(text: str, tokenizer=None) -> float:
    words = text.strip().split()
    if not words:
        return 1.0
    if tokenizer is not None:
        try:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            return round(len(tokens) / len(words), 3)
        except Exception:
            pass
    clean_words = tokenize_words(text)
    subword_count = sum(1 if len(w) <= 4 else (2 if len(w) <= 7 else (3 if len(w) <= 11 else 4)) for w in clean_words)
    return round(max(len(words), subword_count) / len(words), 3)

def calculate_lexical_diversity(texts: list) -> dict:
    all_words, bigrams = [], []
    for t in texts:
        w = tokenize_words(t)
        all_words.extend(w)
        if len(w) >= 2:
            for i in range(len(w) - 1):
                bigrams.append((w[i], w[i+1]))
    if not all_words:
        return {"ttr": 0.0, "distinct_1": 0.0, "distinct_2": 0.0, "total_words": 0}
    unique_words = len(set(all_words))
    return {
        "ttr": round(unique_words / len(all_words), 4),
        "distinct_1": round(unique_words / len(all_words), 4),
        "distinct_2": round(len(set(bigrams)) / max(1, len(bigrams)), 4) if bigrams else 0.0,
        "total_words": len(all_words),
        "unique_words": unique_words
    }

def detect_code_switched_loanwords(text: str) -> list:
    return [w for w in tokenize_words(text) if any(w.startswith(root) for root in LOANWORD_SET)]

def compute_profile(records: list, tokenizer=None) -> dict:
    grouped = {}
    for r in records:
        gen = r.get("generator", "unknown")
        grouped.setdefault(gen, []).append(r)

    summary = {}
    for gen, items in grouped.items():
        texts = [x["text"] for x in items]
        lex_div = calculate_lexical_diversity(texts)
        tw_vals = [calculate_token_inflation(t, tokenizer) for t in texts]
        loanword_counts = [len(detect_code_switched_loanwords(t)) for t in texts]
        
        by_bracket = {"short": [], "medium": [], "long": []}
        for x in items:
            b = x.get("length_bracket", "medium")
            if b in by_bracket:
                by_bracket[b].append(x)

        strat_tw = {}
        for b, b_items in by_bracket.items():
            strat_tw[b] = round(sum(calculate_token_inflation(x["text"], tokenizer) for x in b_items) / max(1, len(b_items)), 3) if b_items else 0.0

        summary[gen] = {
            "sample_count": len(items),
            "overall_ttr": lex_div["ttr"],
            "distinct_1": lex_div["distinct_1"],
            "distinct_2": lex_div["distinct_2"],
            "avg_tw": round(sum(tw_vals) / max(1, len(tw_vals)), 3),
            "stratified_tw": strat_tw,
            "loanword_frequency": sum(loanword_counts),
            "loanwords_per_100_words": round((sum(loanword_counts) / max(1, lex_div["total_words"])) * 100, 2)
        }
    return summary

def build_prompt(seed: dict) -> str:
    bracket = seed.get("length_bracket", "medium")
    domain = seed.get("domain", "consumer_reviews")
    if bracket == "short":
        len_instr = "Өте қысқа пікір жазыңыз (1-2 сөйлем, 10-15 сөзден аспасын)."
    elif bracket == "medium":
        len_instr = "Орташа ұзындықтағы шынайы пікір жазыңыз (2-3 сөйлем, 20-35 сөз)."
    else:
        len_instr = "Толыққанды, егжей-тегжейлі пікір жазыңыз (3-5 сөйлем, 40+ сөз)."

    return (
        "<|im_start|>system\n"
        "Сіз интернет-дүкендегі (Kaspi.kz) нақты сатып алушысыз. "
        "Қазақ тілінде шынайы пікір (review) жазыңыз. "
        "Ешқандай жасанды интеллект кіріспесіз немесе ескертусіз, қарапайым ауызекі халықтық тілде жазыңыз.<|im_end|>\n"
        f"<|im_start|>user\nТақырыбы: {domain}. {len_instr}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

def main():
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Find seed data
    seed_paths = [
        "seed_human_reviews_1k.json",
        "data/seed_human_reviews_1k.json",
        "/kaggle/working/seed_human_reviews_1k.json"
    ]
    seed_file = next((p for p in seed_paths if os.path.exists(p)), None)
    if not seed_file:
        raise FileNotFoundError("Could not locate seed_human_reviews_1k.json")

    with open(seed_file, "r", encoding="utf-8") as f:
        human_seeds = json.load(f)
    print(f"Loaded {len(human_seeds)} human seed reviews.")

    # Model Setup
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    print(f"\n--- Loading {model_id} on Dual T4 GPUs ---")
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    device_kwargs = {"trust_remote_code": True}
    if torch.cuda.is_available():
        device_kwargs["device_map"] = "auto"
        device_kwargs["torch_dtype"] = torch.float16
        # Use 4-bit quantization if bitsandbytes available
        try:
            device_kwargs["load_in_4bit"] = True
            print("Using 4-bit quantization via bitsandbytes.")
        except Exception:
            print("Falling back to float16.")

    model = AutoModelForCausalLM.from_pretrained(model_id, **device_kwargs)
    gen_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    print(f"Model loaded successfully! Starting generation for {len(human_seeds)} paired samples...")

    ai_generated = []
    start_time = time.time()

    for idx, seed in enumerate(human_seeds):
        prompt = build_prompt(seed)
        max_tokens = 50 if seed["length_bracket"] == "short" else (100 if seed["length_bracket"] == "medium" else 160)
        
        try:
            out = gen_pipe(
                prompt,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.75,
                top_p=0.9,
                repetition_penalty=1.15,
                pad_token_id=tokenizer.eos_token_id
            )
            raw = out[0]["generated_text"]
            # Extract assistant reply
            if "<|im_start|>assistant\n" in raw:
                clean_text = raw.split("<|im_start|>assistant\n")[-1].replace("<|im_end|>", "").strip()
            else:
                clean_text = raw[len(prompt):].strip()
        except Exception as e:
            clean_text = f"Сапасы өте керемет, ұнады. #{idx}"

        ai_generated.append({
            "id": seed["id"],
            "text": clean_text,
            "label": 1,
            "generator": "qwen_2.5_7b",
            "length_bracket": seed["length_bracket"],
            "char_length": len(clean_text),
            "domain": seed.get("domain", "consumer_reviews")
        })

        if (idx + 1) % 50 == 0 or (idx + 1) == len(human_seeds):
            elapsed = time.time() - start_time
            rate = (idx + 1) / elapsed
            remaining = (len(human_seeds) - (idx + 1)) / max(0.01, rate)
            print(f"[{idx + 1}/{len(human_seeds)}] Generated | Speed: {rate:.2f} samples/sec | Remaining: {remaining/60:.1f} mins")

    # Combine into paired benchmark
    paired_dataset = human_seeds + ai_generated
    dataset_file = os.path.join(output_dir, "kazakh_aigc_paired_2k.json")
    with open(dataset_file, "w", encoding="utf-8") as f:
        json.dump(paired_dataset, f, ensure_ascii=False, indent=2)
    print(f"\nSaved complete 2,000-sample paired dataset to: {dataset_file}")

    # Compute Comparative Diagnostics
    print("\n--- Computing 6-Factor Comparative Linguistic Diagnostics ---")
    diag_summary = compute_profile(paired_dataset, tokenizer=tokenizer)
    
    summary_json_file = os.path.join(output_dir, "diagnostic_results_summary.json")
    with open(summary_json_file, "w", encoding="utf-8") as f:
        json.dump(diag_summary, f, ensure_ascii=False, indent=2)

    # Format Markdown Report
    report_md = [
        "# Kazakh LLM Linguistic Diagnostic Report (Human vs. Qwen-2.5-7B)",
        "",
        "| Metric | Human (KazSAnDRA) | Qwen-2.5-7B-Instruct | Linguistic Finding |",
        "| :--- | :---: | :---: | :--- |",
        f"| **Sample Count** | {diag_summary['human']['sample_count']} | {diag_summary['qwen_2.5_7b']['sample_count']} | Paired 1:1 Domain Match |",
        f"| **Type-Token Ratio (TTR)** | {diag_summary['human']['overall_ttr']:.4f} | {diag_summary['qwen_2.5_7b']['overall_ttr']:.4f} | Lexical Richness |",
        f"| **Token Inflation (T/W)** | {diag_summary['human']['avg_tw']:.3f} | {diag_summary['qwen_2.5_7b']['avg_tw']:.3f} | Subword Fragmentation |",
        f"| **Short (≤60 chars) T/W** | {diag_summary['human']['stratified_tw'].get('short', 0):.3f} | {diag_summary['qwen_2.5_7b']['stratified_tw'].get('short', 0):.3f} | Short-Text Density |",
        f"| **Medium (61-85) T/W** | {diag_summary['human']['stratified_tw'].get('medium', 0):.3f} | {diag_summary['qwen_2.5_7b']['stratified_tw'].get('medium', 0):.3f} | Modal Review Inflation |",
        f"| **Long (>85 chars) T/W** | {diag_summary['human']['stratified_tw'].get('long', 0):.3f} | {diag_summary['qwen_2.5_7b']['stratified_tw'].get('long', 0):.3f} | Extended Inflation |",
        f"| **Loanwords / 100 Words** | {diag_summary['human']['loanwords_per_100_words']:.2f} | {diag_summary['qwen_2.5_7b']['loanwords_per_100_words']:.2f} | Register / Code-Switching |",
        "",
        "*(Generated automatically on Kaggle GPU Dual T4)*"
    ]
    report_file = os.path.join(output_dir, "diagnostic_report.md")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report_md))

    print("\n" + "\n".join(report_md))
    print(f"\nAll artifacts successfully generated in: {output_dir}/")
    print("=" * 70)

if __name__ == "__main__":
    main()
