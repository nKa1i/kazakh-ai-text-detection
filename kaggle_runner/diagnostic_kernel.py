import os
import sys
import json
import re
import time
from collections import Counter

print("=" * 70)
print("KAZAKH LLM LINGUISTIC DIAGNOSTIC & BENCHMARK ENGINE")
print("Target Venues: ACL / EMNLP / COLING & Master's Thesis")
print("=" * 70)

# Check PyTorch & GPU
try:
    import torch
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"VRAM Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
except ImportError:
    print("PyTorch not installed in current environment (will run with GPU on Kaggle).")
    torch = None

# 1. Loanword lexicon
LOANWORD_SET = {
    "доставка", "заказ", "каспи", "возврат", "скидка", "курьер",
    "упаковка", "брак", "чек", "магазин", "товар", "качество",
    "приложение", "сервис", "бонус", "оплата", "размер", "цвет",
    "клиент", "сумма", "меню"
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
    subword_count = 0
    for w in clean_words:
        if len(w) <= 4:
            subword_count += 1
        elif len(w) <= 7:
            subword_count += 2
        elif len(w) <= 11:
            subword_count += 3
        else:
            subword_count += 4
    subword_count = max(len(words), subword_count)
    return round(subword_count / len(words), 3)

def calculate_lexical_diversity(texts: list) -> dict:
    all_words = []
    bigrams = []
    for t in texts:
        words = tokenize_words(t)
        all_words.extend(words)
        if len(words) >= 2:
            for i in range(len(words) - 1):
                bigrams.append((words[i], words[i+1]))

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
    words = tokenize_words(text)
    return [w for w in words if any(w.startswith(root) for root in LOANWORD_SET)]

def compute_profile(records: list, tokenizer=None) -> dict:
    grouped = {}
    for r in records:
        gen = r.get("generator", "unknown")
        if gen not in grouped:
            grouped[gen] = []
        grouped[gen].append(r)

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
            strat_tw[b] = round(sum(calculate_token_inflation(x["text"], tokenizer) for x in b_items) / max(1, len(b_items)), 3)

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

def run_pipeline():
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load seed human data if available
    seed_path = "data/seed_human_reviews_1k.json"
    if not os.path.exists(seed_path):
        seed_path = "/kaggle/working/seed_human_reviews_1k.json"
    
    if os.path.exists(seed_path):
        with open(seed_path, "r", encoding="utf-8") as f:
            human_seeds = json.load(f)
        print(f"Loaded {len(human_seeds)} human seed reviews.")
    else:
        print("Creating baseline human seed reviews for Kaggle test run...")
        human_seeds = [
            {"id": i, "text": f"Керемет тауар, сапасы өте керемет, сатып алуға кеңес беремін #{i}", "length_bracket": "medium", "label": 0, "generator": "human"}
            for i in range(100)
        ]

    # Generate or evaluate
    print("\n--- Computing Baseline Linguistic Diagnostics on Human Reviews ---")
    human_profile = compute_profile(human_seeds)
    print(json.dumps(human_profile, indent=2, ensure_ascii=False))

    summary_file = os.path.join(output_dir, "diagnostic_results_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(human_profile, f, ensure_ascii=False, indent=2)

    print(f"\nDiagnostic pipeline completed successfully. Output saved to: {summary_file}")
    print("=" * 70)

if __name__ == "__main__":
    run_pipeline()
