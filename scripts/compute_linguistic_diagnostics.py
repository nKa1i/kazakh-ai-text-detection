import json
import os
import re
import sys
from collections import Counter

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Core Kazakh commercial loanword roots frequently observed in KazSAnDRA
DEFAULT_LOANWORDS = {
    "доставка", "заказ", "каспи", "возврат", "скидка", "курьер",
    "упаковка", "брак", "чек", "магазин", "товар", "качество",
    "приложение", "сервис", "бонус", "оплата", "размер", "цвет"
}

def load_loanword_lexicon(lexicon_path: str = "data/code_switched_loanword_lexicon.json"):
    """Loads external loanword lexicon if present, combining with default set."""
    loanwords = set(DEFAULT_LOANWORDS)
    if os.path.exists(lexicon_path):
        try:
            with open(lexicon_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    loanwords.update(k.lower() for k in data.keys())
                elif isinstance(data, list):
                    loanwords.update(k.lower() for k in data)
        except Exception:
            pass
    return loanwords

LOANWORD_SET = load_loanword_lexicon()

def tokenize_words(text: str):
    """Clean Kazakh word tokenizer."""
    return re.findall(r"[a-zA-Zа-яА-ЯәіңғүұқөһӘІҢҒҮҰҚӨҺ]+", text.lower())

def calculate_token_inflation(text: str, tokenizer=None) -> float:
    """
    Computes Token Inflation Ratio (T/W): Total Subword Tokens / Total Whitespace Words.
    If a Hugging Face tokenizer is provided, uses its encode function;
    otherwise uses an empirical subword heuristic for Kazakh morphology.
    """
    words = text.strip().split()
    if not words:
        return 1.0

    if tokenizer is not None:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        return round(len(tokens) / len(words), 3)
    
    # Robust fallback empirical subword estimator for agglutinative Kazakh
    # (average subword split based on syllable and suffix boundary patterns)
    clean_words = tokenize_words(text)
    subword_count = 0
    for w in clean_words:
        # Monosyllabic / short stems (<4 chars) typically stay 1 token
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
    """
    Computes Type-Token Ratio (TTR), Distinct-1, and Distinct-2 scores across a corpus.
    """
    all_words = []
    bigrams = []
    
    for t in texts:
        words = tokenize_words(t)
        all_words.extend(words)
        if len(words) >= 2:
            for i in range(len(words) - 1):
                bigrams.append((words[i], words[i+1]))

    if not all_words:
        return {"ttr": 0.0, "distinct_1": 0.0, "distinct_2": 0.0}

    unique_words = len(set(all_words))
    ttr = round(unique_words / len(all_words), 4)
    distinct_1 = ttr
    distinct_2 = round(len(set(bigrams)) / max(1, len(bigrams)), 4) if bigrams else 0.0

    return {
        "ttr": ttr,
        "distinct_1": distinct_1,
        "distinct_2": distinct_2,
        "total_words": len(all_words),
        "unique_words": unique_words
    }

def detect_code_switched_loanwords(text: str) -> list:
    """
    Detects Russian/foreign commercial loanwords with Kazakh agglutinative suffixes
    (e.g., доставкасы, заказды, каспиден).
    """
    words = tokenize_words(text)
    detected = []
    
    for w in words:
        for root in LOANWORD_SET:
            if w.startswith(root):
                detected.append(w)
                break
    return detected

def compute_dataset_profile(records: list) -> dict:
    """
    Aggregates diagnostic metrics across generators and length brackets.
    """
    grouped_by_gen = {}
    for r in records:
        gen = r.get("generator", "unknown")
        if gen not in grouped_by_gen:
            grouped_by_gen[gen] = []
        grouped_by_gen[gen].append(r)

    summary = {}

    for gen, items in grouped_by_gen.items():
        texts = [x["text"] for x in items]
        lex_div = calculate_lexical_diversity(texts)
        tw_values = [calculate_token_inflation(t) for t in texts]
        loanword_counts = [len(detect_code_switched_loanwords(t)) for t in texts]
        
        # Length stratification
        by_bracket = {"short": [], "medium": [], "long": []}
        for x in items:
            b = x.get("length_bracket", "medium")
            if b in by_bracket:
                by_bracket[b].append(x)

        stratified_tw = {}
        for b, b_items in by_bracket.items():
            if b_items:
                stratified_tw[b] = round(sum(calculate_token_inflation(x["text"]) for x in b_items) / len(b_items), 3)
            else:
                stratified_tw[b] = 0.0

        summary[gen] = {
            "sample_count": len(items),
            "overall_ttr": lex_div["ttr"],
            "distinct_1": lex_div["distinct_1"],
            "distinct_2": lex_div["distinct_2"],
            "avg_tw": round(sum(tw_values) / max(1, len(tw_values)), 3),
            "stratified_tw": stratified_tw,
            "loanword_frequency": sum(loanword_counts),
            "loanwords_per_100_words": round(
                (sum(loanword_counts) / max(1, lex_div.get("total_words", 1))) * 100, 2
            )
        }

    return summary

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="data/seed_human_reviews_1k.json")
    parser.add_argument("--output_file", type=str, default="data/diagnostic_metrics_results.json")
    args = parser.parse_args()

    if os.path.exists(args.input_file):
        with open(args.input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        results = compute_dataset_profile(data)
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Computed diagnostic profile and saved to {args.output_file}")
