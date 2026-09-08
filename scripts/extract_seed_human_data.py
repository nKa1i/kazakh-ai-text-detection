import csv
import json
import os

def categorize_length(length: int) -> str:
    """
    Categorizes consumer reviews based on empirical KazSAnDRA length quantiles:
    - short: <= 60 characters (crucial false-positive boundary from AIST 2026 paper)
    - medium: 61 - 85 characters (modal review length)
    - long: > 85 characters (extended multi-sentence reviews)
    """
    if length <= 60:
        return "short"
    elif length <= 85:
        return "medium"
    return "long"

def extract_stratified_human_reviews(
    source_csv: str = "dataset_package/data/train.csv",
    output_json: str = "data/seed_human_reviews_1k.json",
    total_samples: int = 1000
):
    short_pool = []
    med_pool = []
    long_pool = []

    with open(source_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("label", "").strip() == "0":
                text = row.get("text", "").strip()
                if not text:
                    continue
                char_len = len(text)
                bracket = categorize_length(char_len)
                entry = {
                    "text": text,
                    "label": 0,
                    "generator": "human",
                    "char_length": char_len,
                    "length_bracket": bracket,
                    "domain": row.get("domain", "consumer_reviews")
                }
                if bracket == "short":
                    short_pool.append(entry)
                elif bracket == "medium":
                    med_pool.append(entry)
                else:
                    long_pool.append(entry)

    # 30% short, 45% medium, 25% long
    n_short = int(total_samples * 0.30)
    n_med = int(total_samples * 0.45)
    n_long = total_samples - n_short - n_med

    selected = short_pool[:n_short] + med_pool[:n_med] + long_pool[:n_long]
    for idx, item in enumerate(selected):
        item["id"] = idx

    output_dir = os.path.dirname(output_json)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)

    return selected

if __name__ == "__main__":
    data = extract_stratified_human_reviews()
    print(f"Extracted {len(data)} stratified human reviews into data/seed_human_reviews_1k.json")
