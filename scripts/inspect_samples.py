import json

with open("data/kazakh_aigc_paired_2k.json", "r", encoding="utf-8") as f:
    data = json.load(f)

human = [x for x in data if x["generator"] == "human"]
qwen = [x for x in data if x["generator"] == "qwen_2.5_7b"]

with open("data/sample_comparisons_utf8.txt", "w", encoding="utf-8") as f:
    f.write("=== SAMPLES COMPARISON ===\n\n")
    for i in range(5):
        b = human[i].get("length_bracket", "medium")
        f.write(f"--- Pair #{i+1} [{b.upper()}] ---\n")
        f.write(f"HUMAN ({human[i]['char_length']} chars): {human[i]['text']}\n")
        f.write(f"QWEN  ({qwen[i]['char_length']} chars): {qwen[i]['text']}\n\n")

print("Generated sample_comparisons_utf8.txt successfully!")
