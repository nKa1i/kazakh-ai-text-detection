with open("kaggle_runner/embedded_seeds_b64.txt", "r", encoding="utf-8") as f:
    b64_data = f.read().strip()

with open("kaggle_runner/diagnostic_kernel.py", "r", encoding="utf-8") as f:
    content = f.read()

# Replace file reading with self-contained embedded string
import re
target_pattern = r"    # Find seed data[\s\S]+?print\(f\"Loaded \{len\(human_seeds\)\} human seed reviews\.\"\)"

replacement = f'''    # Self-contained embedded seed human reviews (1,000 samples from KazSAnDRA)
    import base64, zlib
    SEEDS_B64 = "{b64_data}"
    human_seeds = json.loads(zlib.decompress(base64.b64decode(SEEDS_B64.encode('ascii'))).decode('utf-8'))
    print(f"Loaded {{len(human_seeds)}} self-contained human seed reviews.")'''

new_content = re.sub(target_pattern, replacement, content)

with open("kaggle_runner/diagnostic_kernel.py", "w", encoding="utf-8") as f:
    f.write(new_content)

print("Embedded seed reviews successfully! Script length:", len(new_content))
